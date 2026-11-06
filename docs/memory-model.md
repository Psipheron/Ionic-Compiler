# Ionic Memory Model

How memory is allocated, shared, and reclaimed — and the long-term plan for
eliminating leaks without a tracing collector.

---

## 1. Current state (as shipped)

Ionic today has **no garbage collector and no linear lifetime system**, and
reference counting that is **partial**: strings are counted and reclaimed at
function scope, arrays and structs are not counted at all, and the only explicit
free is `model_free`. What follows describes the shipped baseline; §8 and §9 track
which parts of the target model have landed.

- **Strings** are `malloc`'d C strings (`char *`), treated as immutable `const`
  values, and every producer routes through `ionic_rc_alloc` so the block carries
  an 8-byte count (§4.1). Codegen raises and discharges a claim per string slot
  (§4.2), so a string whose owner goes out of scope is freed.
- **Structs** are heap-allocated with `malloc` at construction
  (`src/codegen/native.ionic`, `ND_STRUCT_LIT`) and are **not** counted.
- **Arrays** use a 24-byte header `[len:i64][cap:i64][data_ptr]` plus a separate
  `malloc`/`realloc`'d data region (`IonicArray`, `src/ionic_model_runtime.c`) and
  are **not** counted. `ionic_arr_reset` only sets `len = 0` (§11.4).
- **The only explicit free** is `model_free(m)`, which releases ML model
  resources (ONNX Runtime handles) via `ionic_model_free`.

The policy is stated in the runtime, and it is now true only of the uncounted
kinds:

```c
return buf; /* caller owns — but Ionic treats strings as const; leak is acceptable for now */
```

`src/ionic_model_runtime.c`

Consequence, measured rather than assumed: a program that churns **strings**
inside a function is flat in iteration count (`scripts/rss_check.sh` reports 0
bytes growth across N=200000..800000), while one that churns **arrays or
structs** still grows without bound because those are not counted (§11.4). The
compiler's own peak came from string copying and is fixed (§8 step 3); it still
allocates every aggregate it builds.

---

## 2. Constraints that shape the design

The memory model must respect the language's defining traits, not fight them.

- **Static types, but the backend's type tracking was incomplete.** The native
  backend tracks which locals and globals hold strings vs. floats vs. structs
  (`local_is_str`, `local_is_float_slot`, `local_get_stype`, `glob_is_str`), and
  that was *not* sufficient on its own: string-ness was lost on array element
  loads, on parameters, and on several expression paths (see §10). A **type
  oracle**, `ncg_ty_of`/`ncg_elem_ty` in `src/codegen/native.ionic`, now answers
  the question directly. It is deliberately conservative: unknown answers are
  distinguishable from "int64", which is what keeps an ownership pass sound.
- **GPU placement (`@cpu` / `@gpu`).** Tensor and model memory crosses a device
  boundary. Reclamation of device memory must be *deterministic* — a delayed
  collector pause during inference is unacceptable.
- **C / library interop** (ONNX Runtime, POSIX sockets) hands out
  raw pointers that are indistinguishable at runtime from managed pointers.
- **A shared ABI of bare pointers.** A string is one `i64`; there is no header or
  type tag at runtime. Foreign pointers and compile-time literals are both
  `char *`.
- **No stack maps.** The native ARM64 backend emits no metadata describing which
  stack slots are pointers, so a precise tracing collector has no roots to walk.

---

## 3. Decision

Adopt **deterministic ownership** as the language memory model:

1. **Non-atomic reference counting** for shared, immutable data — strings, array
   data, and tensor storage.
2. **Unique ownership + RAII** for resources with scarce, device-affine lifetimes
   — `model`, GPU buffers, file/socket handles.
3. **Arena scopes** for compiler-internal temporaries.
4. **Cycles: no collector.** Tracing GC and cycle collectors are both rejected
   (see §5). Strings and tensor storage are immutable, so no cycle can form over
   them and refcounting is complete for both. Mutable aggregates *can* form
   cycles, and a cyclic aggregate is an **accepted, documented leak** — decided
   in §6.1, not an open question. Nothing walks the heap and nothing pauses.

The three mechanisms share one property: reclamation is *deterministic* and
*local*. Nothing walks the heap, nothing pauses, and nothing needs to understand
a foreign pointer to be correct.

---

## 4. Mechanism

### 4.1 Reference counting (shared immutable data)

- **Header.** Each object carries an 8-byte refcount immediately before its
  payload. `rc_alloc(n)` returns `payload + 8`.
- **Operations (single-threaded, non-atomic):**
  - `rc_retain(p)` → `*(int64_t *)(p - 8) += 1`
  - `rc_release(p)` → `if (--*(int64_t *)(p - 8) == 0) free(p - 8)`
- **Literals.** Either (a) statically exclude `ND_STR` literals from any
  retain/release (they are known at every use site), or (b) materialize them once
  through a runtime wrapper (`ionic_str_lit`) as *pinned* managed copies keyed by
  rodata address, so literals participate uniformly and `release` on a literal is
  a safe no-op. Option (b) proved more robust when the codegen could not reliably
  classify a value's string-ness at every site (see §10). With `ncg_ty_of` in
  place, a literal is provably a string at every use site, so (a) becomes viable
  again — but (b) is already the safer default.
- **Runtime builtins route through `rc_alloc`** so the objects they return begin
  life at refcount 1: `ionic_str_concat`, `ionic_str_join`, `ionic_int64_to_str`,
  `ionic_float64_to_str`, `ionic_str_slice`, `ionic_str_replace`, `ionic_format`,
  `ionic_char_to_str`, `ionic_read_line`, `ionic_file_read`, `ionic_http_get`,
  `ionic_http_post`, `ionic_http_body`, `ionic_http_urlencode`, `ionic_bpe_decode`,
  and the two shared helpers `rt_empty_str` / `rt_dup_str` (which are how
  `mlx_weights_key_at` and the three `ionic_http_*` request accessors get an owned
  block). Every one of these now allocates on every path — including the failure
  paths, which is what §11.1 was about.
  **Not** `ionic_get_arg`, `ionic_getenv`, `ionic_exe_dir`, or
  `ionic_mlx_last_error` — those return process-, libc-, or statically-owned
  buffers and must never be instrumented. The three `ionic_http_*` *request*
  accessors belong on this list too: they now copy out of the `HttpRequest`, so
  they are fresh allocations and safe to instrument (§11.2). See §11.3; an
  earlier version of this list wrongly included `get_arg` and `getenv`.
  **Not `ionic_array_alloc` either**, despite an earlier draft of this list naming
  it: an array is a *two-allocation* object (`{len, cap, data}`), so a header on
  the `IonicArray *` would not cover the element buffer. Arrays need their own
  release path (§11.4) and are not part of step 4's string-only stage.

### 4.2 Insertion points (compile time)

Using the existing per-slot type tracking to know which slots own a reference:

- **retain** when a counted value is copied into another owning slot:
  assignment, struct-field store, array element store / `push`, and `return`
  (the callee's reference becomes the caller's).
- **release** at **end of block** for each owning local, and on **reassignment**
  of an owning slot (release old, retain new).
- **release** the caller's temporary references to arguments after a call
  returns, unless ownership was transferred.

### 4.3 Unique ownership / RAII (resources)

`model` and future handle-like types (`tensor` device buffers, file/socket
handles) are **affine**: owning one means exactly one live reference. They get a
deterministic destructor rather than a refcount. This preserves the current
explicit `model_free` behaviour while allowing it to become implicit at
end-of-scope.

### 4.4 Arena scopes (compiler internals)

The compiler is phase-structured — lex → parse → check → optimize → emit. Each
phase gets an arena that is reset when the phase ends. This reclaims the
compiler's own transient allocations in bulk, with no per-object accounting and
no semantic change to the language.

---

## 5. Why not the alternatives

- **Tracing GC (mark–sweep / copying / generational):** no stack maps for root
  discovery; unsafe and unsound across the GPU boundary; introduces stop-the-world
  pauses that hurt both inference latency and the ~3 s compiler build. A *moving*
  collector is impossible with C interop, since foreign raw pointers cannot be
  updated.
- **Full Rust-style borrow checking:** correct and leak-free, but a type-system
  rewrite with lifetimes and move semantics that would invalidate existing Ionic
  programs. Not proportionate to the language's goal.
- **Region/arena inference for all values:** fastest to allocate, but *unsound*
  here — values escape into returns, struct fields, globals, and arrays. Viable
  only for temporaries (see §4.4).
- **RC plus a cycle collector (trial deletion):** would make the runtime
  unconditionally leak-free, but adds a mark phase and full object enumeration.
  It was rejected first on the grounds that the type system prevents cycles
  outright, and then rejected on its own merits once §11.6 showed it does not:
  a collector contradicts the determinism-and-locality premise this whole model
  rests on (§3). The cycle question is settled as an accepted leak (§6.1), not
  deferred to a collector.

---

## 6. Cycles: the intended invariant, and where it does not hold

> **Correction (§11.6).** This section previously claimed cycles were structurally
> impossible. They are not: two names can alias one mutable aggregate
> (`let q = p`) and aggregates can contain aggregates (`a.push(b)`), so a
> reference cycle is constructible from ordinary Ionic. Read this section as the
> *intended* design and §11.6 for the shipped behaviour.
>
> **Decision (step 7, §8): the invariant is formally abandoned for aggregates.**
> Refcounting reclaims acyclic aggregate graphs only; cyclic ones leak, and that
> leak is **accepted and documented** rather than prevented at the type level.
> Strings and tensor storage have no such exemption — both are immutable, so no
> cycle can form over them and their graphs stay DAGs. See §6.1 for the decision,
> the evidence, and the exact scope of what is accepted.

A cycle collector is only necessary if a cycle can form. As designed, Ionic's
data model was meant to forbid it:

- **Shared data is immutable** (strings, tensor storage). You cannot create a
  cycle by mutating shared immutable data.
- **Mutable ownership is unique / affine.** A `mut` value was meant to have
  exactly one owner, with no aliasing through which a back-edge could be built.

Under those two rules every reference graph is a DAG, and refcounting alone
reclaims it completely — which is what would make Ionic leak-free *without* a
general-purpose collector.

The first rule holds. The second does not, and cannot be made to hold without
removing capabilities the language is built around:

- `let q = p` copies a pointer rather than moving it, for both structs and
  arrays, so two names alias one mutable aggregate.
- Structs and arrays can contain aggregates *of their own type*, so the
  back-edge does not need two objects or two names — one object can point at
  itself.

### 6.1 The decision (step 7)

**Accepted: cyclic aggregates leak.** No cycle collector is planned, and the
language makes no promise to reclaim a cyclic aggregate graph. This is a
deliberate, bounded, programmer-visible leak, in the same category as a `push`
that drops an ownership claim (§4.2) and the two known gaps noted in §9.

Scope, stated precisely so it can be tested rather than assumed:

- **What is exempt.** Any aggregate (`struct`, `[T]`, or array-of-struct)
  reachable from a cycle. When such a graph loses its last external reference,
  its members' counts never reach zero and none of the memory returns.
- **What is not exempt.** Strings and tensor storage. Both are immutable, so a
  cycle cannot form over them and refcounting is complete for them. This is why
  `scripts/rss_check.sh` — a string-churn benchmark — is a valid hard gate
  (§8 step 4), and why a flat RSS result there is a statement about strings, not
  about aggregates.
- **The obligation this creates.** Any benchmark guarding aggregate
  reclamation must be **cycle-free**, or it will never go flat and its failure
  will be indistinguishable from "aggregate RC is not implemented yet". §8
  step 4 stage 2 must therefore land aggregate RC with its own acyclic
  benchmark, and this document must not point at a cyclic one.

**Rejected: banning aggregate-in-aggregate at the type level (Option B).** It was
the alternative on the table and it does not work, for a reason worth recording
because it invalidates the earlier framing in this document and in §11.6:

> The ban would have to cover a struct with a field of *its own* type, not just
> nested containers. `struct Node { val: int64, next: Node }` is legal today, and
> `n.next = n` produces a traversable self-cycle with **no array and no second
> object involved**:

```ionic
struct Node { val: int64, next: Node }
let m = Node { val: 2 };
let n = Node { val: 1, next: m };
n.next = n;
n.next.val;          // 1  — n.next is n
n.next.next.val;     // 1  — and still is, unboundedly
```

A rule phrased as "aggregates may not contain aggregates" therefore forbids
recursive structs — trees, lists, any self-referential shape — which is a much
larger language restriction than "cycles leak". Narrower rules did not survive
either: permitting recursion but rejecting cycles needs either a static
must-not-alias proof or an occurs-check on value identity, and neither is
available in a language where every aggregate assignment aliases. Rejecting only
`[T]` inside `[U]` would fix nothing, because the cycle routes through a struct
just as easily.

**Rejected: a cycle collector.** The whole model is premised on reclamation being
deterministic and local (§3): nothing walks the heap and nothing pauses. A
collector contradicts the property the memory model exists to provide, and the
leak it would remove is bounded and rare in this language's domain code (see
§11.6).

**What is unchanged.** Aliasing of mutable aggregates (`let q = p`; writes
through `q` visible through `p`) stays exactly as it is. It is a consequence of
aggregates being single-word heap pointers, it is relied on by the compiler's own
IR-building code, and it is orthogonal to the cycle question — it only *enables*
cycles, it does not require them. Changing it to a value-copy or a move is not
part of this decision and would be a separate language-semantics change.

---

## 7. Consequences

**What improves** (once the remaining stages land — see §8/§9 for what is live)

- Transient strings are freed when their owner dies — the dominant leak in user
  programs. **This is live today** (`rss_check.sh` is flat).
- Transient *arrays and structs* are freed when their owner dies — **not yet**:
  they are uncounted (§11.4).
- Device/model memory is released deterministically, never at the whim of a
  collector pause. (`model` RAII is live; tensor storage RC is step 5.)
- The compiler's own peak is bounded (§8 step 3) — by eliminating the quadratic
  copy, not by an arena. The arena in §4.4 was never built and its ordering
  question is still open.

**What RC does not solve**

- **Reference cycles among counted objects.** Constructible today (§11.6), and the
  objects in a cycle leak. **Settled as an accepted leak** (§6.1): no type-level
  restriction, no collector. Strings and tensor storage are unaffected — they are
  immutable, so a cycle cannot form over them.
- **C-runtime buffers not routed through `rc_alloc`.** Anything `malloc`'d inside
  `src/ionic_model_runtime.c` (for example the `piper_forward` samples array, the
  `http_get` body) is invisible to the scheme until it is re-routed. These are
  fixed individually.
- **Interior / foreign pointers.** A pointer stashed where the compiler cannot see
  it (a C callback, a cache) is not reflected in any refcount. This is the main
  soundness risk and the reason a type-level owned-vs-borrowed distinction
  (§8, step 6) is the durable answer. §11.2 and §11.3 show this is not
  hypothetical: four runtime accessors still return process- or statically-owned
  pointers that the checker types as ordinary strings. The interior cases that
  used to be listed first were made safe by copying.

**Cost**

- 8 bytes per counted object.
- One increment per reference copy (non-atomic — no `lock` prefix).
- A branch and possible `free` on release.
- The dominant risk during rollout: a single misplaced `release` turns a benign
  leak into a use-after-free. Mitigation is incremental landing with verification
  at each step.

---

## 8. Sequencing

Landed incrementally, lowest risk first, each step verified with
`./build.sh --verify-self-hosting`.

0. **Allocation audit (§11).** Establish which runtime entry points can legally
   be refcounted. **Done** — and it changed the order of everything below: the
   interior-pointer cases (§11.2) turned out to be a design blocker, not an
   implementation detail.
1. **Make every string return path allocate (§11.1).** **Done.** Five failure
   paths were returning a rodata literal or the caller's own input; all now
   allocate, so "returned a `TY_STR` ⇒ the result is owned" holds. Both the C
   runtime and the inline LLVM IR definition of `ionic_file_read` were fixed.
2. **Resolve the interior/static accessors (§11.2, §11.3).** **Done for the
   interior class.** The three `http_*` request accessors and
   `mlx_weights_key_at` now copy on access, so they return owned blocks. Four
   foreign/static sources remain (`get_arg`, `getenv`, `exe_dir`,
   `mlx_last_error`); the retain/release pass must special-case those four by
   name until step 6 replaces the list with a type-level distinction.
3. **Compiler-internal peak elimination.** Immediate, self-contained, no language
   semantics change, and it does not depend on 1–2. **Done, but not as an arena.**
   An arena reclaims memory *between* phases, and measuring first showed the
   blowup was not a phase-lifetime leak: it was O(lines × file size) copying
   inside a single phase. At constant byte count, RSS tracked line count — a
   328 KB input as 4 lines cost 42 MB, the same bytes as 8000 lines cost 798 MB,
   and 30k statements in one function cost 192 MB. The cause was
   `acc = str_concat(acc, str_concat(line, "\n"))` in `imp_expand`, plus the same
   shape in the driver's fragment join: because `str_concat` allocates and
   nothing frees, each line re-copied the whole accumulated prefix.

   The fix is `str_join`, a linear bulk join that takes a `[string]` and the
   separator, allocated once. Both call sites now collect fragments and join
   them in one pass. The compiler's own source went from a 397 MB peak to 47 MB,
   and RSS is now flat in line count. An arena would not have moved either
   number, so the ordering in this list stands: the quadratic had to go first,
   and the arena question is separate.

   This did require a language surface addition (`str_join`), which §8 was not
   expecting — Ionic strings are immutable and there is no builder type, so a
   linear join cannot be expressed from existing builtins at all. The committed
   `ionic_self` had to be refreshed for the same reason: it cannot compile
   source that calls a name it does not know.
4. **RC for strings and arrays.** Non-atomic. Literals are wrapped as pinned
   managed copies. Fixes the dominant user-program leak.

   **Stage 1 of 3 — DONE (primitives, no emission).** `src/ionic_model_runtime.c`
   now has the machinery and nothing calls it:

   - `ionic_rc_alloc` / `_alloc_pinned` / `_retain` / `_release` / `_realloc`,
     with the count in an 8-byte header *before* the payload. That layout is the
     whole reason the step is cheap: the ABI value is still one `i64`, so no call
     site, struct, or array slot changes shape.
   - `IONIC_RC_PINNED` (`-1`) as the "never freed" count. Release on a pinned
     object is a no-op, which is how literals and foreign pointers participate
     uniformly instead of needing an exemption at every use site.
   - `ionic_str_lit`, which interns a literal's *pinned managed copy* by its
     rodata address — codegen already interns each literal node to a fixed
     `__cstring` offset, so the address is a stable key and costs nothing to
     compare.
   - Every producer allocates through `ionic_rc_alloc`: the two `rt_*` helpers,
     `read_line`, `str_slice`, `str_replace`, `str_concat`, `str_join`,
     `int64_to_str`, `float64_to_str`, `format`, `char_to_str`, `file_read`,
     `http_get`/`http_post`/`http_body`/`http_urlencode`, `bpe_decode`, and
     `mlx_weights_key_at` via `rt_dup_str`. Producers that *grow* a buffer
     (`http_request`, `bpe_decode`) go through `ionic_rc_realloc`, because a
     plain `realloc(payload)` frees the header.

   Output is unchanged: `./build.sh --verify-self-hosting` is byte-identical at
   542916 bytes, and the only compiler warnings are `ionic_rc_retain` and
   `ionic_rc_release` being unused — the correct state for a stage that emits
   nothing.

   **The producer contract is a gate (`scripts/rc_check.sh`).** Stage 1's
   weakness is that it is unobservable: nothing reads the header, so a producer
   that quietly kept a raw `malloc` would compile, run, and pass every other
   check in the repo. `scripts/rcprobe.c` `#include`s the runtime TU (the rc
   primitives are `static`) and asserts, under ASan+UBSan, that every producer
   returns `rc == 1`, including on its failure paths — plus pinned-literal
   interning and a no-op release on one, `retain`/`release` balance, and
   `rc_realloc` over both a live and a pinned block. Coverage includes the three
   `http_*` request accessors and `bpe_decode`, i.e. exactly the functions that
   used to be in the interior and foreign/static classes, so the copy-on-access
   fix (§11.2) is checked rather than assumed. It is wired into CI as a hard gate
   ahead of step 2, because that is when a missed producer turns into `free()`
   eight bytes into a `malloc` block. Verified to fail on both shapes of
   regression: a producer reverted to `malloc` (ASan reports the header load at
   the producer's line) and one returning a shared `rc=2` block (the `rc == 1`
   assertion).

   **What stage 1 did *not* change.** Peak RSS is unchanged, and the "48 → 80
   bytes/iteration" figure that looked like an 8-byte-header regression was a
   measurement error: 48 is a *one*-allocation benchmark (`s = int64_to_str(i)`)
   and 80 is the shipped *two*-allocation one (`int64_to_str` + `str_concat`).
   Same commit, same compiler, both figures reproduce exactly, and pre-header
   `b0fc5a4` measures 80 too. The header rides inside malloc's existing size
   classes at these sizes, so it is free until something actually frees — which
   is the point of stage 2.

   **The Rust/IR divergence is CLOSED — one implementation, one layout.**
   `src/codegen.rs` used to `define` five producers (`ionic_str_concat`,
   `int64_to_str`, `float64_to_str`, `char_to_str`, `file_read`) *inline* in LLVM
   IR, each with a raw `malloc`. Because the C definitions are
   `__attribute__((weak))`, the IR versions won at link time, so a string built
   by the Rust bootstrap had no refcount header. That was harmless only while
   nothing read one.

   They are now *declared* there, not defined, and the calls resolve to the C
   runtime. The runtime object is always linked (`src/main.rs`), so this needs no
   new link step; `nm` confirms the symbols are undefined in the emitted `.o` and
   the header is therefore whatever the runtime says rather than a second copy of
   the arithmetic. Re-deriving the layout in IR was rejected precisely because two
   copies is how they drifted — and they *had* drifted: the IR `%lf` returned
   `2.500000` where the C `%g` returned `2.5`, so the same program printed
   differently depending on which compiler built it. `println`'s float format had
   drifted the same way and is fixed with it.

   Reverting to a definition here would silently reintroduce the
   no-header-on-Rust-payloads bug, so the *declarations* are the invariant, not an
   optimisation.

   **Stage 2 — emission — PARTLY DONE (strings).** The gate is
   `ncg_ty_of() == OTY_STR` and explicitly *not* `OTY_OTHER`, so a release is never
   emitted for an unclassified pointer. `get_arg`, `getenv`, `exe_dir` and
   `mlx_last_error` are exempted by name (§8 step 2) until step 6 makes the
   distinction type-level.

   Landed: per-slot claims on `let`/`=`/`return`/exit (§4.2, first two insertion
   points) and reclamation of argument temporaries (§4.2 third insertion point).
   Measured: `scripts/rss_check.sh` reports 0 bytes growth in N. Not yet landed:
   array element stores and `push` (arrays are uncounted, §11.4), and the two
   documented gaps in §9 — a loop-re-executed `let`, and a producer temp passed to
   a *user* function.

5. **RC for tensor storage + RAII for `model`/handle types.** The domain core:
   deterministic release of scarce, device-affine memory. *(The `model` RAII half
   is implemented — see §9.)*
6. **Type-level `owned` vs. `borrowed` distinction.** Makes the ownership
   discipline explicit in the type system rather than a runtime convention, and
   is the correct handling of foreign pointers from C/library interop.
7. **Decide the cycle question (§11.6). — DECIDED.** Amend §6 to accept bounded
   leaks for cyclic aggregates. The alternative (prevent aggregate-in-aggregate at
   the type level) was rejected because the cycle does not require a nested
   container: a struct with a field of its own type is legal, and `n.next = n` is
   a traversable self-cycle, so the ban would forbid recursive structs outright.
   No cycle collector. See §6.1 for the decision and §11.6 for the evidence.

   This unblocks step 4's gate: `rss_check.sh` is a valid hard gate because its
   benchmark is string churn, and strings have no cycle exemption.

---

## 9. Status

This document records the decision and plan.

- **Slice 3 (`model` RAII) — IMPLEMENTED.** The native backend now tracks which
  locals own a `model` resource and emits `ionic_model_free` automatically at
  function end, on reassignment of an owning local, and skips it when ownership is
  transferred by `return`. An explicit `model_free(m)` also clears the local's
  ownership, so it is not freed a second time at scope exit. Explicit `model_free`
  remains available. Zero effect on programs that do not use `model`
  (byte-identical object output).

  Two invariants make the emitted code correct, and both are covered by
  `scripts/model_raii_check.sh`:

  1. **One destructor emit point.** Every `return` stores its value and branches
     to a single exit block; the destructor runs there and nowhere else. Emitting
     it both at the `return` and at the fall-through epilogue (the original
     design) produced two `ionic_model_free` references per model, i.e. a double
     free.
  2. **The return value is parked across the destructor.** `ionic_model_free`
     takes the handle in `X0` and clobbers every caller-saved register, so a
     function that owns a model spills its return value to a dedicated frame slot
     before the free and reloads it afterwards. Without that, `return 42;` in
     such a function returned whatever the free left in `X0`.

  **Known limitation — ownership transfer is path-insensitive.** `return m`
  clears `m`'s ownership in compile-time state that every runtime path shares.
  A function that transfers a *different* model from each of two branches
  therefore leaks: the first `return` clears its local, and the second path
  still owns a handle the exit block no longer frees.

      let a = load_model(p); let b = load_model(p);
      if (c > 0) { return a; }   // frees b, clears a
      return b;                  // clears b; a is never freed

  Making this sound needs a per-path live set, or per-`return` destructors in a
  reachability-aware exit block — neither fits the current single-exit design.
  (Emitting a destructor at each transferring `return` was tried and reverted:
  it widens the static free count without fixing the leak above.) The supported
  shapes are one owner, one transfer point, and several owners with no transfer.
  See the `ND_RETURN` note in `src/codegen/native.ionic`.
- **Slice 2 (RC for strings/arrays) — prototyped, REVERTED, then RELANDED in stages.**
  See §10 for the implementation log and the prerequisite it exposed. Stage 1
  (primitives, no emission) is in §8 step 4. **Stage 2 (emission) has partly
  landed**, for strings only:
  - `let` and `=` raise, retain and release a claim per slot; `return` transfers;
    the single exit block discharges whatever is left. See §4.2.
  - **Argument temporaries are reclaimed.** A producer result passed straight into
    a copying builtin (`s = str_concat(int64_to_str(i), "-x")`) has no slot to be
    discharged from, so nothing used to release it. Nine frame slots per function
    stash one pointer per ABI argument and park the call's result across the
    releases; the stash sits between argument evaluation and the `bl`. Only
    `ncg_str_copies_args` callees are eligible, because a user function may retain
    the pointer past the return. This is §4.2's third insertion point, and it is
    what took `scripts/rss_check.sh` from ~48 B/iter to **0 bytes growth** across
    N=200000..800000.
  - **Two gaps remain, both documented at the point of emission** in
    `src/codegen/native.ionic`: a `let` re-executed by a loop leaks one block per
    iteration (releasing on re-entry is unsound under the single-exit design,
    because the slot may hold a `getenv` borrow left by another path), and a user
    function receiving a producer temp still leaks by design.
  - Arrays are still not counted at all (§11.4), so this is a string-only stage.
- **Cycle question (§8 step 7) — DECIDED: accepted leak.** See §6.1. Cycles over
  mutable aggregates are constructible and now formally exempt; strings and tensor
  storage are unaffected. No collector, no type-level ban.
- **Prerequisite for Slice 2 — the type oracle, step 1 SHIPPED.** `ncg_ty_of` /
  `ncg_elem_ty` answer "int64 / string / float64 / none of those" for an
  expression and for an array element, and print dispatch now consults them. That
  closed the observable `println` defects: array elements (including an array
  *returned* from a function, whose element type the checker was losing at the
  boundary), string parameters, struct fields, and `.len`. It stays deliberately
  conservative — a nested-array element and a method call are still `OTY_OTHER` —
  and the remaining widening is described in §10.
- **Allocation audit (§11) — COMPLETE, both bug classes fixed.** Every pointer
  the runtime hands to a program is classified as fresh, interior,
  foreign/static, or caller-buffer. §11.1 (the alias returns) and §11.2 (the
  interior request accessors) are both resolved: the second by copy-on-access,
  which is what the audit said was preferred. Nothing in §11 now blocks the
  retain/release pass.

## 10. Implementation log — the Slice 2 attempt and its prerequisite

Slice 2 (RC for strings) was prototyped end-to-end and then **reverted**, because
the native backend lacks a prerequisite. The findings are recorded here so the
next attempt starts from them.

**What was built and verified**

- Runtime primitives: `ionic_rc_alloc` (8-byte refcount header before the
  payload), `ionic_rc_retain`, `ionic_rc_release`, `ionic_rc_realloc`.
- Every managed string producer routed through `ionic_rc_alloc`
  (`str_concat`, `int64_to_str`, `float64_to_str`, `str_slice`, `str_replace`,
  `format`, `char_to_str`, `read_line`, `file_read`, `get_arg`, `getenv`,
  `http_*`). This keeps the runtime self-consistent and the compiler `.o` unchanged.
- A literal wrapper, `ionic_str_lit`, that returns a *pinned* managed copy of a
  string literal keyed by its rodata address, so literals can participate
  uniformly and `release` on a literal is a safe no-op.
- Codegen ownership rules: string-typed locals own one reference; `return`
  transfers (retain before move to `X0`, release all owning locals before the
  epilogue); reassignment releases the old value then retains the new
  (retain-new-first so `x = x` is safe); `let b = a` retains; array-element and
  struct-field string stores retain; discarded string-call statements release.

**Measured result when it worked**

A function-local churn loop (`s = str_concat(int64_to_str(i), "-x")` × 2,000,000)
dropped from **160 MB to 1.6 MB** peak RSS — the leak was eliminated for that case.

**The blocker: the codegen cannot reliably classify a value as a managed string**

`ionic_rc_release` fired on non-managed pointers (stack/rodata addresses). The
insertion logic gates on "is this expression a string?", and that predicate is
not knowable from the current backend state:

1. **Local name lookup inconsistency** — `local_find` returns the *last* slot for
   a name (shadowing) while `local_get_str` returned the *first*. If a string
   parameter `s` is shadowed by an integer local `s`, the integer was treated as
   a string. (Fixed to last-match, but only one instance of the class.)
2. **`ncg_last_ty` is set only on some paths** — notably `ND_LET` never recorded a
   string type when the initializer is a non-literal string call, and array
   element *loads* never set `ncg_last_ty = 1` at all.
3. **Parameters were never typed** — a parameter never enters the checker's
   symbol table, so `fn show(string s) { println(s); }` had no way to know `s` was
   a string.
4. **Observable pre-existing symptom** — `println(arr[0])` printed the element's
   **pointer as an integer**, not the string. This was a latent correctness bug
   independent of RC, and it was the same missing information RC needs.

Because a value's string-ness was not propagated through element loads, function
returns, parameters, and some locals, the pass could not prove a value was
managed before emitting a release — and a single mis-placed release turns the
benign leak into a use-after-free (which is worse).

**Prerequisite for Slice 2 — step 1 is now done**

The recommended next step was a **type-tagging pass** carrying the checker's
static types into codegen. The additive half of it has shipped:

- `ncg_ty_of(node)` / `ncg_elem_ty(node)` in `src/codegen/native.ionic` — a
  **pure** function over the AST and the symbol tables that answers "int64,
  string, float64, or none of those" for an expression, and the same for an
  array element.
- It is what `print`/`println` dispatch consults, replacing the hand-written
  builtin-name list that was effectively guessing. That fixed items 2, 3, and 4
  above as observable defects: `println(names[0])` and `println(s)` for a
  `string` parameter now print the string.
- Element types are recorded per local slot (`local_elem`), and parameters record
  their declared type, so the information survives the way it needs to.

**Struct fields and `.len` — the last two `OTY_OTHER` answers that were knowable.**
Both made `println(b.val)` print the field's pointer as an integer, and left
`println(a.len)` one bad dispatch away from the same. Two answers were already in
the checker and neither was being asked for:

- `ncg_stype_of(node)` — the pure equivalent of `ncg_last_stype`, which is a side
  effect of emission and therefore unusable at a point that has not emitted yet.
  `ncg_ty_of` now resolves an `ND_FIELD_ACCESS` receiver through it and reads the
  field's declared type from `st_field_ty`.
- `.len` is special-cased to `OTY_INT`, because the emitter always compiles it to
  `ionic_array_len` whatever the receiver's type is. The oracle has to agree with
  the emitter, or dispatch picks a variant for a value that was never produced.

A struct- or array-valued field still answers `OTY_OTHER`: the oracle's vocabulary
is three scalar classes and has no code for an aggregate.

**A returned array's element type — the root cause of a whole bug class.** The
checker's `register_fn_node` stored the return type's `sv` via `node_ty_sv`, which
only encodes a *named* type. So `-> [string]` recorded `""`, and every call
returning an array lost its element type at the boundary: the array itself
survived (`.len` worked) but `p[0]` typed as unknown. `sc_sv` for an array *is* its
element type, so the fix is `node_ty_val_sv` — the function parameters already
used it. Element types are then recorded at the `let` binding (`local_elem`),
because `p[0]`'s receiver is the local, not the call.

**What is still required before Slice 2 may be retried.** The oracle is
deliberately conservative. It answers "none of those" (`OTY_OTHER`) for a
nested-array element and a method call. A retain/release pass must therefore:

- emit operations **only** where `ncg_ty_of` proves the value is managed, and
- treat `OTY_OTHER` as *unknown*, not as "not managed" — the two differ, and
  conflating them is exactly how the first attempt produced a release on a
  stack address.

That is a narrower, sound starting point. Widening it means teaching the oracle
about method return types (needs `fn_ret_ty` to describe methods).

A second, cheaper prerequisite is to give the `@gpu`/tensor resource path
(Slice 3, RAII) priority: it needs no value-type oracle, because handles are
released explicitly and deterministically.

## 11. Allocation audit — every pointer the runtime can hand to a program

A reference count lives in an 8-byte word immediately *before* the payload, so
`rc_release(p)` is `--((int64_t*)p)[-1]` and `rc_alloc` frees `p - 8`. That
design has three preconditions, and none of them had been checked:

1. every value the codegen believes is managed came from `rc_alloc`;
2. `p - 8` is inside the allocation that owns `p` — i.e. `p` is the *start* of a
   payload, never an interior or foreign address;
3. releasing it exactly once balances the retain that produced it.

This section is the audit of (1) and (2) across `src/ionic_model_runtime.c`.
It is the missing information the Slice 2 log blamed on "type classification":
classification was necessary but not sufficient. The §11.2 findings failed
*even when the oracle was right*, because the checker declared the value to be a
`TY_STR` and the runtime still returned something a refcount cannot own. Both
classes are now closed; this section is kept as the record of why the audit came
before RC rather than after it.

Classification key:

- **fresh** — `malloc`'d in the callee, exactly one allocation, payload starts at
  offset 0. Safe to `rc_alloc` and safe to release. *18 functions.*
- **alias** — returns one of its **inputs**, unchanged. Retaining/releasing it
  changes the input's refcount. **Unsafe as-is.** *0 functions* — resolved, §11.1.
- **interior** — returns a pointer *into* a larger allocation. A refcount word at
  `p - 8` lands inside that allocation. **Unsafe as-is.** *0 functions* —
  resolved by copy-on-access, §11.2.
- **foreign/static** — not from this allocator at all: rodata literal, `getenv`
  buffer, argv, a `static` buffer. **Unsafe to release.** *4 functions, 5
  definitions* (the two `mlx_last_error` builds plus the shared `get_arg` /
  `getenv` / `exe_dir` sources).
- **caller-buffer** — writes into a buffer the caller owns. Not a return of new
  memory at all. *1 function* (`ionic_fgets_stdin`).

23 distinct functions return a `char *`/`const char *` to a program. That is 25
definitions once you count the two `mlx_*` accessors, which have separate MLX and
stub builds. (An earlier revision said 26: `ionic_file_read` used to exist twice,
as a weak C definition *and* an inline LLVM IR definition emitted by the Rust
bootstrap. The IR definition is gone — the IR now only declares it — so there is
one implementation again. See §8 step 4.)

### 11.1 `alias` — returns an input, not a copy — RESOLVED

This class is now empty. Every string-returning entry point allocates on every
path, so "returned a `TY_STR` ⇒ the result is an owned `malloc` block" holds
unconditionally, with one exception noted at the end (OOM).

What was wrong, for the record:

| function | condition | what it returned | why it was unsafe |
|---|---|---|---|
| `ionic_file_read` | `fopen` failed | `(char *)""` | rodata literal — a release is a free of static data |
| `ionic_str_replace` | any of `s`/`from`/`to` is NULL | `s` (or `""`) | the **input**, or rodata |
| `ionic_http_body` | no `\r\n\r\n` in the response | `raw` | the **input** |
| `ionic_http_get` | socket/lookup/connect failure | `(char *)""` | rodata literal |
| `ionic_http_post` | socket/lookup/connect failure | `(char *)""` | rodata literal |

The last two were not in the original audit: they have the same shape as
`file_read`'s failure path (`x ? x : ""`) and are equally reachable, so they are
included in the class.

Reachability from Ionic source, not from C:

- `file_read` on a missing path is an ordinary, easily-reached program shape;
  `examples/mastery/03_file_io.ionic` does it and prints `len = 0`.
- `http_get` against an unreachable host is similarly ordinary.
- `str_replace`/`http_body` with a *NULL* argument is **not** reachable — Ionic
  strings are never NULL (a failed allocation aside). Those paths were fixed
  anyway because the C functions are part of the runtime's own surface and a C
  caller has no way to know the contract differs by path.

Measured, not assumed (`clang -O0`, mirroring the exact bodies). The check is
"is the returned pointer the start of a `malloc` block?", via
`malloc_zone_from_ptr` — the same precondition `rc_alloc` needs, asked of the
allocator rather than inferred from the contents:

```
before:  11 paths probed → 5 not owned → first free() aborts (exit 134)
after:   11 paths probed → 11 owned   → all free()able      (exit 0)
```

The fix is uniform: **`rc_alloc` returns are the contract, so every return path
must allocate.** `src/ionic_model_runtime.c` now routes each failure path
through one helper:

```c
static char *rt_empty_str(void) {          /* allocated, not rodata */
    char *e = (char *)ionic_rc_alloc(1);
    if (e) e[0] = '\0';
    return e;
}
```

and the three alias paths copy instead of returning their input. (The helper
originally called `malloc`; step 4 stage 1 moved it onto `ionic_rc_alloc`, so a
failure path returns a counted block like every other producer. Because *every*
path in every producer now allocates, a `release` on any of them is well
defined — which is the property `scripts/rc_check.sh` asserts directly.)

Note `ionic_file_read` is `__attribute__((weak))` because the LLVM IR path used
to define it inline in `src/codegen.rs`. The C fix alone was **not** enough: the
IR definition also returned `@empty_str`, a rodata constant, and the
`--verify-self-hosting` check would not have caught the divergence because both
builds emit the same object *file* while differing only in a runtime routine's
behaviour.

That divergence class is now gone rather than kept in step by hand: the IR only
*declares* the producer, so both builds call one C implementation (§8 step 4).
The `weak` attribute stays as the mechanism that makes the IR's absence legal —
it costs nothing and documents the original reason.

**One path still cannot honour the contract: OOM.** If `malloc` fails inside a
failure path, `rt_empty_str` returns NULL. That is a hard error everywhere else
in this runtime too (Ionic strings are otherwise never NULL), and it is not
reachable in practice; it is recorded here so the invariant is stated honestly
rather than absolutely.

### 11.2 `interior` — a pointer into a larger object — RESOLVED

This was the severe one, because the *checker type-checks the result as a
string*, so the oracle was right and the release still fired.

`HttpRequest` (`src/ionic_model_runtime.c:3079`) stores two strings **inline**:

```c
typedef struct {
    int   conn;
    char  method[16];   /* offset 4  */
    char  path[1024];   /* offset 20 */
    char *body;
    ...
} HttpRequest;
```

and three accessors handed out pointers into it:

| function | line | returned | declared type |
|---|---|---|---|
| `ionic_http_method` | 3241 | `r->method` (offset 4) | `TY_STR` |
| `ionic_http_path` | 3246 | `r->path` (offset 20) | `TY_STR` |
| `ionic_http_req_body` | 3251 | `r->body` (a heap pointer, or `""`) | `TY_STR` |

`http_path` returned the *start of an inline array*, so `p - 8` was
`offset 12` — the last 8 bytes of `method`. Measured before the fix:

```
before:  method[0..15]  47 45 54 00 00 00 00 00 00 00 00 00 00 00 00 00
one release ->
after:   method[0..15]  47 45 54 00 00 00 00 00 ff ff ff ff ff ff ff ff
```

`method` silently became `"GET"` followed by eight `0xff` bytes. When the count
reached zero, `free(p - 8)` was called on `offset 12` of a `calloc`'d block,
which aborts.

**Fix applied: copy on access.** All three now return a `rt_dup_str()` copy, so
`p` is always the start of an allocation the callee made. Copy-on-access was
chosen over the "declare them borrowed" alternative because both compilers
already classified these three as *owned* (`ncg_ty_of`, and the `ncg_last_ty = 1`
list in `src/codegen/native.ionic`), so the copy makes the existing model true
instead of introducing an owned-vs-borrowed distinction neither backend has
today. That distinction (§8 step 6) remains the durable answer for foreign
pointers generally; it was not needed for this class.

`ionic_http_req_body` was a third case again: `r->body` is a real heap pointer
owned by the *request*, so releasing it as a program string freed memory the
request also frees in `http_req_free` — a double free on a pointer that at least
began an allocation. Same treatment, same one-line shape.

Measured after the fix with an ASan probe over all four accessors (the three HTTP
ones plus `mlx_weights_key_at`, below):

```
m=0x602000000970 p=0x602000000990 b=0x6020000009b0 k=0x6020000009d0
lens 0 0 0 0
freed OK            exit 0
```

Four distinct heap allocations, each freed cleanly. The same probe against the
pre-fix bodies reports `ALIASED` — all three HTTP accessors returned the one
rodata pointer `0x1048d8940` — and `free()` on any of them aborts with exit 134.

Inside the JSON/BPE code there are further interior pointers — `JVal.str` is
handed out by `jget` as a pointer into a tree whose lifetime the caller does not
own. None of them is reachable as a *program string* (they are internal `JVal`
plumbing, not registered builtins), so they were never candidates for
instrumentation. They are recorded here so that a future widening of the oracle,
or a decision to expose the JSON tree to Ionic programs, does not sweep them in
by accident.

### 11.3 `foreign/static` — never from this allocator

| function | line | source | rule |
|---|---|---|---|
| `ionic_get_arg` | 164 | `argv[idx]` (process-owned) | never release |
| `ionic_getenv` | 173 | `getenv` buffer (libc-owned) | never release |
| `ionic_exe_dir` | 184 | `static char dir[PATH_MAX]` | never release |
| `ionic_mlx_last_error` | 1498 | `static char g_mlx_err[512]` | never release |
| `ionic_mlx_last_error` (stub) | 1741 | rodata | never release |

`ionic_mlx_weights_key_at` **was** in this class and no longer is: it returned a
`static char keybuf[512]`, so `let k = mlx_weights_key_at(h, 0);
let k2 = mlx_weights_key_at(h, 1);` left both names reading `k2`'s text — a plain
aliasing bug that predated RC and was independent of it. It now returns a
`rt_dup_str()` copy of that buffer on the MLX build, and `rt_empty_str()` on the
stub, so the value is owned even though its *source* is still static. See §11.2.

`ionic_mlx_last_error` is the one remaining trap: it returns `const char *` into
`static` storage while the checker types the builtin as an ordinary `TY_STR`.
Copy-on-access would fix it identically; it is not done because the buffer is a
process-wide sticky diagnostic rather than per-object data, and no program has
been observed to hold two live error strings. Recorded so the choice is explicit.

### 11.4 Arrays and aggregates

`IonicArray` is `{ i64 len; i64 cap; void *data; }`, allocated by
`ionic_array_alloc`. It is a **two-allocation** object: the header and
`data`. An RC header on the `IonicArray *` covers only the header, so releasing
an array must also free `data` (and, if `data` holds managed slots, release
them first). `ionic_arr_reset` (`:138`) currently sets `len = 0` and does
nothing else — under RC it must release each element it drops.

Strings stored in array slots are a separate retain: `ionic_array_get` returns
`((int64_t*)arr->data)[i]`, an element *value*. Reading it out does not copy or
retain, so `let s = names[0]` aliases the slot; writing `names[0] = s` stores a
borrow. Both directions need explicit handling, and this is why the oracle
needed `local_elem`.

### 11.5 The audit's verdict

Of the 25 definitions (23 distinct functions; two `mlx_*` accessors have separate
MLX and stub builds):

| class | count | verdict |
|---|---|---|
| fresh | 18 | safe to `rc_alloc`; no change needed |
| alias on some path | 0 | **resolved** — every return path now allocates (§11.1) |
| interior | 0 | **resolved** — copy-on-access (§11.2) |
| foreign/static | 4 (5 definitions) | must never be instrumented (§11.3) |
| caller-buffer | 1 | `fgets_stdin` writes into a buffer the caller allocated |

The interior count moved to zero in two directions: the three `http_*` request
accessors copy (§11.2), and `mlx_weights_key_at` moved out of
foreign/static into fresh because it now copies its static buffer. So
`fresh` absorbed four functions (14 → 18), `interior` emptied (3 → 0), and
`foreign/static` dropped from 5 functions / 6 definitions to 4 / 5. The totals
still reconcile: 18 + 0 + 0 + 4 + 1 = 23 functions.

Every string a program can obtain is now either a fresh allocation it owns or
one of the four documented foreign/static sources that must never be
instrumented. There are no remaining *bugs in this class*: no accessor returns a
pointer whose release corrupts a neighbour, and no two live values alias by
accident.

So the ordering in §8 was wrong in one respect: this audit is not a *step*
before RC, it is the thing that determines whether RC's design is viable. It is
viable. The retain/release pass still needs the owned-vs-borrowed type
distinction (§8 step 6) to be *sound by construction* rather than by a
hand-maintained exemption list, but nothing in §11 blocks landing it.

### 11.6 Cycles are constructible, contrary to §6 — RESOLVED as an accepted leak

§6 asserted cycles cannot form because "shared data is immutable and mutable
ownership is unique/affine". One half of that holds and one does not:

- **Immutable-shared-data half: HOLDS.** Strings and tensor storage are never
  mutated in place, so no cycle can form over them. Refcounting is complete for
  both.
- **Affine-ownership half: DOES NOT HOLD.** Two names can alias one mutable
  aggregate, and an aggregate can contain a reference to an aggregate.

Measured (`/tmp/alias1.ionic`, `/tmp/rs11.ionic`, `/tmp/rs13.ionic`):

```
let p = P { x: 1, y: 2 }; let q = p; q.x = 99;   // p.x == 99
p.y = 7;                                          // q.y == 7   (both directions)

struct Bag { n: int64, items: [Bag] }
let b = Bag { n: 7, items: empty };
b.items.push(b);
b.items[0].n;              // 7   — the same Bag
b.items[0].items[0].n;     // 7   — and again, unboundedly
```

`let q = p` copies the *pointer* (structs and arrays are heap objects behind a
single-word value), so two names alias one mutable aggregate. The second example
is the more important one: **a single object points at itself**, so the back-edge
needs neither two objects nor two names.

**The earlier framing of the fix was wrong, and this is the correction.** This
section used to propose "prevent aggregate-in-aggregate at the type level" as a
way out. That does not work, because the cycle does not require a nested
container at all:

```
struct Node { val: int64, next: Node }
let m = Node { val: 2 };
let n = Node { val: 1, next: m };
n.next = n;
n.next.val;        // 1
n.next.next.val;   // 1   — traversable, well-typed, no array involved
```

A struct with a field of its own type is legal today. So a rule banning
"aggregate in aggregate" would forbid recursive structs — trees and lists
included — which is a far larger language restriction than accepting the leak
would be a cost.

**Decision: accepted, bounded leak (§6.1).** Refcounting reclaims acyclic
aggregate graphs only; cyclic ones are documented as leaking, and no cycle
collector is planned. The rationale, the scope, and the rejected alternatives are
in §6.1 rather than duplicated here.

**Consequence for benchmarks — the one operational obligation.** Because
cyclic aggregates never return memory, a benchmark that builds one can never go
flat, and its failure would be indistinguishable from "aggregate RC is not
implemented yet". Measured today, with arrays not yet counted, an acyclic and a
self-cyclic benchmark of the same shape are indistinguishable in RSS (~192 B/iter
both) precisely because *nothing* aggregate-shaped is reclaimed yet. Step 4
stage 2 must therefore (a) land aggregate RC with a **cycle-free** acyclic
benchmark, and (b) add a second, small, **deliberately cyclic** benchmark that
asserts *the leak is present and bounded* — a growth in proportion to the number
of cycles created, not to iteration count. Without (b) the accepted leak is
silent, and a future regression that turned a bounded leak into an unbounded one
would look exactly like the documented behaviour.

`scripts/rss_check.sh` remains a valid hard gate for strings throughout: its
benchmark is string churn, and strings have no cycle exemption.

## See also

- `docs/language.md` — language reference (types, arrays, structs, placement)
- `src/ionic_model_runtime.c` — the C runtime and its allocators
- `src/codegen/native.ionic` — native ARM64 codegen (allocation, local/global typing)
