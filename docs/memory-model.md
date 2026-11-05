# Ionic Memory Model

How memory is allocated, shared, and reclaimed — and the long-term plan for
eliminating leaks without a tracing collector.

---

## 1. Current state (as shipped)

Ionic today has **no garbage collector, no reference counting, and no lifetime
system**. It is C-style, allocation-heavy, and reclamation-free by design:

- **Strings** are `malloc`'d C strings (`char *`), treated as immutable `const`
  values. Every `str_concat`, `int64_to_str`, `str_slice`, `format`,
  `char_to_str`, `file_read`, and `read_line` allocates a fresh buffer and never
  frees it (`src/ionic_model_runtime.c`).
- **Structs** are heap-allocated with `malloc` at construction
  (`src/codegen/native.ionic`, `ND_STRUCT_LIT`).
- **Arrays** use a 24-byte header `[len:i64][cap:i64][data_ptr]` plus a separate
  `malloc`/`realloc`'d data region (`IonicArray`, `src/ionic_model_runtime.c`).
- **The only explicit free** is `model_free(m)`, which releases ML model
  resources (ONNX Runtime handles) via `ionic_model_free`.

The runtime states the policy directly:

```c
return buf; /* caller owns — but Ionic treats strings as const; leak is acceptable for now */
```

`src/ionic_model_runtime.c`

Consequence: any long-running Ionic program that churns strings, arrays, or
structs grows without bound. The compiler itself leaks for this reason.

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
4. **Cycles are intended to be impossible**, so no tracing GC or cycle collector
   would be required. *This does not currently hold* — see §6 and §11.6.

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
  life at refcount 1: `ionic_str_concat`, `ionic_int64_to_str`,
  `ionic_float64_to_str`, `ionic_str_slice`, `ionic_str_replace`, `ionic_format`,
  `ionic_char_to_str`, `ionic_read_line`, `ionic_file_read`, `ionic_http_get`,
  `ionic_http_post`, `ionic_http_body`, `ionic_http_urlencode`, and
  `ionic_array_alloc`. Every one of these now allocates on every path — including
  the failure paths, which is what §11.1 was about.
  **Not** `ionic_get_arg`, `ionic_getenv`, `ionic_exe_dir`, or
  `ionic_mlx_last_error` — those return process-, libc-, or statically-owned
  buffers and must never be instrumented. The three `ionic_http_*` *request*
  accessors belong on this list too: they now copy out of the `HttpRequest`, so
  they are fresh allocations and safe to instrument (§11.2). See §11.3; an
  earlier version of this list wrongly included `get_arg` and `getenv`.

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
  This was rejected on the grounds that the type system prevents cycles outright —
  which, per §11.6, it currently does not. If §6 is not restored, this moves back
  onto the table.

---

## 6. Cycles: the intended invariant, and where it does not hold

> **Correction (§11.6).** This section previously claimed cycles were structurally
> impossible. They are not: two names can alias one mutable aggregate
> (`let q = p`) and aggregates can contain aggregates (`a.push(b)`), so a
> reference cycle is constructible from ordinary Ionic. Read this section as the
> *intended* design and §11.6 for the shipped behaviour.

A cycle collector is only necessary if a cycle can form. As designed, Ionic's
data model was meant to forbid it:

- **Shared data is immutable** (strings, tensor storage). You cannot create a
  cycle by mutating shared immutable data.
- **Mutable ownership is unique / affine.** A `mut` value was meant to have
  exactly one owner, with no aliasing through which a back-edge could be built.

Under those two rules every reference graph is a DAG, and refcounting alone
reclaims it completely — which is what would make Ionic leak-free *without* a
general-purpose collector.

The second rule is the one that did not survive contact with the implementation:
`let q = p` copies a pointer rather than moving it, for both structs and arrays.
Until that is changed (or aggregate-in-aggregate is rejected at the type level),
refcounting reclaims acyclic graphs only and cyclic aggregates leak by
construction. See §11.6 for the measurement and the options.

---

## 7. Consequences

**What improves**

- Transient strings and arrays are freed the moment they die — the dominant leak
  in user programs disappears.
- Device/model memory is released deterministically, never at the whim of a
  collector pause.
- The compiler binary's own leak is gone (§4.4).

**What RC does not solve**

- **Reference cycles among counted objects.** Intended to be prevented by design
  (§6), but they *are* constructible today (§11.6), and the objects in a cycle
  leak. Mitigation is either a type-level restriction or an accepted, documented
  leak.
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
5. **RC for tensor storage + RAII for `model`/handle types.** The domain core:
   deterministic release of scarce, device-affine memory. *(The `model` RAII half
   is implemented — see §9.)*
6. **Type-level `owned` vs. `borrowed` distinction.** Makes the ownership
   discipline explicit in the type system rather than a runtime convention, and
   is the correct handling of foreign pointers from C/library interop.
7. **Decide the cycle question (§11.6).** Either amend §6 to accept bounded leaks
   for cyclic aggregates, or prevent aggregate-in-aggregate at the type level.
   Step 4 can land before this, but `rss_check.sh` cannot be a hard gate until it
   is decided.

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
- **Slice 2 (RC for strings/arrays) — prototyped and REVERTED.** See §10 for the
  implementation log and the prerequisite it exposed.
- **Prerequisite for Slice 2 — the type oracle, step 1 SHIPPED.** `ncg_ty_of` /
  `ncg_elem_ty` answer "int64 / string / float64 / none of those" for an
  expression and for an array element, and print dispatch now consults them. That
  closed the observable `println` defects (array element, string parameter). It is
  deliberately conservative, and the remaining widening is described in §10.
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

**What is still required before Slice 2 may be retried.** The oracle is
deliberately conservative. It answers "none of those" (`OTY_OTHER`) for a
nested-array element, a method call, and a struct-field load. A retain/release
pass must therefore:

- emit operations **only** where `ncg_ty_of` proves the value is managed, and
- treat `OTY_OTHER` as *unknown*, not as "not managed" — the two differ, and
  conflating them is exactly how the first attempt produced a release on a
  stack address.

That is a narrower, sound starting point. Widening it means teaching the oracle
about struct fields (needs field types in the struct registry) and about method
return types (needs `fn_ret_ty` to describe methods).

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

23 distinct functions return a `char *`/`const char *` to a program. That is 26
definitions once you count the two `mlx_*` accessors, which have separate MLX and
stub builds, and `ionic_file_read`, which exists twice for a different reason: a
weak C definition and an inline LLVM IR definition emitted by the Rust
bootstrap (§11.1).

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
    char *e = (char *)malloc(1);
    if (e) e[0] = '\0';
    return e;
}
```

and the three alias paths copy instead of returning their input.

Note `ionic_file_read` is `__attribute__((weak))` because the LLVM IR path
defines it inline in `src/codegen.rs`. The C fix alone was **not** enough: the
IR definition also returned `@empty_str`, a rodata constant, and the
`--verify-self-hosting` check would not have caught the divergence because both
builds emit the same object *file* while differing only in a runtime routine's
behaviour. Both definitions are now in step, and the comment in `codegen.rs`
says so.

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

Of the 26 definitions (23 distinct functions; two `mlx_*` accessors have separate
MLX and stub builds, and `ionic_file_read` has both a weak C and an inline IR
definition):

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

### 11.6 Cycles are constructible, contrary to §6

§6 asserts cycles cannot form because "shared data is immutable and mutable
ownership is unique/affine". Neither half holds in the shipped language. Measured:

```
let a = [1,2,3]; let b = [4,5,6];
a.push(b); b.push(a);        // a.len == 4: an array element holding an array

let p = P { x: 1, y: 2 }; let q = p; q.x = 99;   // p.x == 99
```

`let q = p` copies the *pointer* (structs and arrays are heap objects behind a
single-word value), so two names alias one mutable aggregate. That gives both
aliasing of mutable data and a back-edge, i.e. a reference cycle.

Consequence: refcounting alone leaves mutually-referencing aggregates
permanently leaked. The `scripts/rss_check.sh` benchmark must therefore remain
cycle-free or it will never go flat, and §6 needs either an amendment ("bounded,
programmer-visible leaks are accepted for cyclic aggregates") or a decision to
prevent aggregate-in-aggregate at the type level. This is a design call, not an
implementation task, and it should be made before the retain/release pass is
written rather than discovered by it.

## See also

- `docs/language.md` — language reference (types, arrays, structs, placement)
- `src/ionic_model_runtime.c` — the C runtime and its allocators
- `src/codegen/native.ionic` — native ARM64 codegen (allocation, local/global typing)
