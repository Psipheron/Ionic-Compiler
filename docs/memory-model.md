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
  resources (ONNX / llama.cpp handles) via `ionic_model_free`.

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

- **Static types, but the backend's type tracking is incomplete.** The native
  backend tracks which locals and globals hold strings vs. floats vs. structs
  (`local_is_str`, `local_is_float_slot`, `local_get_stype`, `glob_is_str`), but
  this is *not* sufficient on its own: string-ness is lost on array/struct
  element loads and on several expression paths (see §10). An ownership model
  needs a *complete* static type oracle — that missing oracle is the
  prerequisite, not the existing ad-hoc tables.
- **GPU placement (`@cpu` / `@gpu`).** Tensor and model memory crosses a device
  boundary. Reclamation of device memory must be *deterministic* — a delayed
  collector pause during inference is unacceptable.
- **C / library interop** (`llama.cpp`, ONNX Runtime, POSIX sockets) hands out
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
4. **Cycles are made structurally impossible**, so no tracing GC or cycle
   collector is ever required.

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
  a safe no-op. Option (b) proved more robust when the codegen cannot reliably
  classify a value's string-ness at every site (see §10).
- **Runtime builtins route through `rc_alloc`** so the objects they return begin
  life at refcount 1: `ionic_str_concat`, `ionic_int64_to_str`,
  `ionic_float64_to_str`, `ionic_str_slice`, `ionic_format`, `ionic_char_to_str`,
  `ionic_read_line`, `ionic_file_read`, and `ionic_array_alloc`.

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
  unconditionally leak-free, but adds a mark phase and full object enumeration for
  a case that the type system can prevent outright (§6).

---

## 6. The key principle: cycles are structurally impossible

A cycle collector is only necessary if a cycle can form. Ionic's data model
forbids it:

- **Shared data is immutable** (strings, immutable array data, tensor storage).
  You cannot create a cycle by mutating shared immutable data.
- **Mutable ownership is unique / affine.** A `mut` value has exactly one owner;
  there is no aliasing through which a back-edge could be constructed.

Because no object can hold a reference back to something that holds it, every
reference graph is a DAG, and refcounting alone reclaims it completely. This is
the reason Ionic can be leak-free *without* a general-purpose collector.

> If the language ever adds mutable, aliased, cyclic references, this invariant
> breaks and a cycle collector becomes necessary. That trade-off should be made
> deliberately, not accidentally.

---

## 7. Consequences

**What improves**

- Transient strings and arrays are freed the moment they die — the dominant leak
  in user programs disappears.
- Device/model memory is released deterministically, never at the whim of a
  collector pause.
- The compiler binary's own leak is gone (§4.4).

**What RC does not solve**

- **Reference cycles among counted objects.** Prevented by design (§6), not
  collected. If a cycle is ever constructed, the participating objects leak.
- **C-runtime buffers not routed through `rc_alloc`.** Anything `malloc`'d inside
  `src/ionic_model_runtime.c` (for example the `gguf_generate` result buffer, the
  `http_get` body) is invisible to the scheme until it is re-routed. These are
  fixed individually.
- **Interior / foreign pointers.** A pointer stashed where the compiler cannot see
  it (a C callback, a cache) is not reflected in any refcount. This is the main
  soundness risk and the reason a type-level owned-vs-borrowed distinction
  (§8, step 4) is the durable answer.

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

1. **Compiler arena per phase.** Immediate, self-contained leak elimination in
   `ionic_new`; no language semantics change.
2. **RC for strings and arrays.** Non-atomic. Literals are wrapped as pinned
   managed copies. Fixes the dominant user-program leak.
3. **RC for tensor storage + RAII for `model`/handle types.** The domain core:
   deterministic release of scarce, device-affine memory. *(The `model` RAII half
   is implemented — see §9.)*
4. **Type-level `owned` vs. `borrowed` distinction.** Makes the ownership
   discipline explicit in the type system rather than a runtime convention, and
   is the correct handling of foreign pointers from C/library interop.

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
- **Slice 2 (RC for strings/arrays) — prototyped and REVERTED.** See §10 for the
  implementation log and the prerequisite it exposed.

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
2. **`ncg_last_ty` is set only on some paths** — notably `ND_LET` never records a
   string type when the initializer is a non-literal string call, and array /
   struct element *loads* never set `ncg_last_ty = 1` at all.
3. **Observable pre-existing symptom** — `println(arr[0])` prints the element's
   **pointer as an integer**, not the string: element loads lose string-ness at
   the codegen level. This is a latent correctness bug independent of RC, and it
   is the same missing information RC needs.

Because a value's string-ness is not propagated through element loads, function
returns, and some locals, the pass cannot prove a value is managed before
emitting a release — and a single mis-placed release turns the benign leak into a
use-after-free (which is worse).

**Prerequisite for Slice 2 (recommended next step)**

Add a **type-tagging pass** that carries the semantic checker's static types into
codegen and makes `ncg_last_ty` correct on *every* expression path — in
particular array/struct element loads. This fixes the latent `println(arr[i])`
bug at the same time and makes the retain/release insertion mechanical and sound.
`local_get_str`/`local_find`/`ncg_string_expr` should be replaced by that
uniform type oracle rather than the current shadow-prone name tables.

A second, cheaper prerequisite is to give the `@gpu`/tensor resource path
(Slice 3, RAII) priority: it needs no value-type oracle, because handles are
released explicitly and deterministically.

## See also

- `docs/language.md` — language reference (types, arrays, structs, placement)
- `src/ionic_model_runtime.c` — the C runtime and its allocators
- `src/codegen/native.ionic` — native ARM64 codegen (allocation, local/global typing)
