# Functional Hardware-Placement Awareness

Status: **plan, not yet implemented** — revised 2026-11-03
Author: audit + design pass
Scope: `src/semantic/checker.ionic`, `src/ionic_model_runtime.c`,
`src/codegen/native.ionic`, `docs/`

---

## 1. What changed in this revision

The previous revision of this plan was built around MLX: it made `tensor@cpu`
mean "host-side" and `tensor@gpu` mean "Metal-resident", with `mlx_copy` as the
transfer mechanism. **That architecture is wrong** for the project's actual
purpose, and this revision replaces it.

Corrections from the maintainer:

1. **The target is AI on servers and PCs.** MLX exists in the tree for testing
   and to keep the build multiplatform. It is **not** the intended GPU backend
   and must not shape the semantics.
2. **`@cpu` / `@gpu` should mean SRAM and VRAM** — host memory versus
   accelerator memory, with **different capacities**.
3. **Metal's unified memory is the exception, not the rule.** On Apple silicon
   CPU and GPU address the same physical memory, so a transfer there is
   semantically a no-op. Elsewhere they are separate pools.

Consequence: the central abstraction is **not** "which device is this tensor
on" but **"which memory pool does this tensor live in, and does that pool have
room for it."** The type system's job is to make a pool-exhaustion failure a
*compile-time* error rather than a runtime crash in the middle of a training
step.

---

## 2. The model

### 2.1 Two pools, not two devices

```
        @cpu  ──────────────►  SRAM          (host DRAM; capacity = system RAM)
                                 │
                              transfer
                                 │
        @gpu  ──────────────►  VRAM          (accelerator memory; capacity = card)
```

- **SRAM** — host memory. Large, slow-ish to reach from the accelerator.
- **VRAM** — accelerator memory. Smaller, faster, and **the scarce resource**.

`tensor@cpu` and `tensor@gpu` are *pool* types. A value of type `tensor@gpu`
means "its bytes live in the accelerator's memory pool."

### 2.2 Why capacity is the point

On a server running a 70B model on one card, the interesting failure is not
"wrong device" — it is "the model did not fit." If `@gpu` does not carry the
capacity constraint, the compiler enforces a distinction that the hardware
does not make and misses the one it does.

So the contract a `@gpu` function must satisfy is:

> Every value of type `tensor@gpu` in this function's scope must fit in the
> accelerator's memory pool, simultaneously.

That is a **static** property in the common case: model weights are known at
compile time (shape × dtype × count), and the pool size is a deployment
constant. This is checkable. It is also checkable *before* the program runs,
which is the entire value of a compile-time type system.

### 2.3 Unified memory: one backend where the transfer is free

On Apple silicon, CPU and GPU share physical memory. A "transfer" there is a
no-op: the same bytes are reachable from both. The type still changes
(`tensor@cpu` → `tensor@gpu`) because the *pool* the value is accounted
against changes — which matters for capacity accounting even when the physical
move is free.

This is why MLX remains useful as a **test backend**: it exercises the
"unified" case, where transfers are free and capacity is shared. It is a
correct implementation of one point in the space, not the definition.

---

## 3. What exists today (audit)

### 3.1 The checker is already close to the right shape

Placement types are real, distinct, and enforced at all eight `ty_compat` call
sites.

| Mechanism | Location | State |
|---|---|---|
| Distinct types `TY_TENSOR_CPU` / `TY_TENSOR_GPU` | `checker.ionic:14-15` | works |
| `ty_compat` — no implicit CPU↔GPU coercion | `checker.ionic:140-146` | works |
| Argument placement check | `checker.ionic:1096` | works |
| Assignment / field / let / return checks | `:1311`, `:1327`, `:1283`, `:1352` | works |
| `tensor@cpu` inside `gpu {}` rejected | `:975` | works |
| `tensor@gpu` outside `gpu {}` rejected | `:980` | works |
| `@cpu fn` called from `@gpu` context rejected | `:1075` | works |
| CPU-only I/O inside GPU context rejected | `:1062` | works |
| `.toGpu()` / `.toCpu()` are the only transfers | `:1120-1131` | works |
| Conversion hints on mismatch | `:478-494` | works |

The **separation** half of the contract is done and sound.

### 3.2 What is missing is the capacity half

Nothing in the tree models pool size. Searching for capacity concepts:

- `grep -rniE 'arena|pool|capacity|oom|out.of.memory'` in
  `src/ionic_model_runtime.c` → **one** hit, and it is `OrtArenaAllocator`
  (`:1857`), an ONNX Runtime internal, not an Ionic concept.
- The only memory-control surface is MLX's: `mlx_set_memory_limit`,
  `mlx_active_memory`, `mlx_peak_memory` (`:1470-1500`).

So the type system separates pools correctly and says nothing about whether
the values fit.

### 3.3 The blocking finding: `tensor@gpu` is a dead type

- **No builtin accepts `tensor@gpu`.** 70 `mlx_*` builtins are registered
  (`checker.ionic:821-940`); 47 declare `TY_TENSOR_CPU` as their return type,
  and the rest take tensor parameters via `bparams_tensor` (`:650-654`), which
  pins every parameter slot to `TY_TENSOR_CPU`. `TY_TENSOR_GPU` appears only in
  type machinery, never in a signature.
- **There is no tensor arithmetic.** Verified: two `tensor@gpu` operands to `+`
  yield `type error in arithmetic: expected numeric type, got tensor@gpu`.

A `tensor@gpu` can be created, passed to `@gpu` functions, and returned — and
nothing can consume it.

### 3.4 The runtime is inverted

`MlxTensor` is `{ magic, alive, mlx_array a }` — no pool field
(`ionic_model_runtime.c:738-742`). All 34 MLX call sites hardcode
`g_mlx_gpu_stream`; there is no CPU stream in the file at all. Every tensor is
GPU-resident regardless of its Ionic type, so `tensor@cpu` is already VRAM and
`tensor@gpu` is unreachable.

`toGpu()`/`toCpu()` compile to nothing (`native.ionic:1375` is an identity) and
`gpu { }` compiles to nothing (`:1957-1959`).

---

## 4. Design

### 4.1 Two orthogonal properties, not one type

The current design conflates two things that should be separable:

- **Pool** — where the bytes live. This is what `tensor@cpu` / `tensor@gpu`
  encode.
- **Capacity** — whether the bytes fit.

Proposal: keep the pool in the type, and add capacity as a **scoped budget**,
not as a type parameter.

The pool size is a **deployment** fact, so it is declared once at top level and
inherited by every `@gpu` scope; the existing `@gpu(f)` syntax optionally
narrows a scope to a fraction of it. See §6.1.

```
let vram = 25769803776;                // bytes of accelerator memory

@gpu fn train_step(...) -> ... { }     // may use the whole pool
@gpu(0.25) fn helper(...) -> ... { }   // may use a quarter of it
```

The budget is a property of the *function's scope*, which is where the
simultaneity requirement lives. A `tensor@gpu` value's contribution to the
budget is its static size.

This avoids adding `tensor@gpu[N]`-style type parameters (an explosion of type
combinations), keeps byte counts out of function signatures, and matches the
fact that the constraint is about a *set* of live values, not about any one
value.

### 4.2 What the compiler can check statically

For a `@gpu` scope with budget `B` and a set of live `tensor@gpu` values:

1. **Direct size check.** Each tensor's size is `product(shape) × dtype_size`.
   If a tensor is produced by a call with a known shape, its size is known.
   If the sum of live GPU tensors exceeds `B`, that is a **compile error**.

2. **Transfer accounting.** `.toGpu()` moves a value from the SRAM budget to
   the VRAM budget. Both budgets are checked.

3. **Live-range scoping.** Only values live at the same time compete for
   capacity. A tensor that dies before another is allocated does not. This
   needs live-range analysis of the kind a register allocator does. **Deferred**
   to a later stage: the first version sums the sizes of all `tensor@gpu`
   values in scope, which over-approximates (it never under-approximates, so
   it can reject a program that would actually fit, but it will not accept one
   that would not).

4. **Static vs dynamic size.** Some sizes are not knowable at compile time
   (a tensor whose shape depends on runtime data). Those are **unbounded**
   contributions. Per decision 4, the checker **rejects** a statically-
   unbounded GPU tensor in a budgeted scope rather than waving it through: an
   unbounded GPU tensor would defeat the budget entirely.

### 4.3 The ownership question: is a transfer a move or a copy?

This is the sharpest decision, and it is **not** answered by the current code.

If `.toGpu()` is a **move**, then after `let g = x.toGpu();` the CPU-side `x`
is dead. Using `x` afterwards is a use-after-move — a *new class of compile
error* the language does not currently have (verified: no `moved`/`consumed`
tracking anywhere in `src/`).

If `.toGpu()` is a **copy**, then capacity accounting must count both sides,
and the whole point of VRAM capacity is lost to duplicated weights.

Recommendation: **move**, with use-after-move as a checker error. Reasons:

- The model-RAII work already established the precedent: `local_set_model`
  clears ownership on transfer (`native.ionic:1814`), and the same mechanism
  generalises.
- Capacity accounting is only meaningful if values have a single owner.
- It is the semantics that makes "weights live in VRAM" a statement about
  *bytes*, not about *aliases*.

A copy variant is still needed for `toGpu()` of a value that must remain
usable — that should be an explicit, named operation rather than a silent
default.

### 4.4 What "GPU compute" means without a GPU backend

`gpu { }` and `@gpu fn` imply GPU execution. On a machine with no accelerator
backend, they must mean something. Two options were considered:

- **(a) Unavailable (refuse).** The scope is checked, but at runtime it
  reports that no accelerator backend is present and the program cannot
  proceed.
- **(b) Portable fallback.** The scope runs on the host, with a diagnostic
  that the accelerator was not used.

**Decision: (a), refuse.** A program that believes it ran on an accelerator
must never silently not have. The refusal is the backstop for what cannot be
checked statically; the compile-time checks (§4.2, §6.2) are the mechanism.

This does not make the feature untestable on Linux: the *checker* behaviour is
purely static and fully testable there, and the runtime refusal is a single
well-defined exit path that CI can assert. What it rules out is silently
running GPU code on the host and calling it a pass.

### 4.5 Where MLX fits

MLX remains a **backend**, with two properties:

- On Apple silicon it is the *unified* implementation: transfers are free, one
  pool, no separate capacity.
- It is what the existing tests use, so the placement machinery has a real
  backend to exercise.

It should **not** define the semantics. A portable backend (threaded host
execution, or a real CUDA/ROCm path) implements the same interface: pools,
budgets, transfers.

---

## 5. Staged plan

Ordered so each stage is independently verifiable, and so the cheap,
platform-independent work lands first.

### Stage A — Document the model (no code)

**Files:** `docs/language.md`, `docs/Tutorial/04-hardware-placement.md`,
`docs/guide.md`

1. State that `@cpu`/`@gpu` mean SRAM/VRAM pools, that the pools have
   different capacities, and that Metal's unified memory is the exception.
2. Correct `docs/guide.md:350`, which currently says "`.toGpu()` is a no-op
   transfer" — true on Apple silicon, false in general, and wrong as a
   general statement.
3. Correct `docs/Tutorial/04-hardware-placement.md:99-110` ("compile-time
   discipline", "no kernel lowering") — still true, but state *why*: the
   backend is pluggable and the contract is about pools and budgets.

**Why first:** three documents currently make claims that will become false.
Fixing them while they are still true is cheap; fixing them after Stage C
lands is archaeology.

### Stage B — Checker: pools become usable, budgets introduced

**Files:** `src/semantic/checker.ionic`

1. **Placement-polymorphic builtins.** Builtins accept any tensor pool; user
   `@cpu`/`@gpu` functions stay strict. This keeps
   `examples/mastery/fail/tensor_mismatch.ionic` failing (it uses a *user*
   function) while giving `tensor@gpu` something to do.
   - ~48 edit sites: 21 lines use `bparams_tensor`, 27 use explicit
     `bparam(..., TY_TENSOR_CPU)`.
   - Builtin results infer their pool from the first tensor argument.
2. **Pool declaration and budget scope** (§6.1). The pool size is declared
   once at top level; `@gpu(f)` narrows a scope to a fraction of it.

   **Parser work this implies.** The `@gpu(...)` form already exists and
   already occupies the parenthesised slot: `parser/parser.ionic:108-118`
   reads `@gpu(<float>)` into an `ND_HW` node with `HW_GPU_FRAC`, storing the
   fraction text in the node's `nd_sv`. So:
   - The **fraction syntax needs no parser change** — it is already parsed and
     simply never read. This work gives it a meaning.
   - The **pool declaration needs no new syntax either** if it is an ordinary
     top-level `let` of an integer. But an ordinary `let` is not
     self-describing: `let vram = 25769803776` reads as a number until you
     know what it is. Options: a distinct declaration form, or a convention
     (a top-level `let` named by the build, referenced by name). Recommendation:
     a **named top-level constant** whose name the checker recognises, so the
     pool size is declared once and every `@gpu` scope inherits it.
   - There is **no size-literal support** in the lexer
     (`grep -n 'GB\|MB\|byte\|suffix' src/parser/parser.ionic` → no hits), so
     the pool is declared in **plain bytes**.
3. **Static size computation.** A pass that computes each tensor's size from
   shape and dtype, where knowable.
4. **Budget checking.** Sum live `tensor@gpu` sizes in scope; error if the
   total exceeds the scope's share of the pool, or if any contribution is
   unbounded (decision 4: reject).
5. **Use-after-move** for transfers (§4.3), per decision 1.
6. **Mixed-pool operands** are a compile error when both placements are known
   (§6.2).

**Verify:** `tensor_mismatch.ionic` still fails with the same message;
`04_placement_types.ionic` still compiles; a new over-budget program is
rejected with the arithmetic shown.

**Risk:** the biggest checker change in the project's history. It is also the
highest-value part, because it is the part that is portable and testable
everywhere.

### Stage C — Runtime: pools as a backend interface

**Files:** `src/ionic_model_runtime.c`, `src/codegen/native.ionic`

1. **Pool abstraction.** A tensor handle carries its pool, not just its array.
   Introduce the concept backend-independently; MLX becomes one implementation.
2. **Backend interface.** Operations are dispatched to a backend that owns
   pools, budgets, and transfers. The MLX backend implements it with the
   unified-memory semantics; a portable backend implements it with host
   execution.
3. **Transfers become real.** `toGpu()`/`toCpu()` move a value between pools
   and adjust the budgets. On a unified backend this is free; elsewhere it
   copies and the source is invalidated (move semantics).
4. **`gpu { }` / `@gpu fn` set the ambient pool**, and the ambient pool is a
   *runtime stack* so that a `return` from inside a `gpu {}` block cannot skip
   a restore. (`gpu { return x; }` compiles today — verified — and the codegen
   has a single-exit epilogue, so codegen-managed restore-on-exit is fragile.)
5. **Refusal without an accelerator** (§4.4a): with no accelerator backend, a
   GPU scope reports that no accelerator is present and the program does not
   proceed. It does not silently run on the host.
6. **Stub parity.** The non-MLX branch (`#if IONIC_MLX_ACTIVE`, `:66`) holds
   69 hand-written stubs whose header (`:1670-1677`) warns that a missing stub
   is a **link** error on Linux. New entry points must be added there too.

**Verify:** the MLX example still produces the same numbers; a deliberate
over-budget runtime allocation fails loudly rather than corrupting; a GPU
scope on a machine with no accelerator refuses rather than running on the host.

### Stage D — Capacity enforcement at runtime

Static budget checking catches what is knowable. The rest needs runtime
enforcement: an allocation that exceeds the pool must fail **at the
allocation**, not corrupt memory later.

This is where the "different capacities" part of the model becomes real rather
than advisory, and it needs a per-backend allocator. Deferred because it
depends on the backend existing (Stage C).

### Stage E — Tests

1. Mastery test: budget exceeded → compile error, with the arithmetic.
2. Mastery test: transfer moves a value between pools; use-after-move is an
   error.
3. Mastery test: unified backend (MLX) — transfer is free, one pool.
4. Mastery test: refusal — a GPU scope on a machine with no accelerator exits
   with a clear message rather than running on the host.
5. Runtime test: an allocation exceeding the pool fails cleanly.
6. **CI coverage on Linux** for 1–4. The checker behaviour is testable there
   without any accelerator; the refusal path (4) is a single well-defined exit
   that CI can assert.

---

## 6. Decisions (resolved 2026-11-03)

| # | Question | Decision |
|---|---|---|
| 1 | Move or copy for transfers? | **Move.** Use-after-move is a checker error. Copy is an explicit, separately-named operation. |
| 2 | Budget syntax | **Top-level pool declaration + optional per-function fraction.** See §4.1. |
| 3 | Budget defaults | **A top-level declaration**, overridable at build time for a different card. |
| 4 | Unbounded GPU tensors in a budgeted scope | **Reject.** |
| 5 | GPU scope with no accelerator backend | **Refuse** — do not silently fall back. |

The principle behind 4 and 5: **catch as much as possible at compile time.**
Every check that can be static should be static, and a runtime refusal is a
last resort for what genuinely cannot be known statically.

### 6.1 Decision 2 in detail — where the pool size lives

The pool size is a **deployment** fact (which card is installed), not a
property of a function. Putting a byte count in every function signature
would mean recompiling for a different card and would clutter signatures.
The chosen form:

```
let vram = 25769803776;              // bytes of accelerator memory, deployment-set

@gpu fn train_step(...) -> ... { }   // may use up to the whole pool
@gpu(0.25) fn helper(...) -> ... { } // may use up to a quarter of it
```

- **The pool is declared once**, at top level, and every `@gpu` scope inherits
  it. A build flag (or environment override) changes it for a different card,
  so one source tree serves many deployments.
- **The existing `@gpu(0.5)` fraction syntax finally gets a meaning** — a
  fraction *of the declared pool*. It is currently parsed and discarded
  (§3.4), so this uses syntax that already exists rather than adding new
  syntax.
- **No per-function byte counts**, and therefore no new surface in function
  signatures.

### 6.2 Consequences of "catch as much at compile time"

Decisions 4 and 5 push the plan further than the original draft. Three
additional checks become part of Stage B:

1. **Mixed-pool operands are a compile error when both placements are known.**
   Once builtins are placement-polymorphic (§5 Stage B1), the checker knows
   each operand's pool statically, so `mlx_add(cpu_tensor, gpu_tensor)` can be
   rejected at compile time. Only a *dynamically*-known placement (a call
   whose result pool cannot be inferred) needs a runtime check.

2. **A program that uses any GPU feature declares that requirement at compile
   time.** "This binary requires an accelerator" is derivable from the
   program's contents, so it can be reported at build time and asserted in CI.
   The runtime refusal (§4.4a) then becomes the backstop, not the mechanism.

3. **Dynamic-sized GPU tensors are rejected by default** (decision 4). For the
   genuinely dynamic case — an inference server whose batch size varies — the
   author must state a bound explicitly rather than get an implicit pass.
   A `tensor@gpu` whose size is unbounded would defeat the budget entirely,
   so the default is rejection.

---

## 7. What this plan *does not* do

- **No new tensor type parameters.** Capacity is a scope budget, not
  `tensor@gpu[N]`.
- **No MLX-specific semantics in the language.** MLX is a backend; the
  language describes pools and budgets.
- **No kernel lowering.** Ionic tensor expressions are still not compiled to
  GPU kernels. Ops remain backend primitives.
- **No `bytes` type.**
- **No change to `ty_compat` for user functions.** The strictness that makes
  the feature meaningful is preserved.

---

## 8. Summary

Ionic's placement types are enforced by a sound checker that separates pools
correctly but says nothing about **whether the values fit** — which is the
constraint that matters on a server with a fixed-size accelerator pool. The
runtime has one implicit pool, `tensor@gpu` is a type no operation can
consume, and the transfers that express pool membership compile to nothing.
The work is to make `@cpu`/`@gpu` mean SRAM/VRAM with distinct capacities, to
let the type system enforce a budget on GPU-resident values, and to make
transfers real moves between pools — with MLX retained as the unified-memory
backend it already is, rather than as the definition of the semantics.
