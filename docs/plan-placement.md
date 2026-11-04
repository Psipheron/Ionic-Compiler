# Functional Hardware-Placement Awareness

Status: **Stage A complete, Stage B (B1–B6) complete, Stage C partially complete
(pool as a runtime fact), Stage D items 2 and 5 complete (allocator backstop,
device-capacity comparison)** — revised 2026-11-03
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
   and to keep the build multiplatform. On Apple silicon it is the **unified
   memory** backend: it is what the pool model is exercised and tested against
   there, and it is a legitimate deployment backend on that platform. It is
   *not* the definition of the model, and it must not shape the semantics —
   the Linux VRAM side needs real SRAM/VRAM pools, which MLX does not provide
   today. Settled: MLX is the unified-memory backend, not "the GPU backend".
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
| A transfer consumes its operand (no use-after-move) | `sv_mvline`/`sv_mvhow`, `sc_consume_operand`, `sc_check_not_moved` | works (B5) |

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

> **Fixed in Stage B1.** See §3.3a. The audit below is kept because it is the
> reason B1 exists and the shape of the fix follows from it.

- **No builtin accepts `tensor@gpu`.** 70 `mlx_*` builtins are registered
  (`checker.ionic:821-940`); 47 declare `TY_TENSOR_CPU` as their return type,
  and the rest take tensor parameters via `bparams_tensor` (`:650-654`), which
  pins every parameter slot to `TY_TENSOR_CPU`. `TY_TENSOR_GPU` appears only in
  type machinery, never in a signature.
- **There is no tensor arithmetic.** Verified: two `tensor@gpu` operands to `+`
  yield `type error in arithmetic: expected numeric type, got tensor@gpu`.

A `tensor@gpu` can be created, passed to `@gpu` functions, and returned — and
nothing can consume it.

### 3.3a Stage B1: builtins are pool-polymorphic *(implemented 2026-11-03)*

The dead type is fixed by one rule, derived from the registration table rather
than restated per builtin:

    pool-polymorphic  ⇔  builtin ∧ returns a tensor ∧ takes a tensor operand

Concretely, in `src/semantic/checker.ionic`:

- `fp_anyplace[]` is parallel to `fp_tys[]`. `bparam()` sets it for any tensor
  parameter of a builtin, so `bparams_tensor()` and every explicit
  `bparam(..., TY_TENSOR_CPU)` are covered by one edit rather than 48. A user
  function's parameters are never marked, which is what preserves strictness.
- `fn_pool_poly[]` is computed by `finalize_pool_poly()` after registration. It
  marks the builtins whose *result* pool is inferred from their operands.
- At a call site the checker tracks the pool of the first tensor operand and
  reports any operand that disagrees. The result type becomes the first
  operand's pool when `fn_pool_poly` is set.

`ty_compat` is deliberately untouched: it stays strict for every call, and only
the builtin argument check consults `fp_anyplace`. That is why
`examples/mastery/fail/tensor_mismatch.ionic` still fails with the identical
diagnostic — it calls a *user* function.

Two consequences worth noting:

- **Constructors stay SRAM.** `mlx_zeros`, `mlx_ones`, `mlx_from_array` and
  friends take no tensor operand, so there is nothing to infer from and they
  allocate in SRAM. Entering VRAM is an explicit `.toGpu()`, which is also the
  operation a capacity budget would be charged for.
- **Mixed-pool operands are rejected** (§6.2.1). This is not an added
  restriction; it is what makes result-pool inference well defined. MLX does
  not check it — a mixed-device add returns silently wrong data — so the
  compiler must.

Regression coverage: `examples/mastery/04_pool_polymorphism.ionic` (compiles and
runs) and `examples/mastery/fail/mixed_pool.ionic` (rejected with exactly one
diagnostic, the intended one).

### 3.4 The runtime is inverted — **fixed in Stage C (pool as a runtime fact)**

`MlxTensor` was `{ magic, alive, mlx_array a }` — no pool field — and every MLX
call site hardcoded `g_mlx_gpu_stream`. Every tensor was GPU-resident regardless
of its Ionic type, so `tensor@cpu` was already VRAM and `tensor@gpu` was
unreachable. `toGpu()`/`toCpu()` compiled to an identity (`native.ionic:1375`)
and `gpu { }` compiled to nothing.

What landed instead (see Stage C):

- `MlxTensor` carries `pool` and `bytes`; `mlx_tensor_new_at(a, pool)` is the
  one constructor, and `mlx_tensor_new(a)` defaults to SRAM.
- `pool_stream(pool)` replaces all 36 hardcoded stream references, so an
  operation runs on the stream belonging to its handle's pool.
- `toGpu()` / `toCpu()` are real runtime calls (`ionic_mlx_to_gpu` /
  `ionic_mlx_to_cpu`) that move the value and charge the destination pool.
- `mlx_pool`, `mlx_pool_bytes`, `mlx_pool_count` and `mlx_pools_unified` make
  the pool model observable from Ionic, and therefore testable — a harness can
  now tell a transfer that works from one that silently does nothing.

`gpu { }` still compiles to its body plus a comment; it is a checked *claim*
about the pool, not a scheduling primitive, and the refusal path (§4.4a) is not
wired yet.

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
   contributions. Decision 4 says the checker rejects a statically-unbounded
   GPU tensor in a budgeted scope rather than waving it through. **As
   implemented (B4), the rejection is narrowed to values *created* in the
   scope**: a tensor *parameter* has no static size, and rejecting it would
   make every function that takes a tensor uncompilable. See the note under
   Stage B item 4 for the reasoning and the resulting "necessary but not
   sufficient" property.

### 4.3 The ownership question: is a transfer a move or a copy?

This is the sharpest decision, and it is **not** answered by the current code.

If `.toGpu()` is a **move**, then after `let g = x.toGpu();` the CPU-side `x`
is dead. Using `x` afterwards is a use-after-move — a *new class of compile
error* the language did not have before B5 (until then there was no
`moved`/`consumed` tracking anywhere in `src/`).

If `.toGpu()` is a **copy**, then capacity accounting must count both sides,
and the whole point of VRAM capacity is lost to duplicated weights.

**Resolved as a move**, with use-after-move as a checker error (B5, 2026-11-03;
see Stage B item 5 for what landed). Reasons:

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

1. **Placement-polymorphic builtins.** ✅ **Done, see §3.3a.** Builtins accept
   any tensor pool; user `@cpu`/`@gpu` functions stay strict. This keeps
   `examples/mastery/fail/tensor_mismatch.ionic` failing (it uses a *user*
   function) while giving `tensor@gpu` something to do.
   - The audit predicted ~48 edit sites. Folding the rule into `bparam()`
     reduced it to one: a parallel `fp_anyplace[]` table marked there covers
     both `bparams_tensor` and every explicit `bparam(..., TY_TENSOR_CPU)`.
   - Builtin results infer their pool from the first tensor argument
     (`fn_pool_poly[]`, computed by `finalize_pool_poly()`).
2. **Pool declaration and budget scope** (§6.1). The pool size is declared
   once at top level; `@gpu(f)` narrows a scope to a fraction of it.
   ✅ **Done (the declaration and the fraction's validity) — 2026-11-03.**

   What landed, in `src/semantic/checker.ionic`:
   - `pool_scan_toplevel()` runs in `sc_program` *before* any function body is
     checked, and reads the top-level `let vram = <int>` into
     `pool_vram_bytes`. It rejects a duplicate declaration, a non-integer
     initialiser, and a non-positive size.
   - `hw_frac_bits()` reads the fraction text the parser already stored in
     `nd_sv[hw_node]` (it was parsed and never read); `hw_frac_is_valid()`
     checks `0 < f <= 1` by comparing IEEE-754 bit patterns, so no float
     arithmetic happens in the checker.
   - `sc_check_hw_frac()` is called from `sc_fn` and rejects a fraction outside
     `(0, 1]`, and a fraction when no pool is declared.

   No parser change was needed, as the audit predicted: `@gpu(<float>)` already
   produced `HW_GPU_FRAC` with the text attached, and `let vram = <int>` is an
   ordinary global. The name is a convention (`POOL_DECL_NAME`), not a keyword.

   **What this does *not* do:** it does not compute whether a set of live
   tensors fits the resulting budget. That is B3 (static sizes) + B4 (sum and
   compare), which landed separately below. Until they landed the fraction was
   a validated claim about intent rather than a proof about capacity, and
   `docs/guide.md` / `docs/Tutorial/04-hardware-placement.md` were corrected to
   say so.

   Regression coverage: `scripts/placement_check.sh` (35 assertions across the
   B2, B3/B4 and B5 sections, all static and accelerator-free, run in both
   `ci.yml` and the Linux `elf` job in `multitarget.yml`) and
   `examples/mastery/04_pool_budget.ionic`.

   **Parser work this implied — none.** The `@gpu(...)` form already existed:
   `parser/parser.ionic:108-118` reads `@gpu(<float>)` into an `ND_HW` node with
   `HW_GPU_FRAC` and the fraction text in `nd_sv`. The audit's prediction held —
   the syntax was already parsed and simply never read, so giving it a meaning
   needed no parser change. The pool declaration is likewise an ordinary
   top-level `let` of an integer (there is no size-literal support in the lexer,
   so it is written in plain bytes). The only design choice actually taken was
   to recognise the declaration **by name** (`POOL_DECL_NAME = "vram"`) rather
   than add a keyword.
3. **Static size computation.** ✅ **Done (B3) — 2026-11-03.** A tensor's size
   is `prod(shape) x dtype_bytes`. `expr_bytes()` derives it for the
   constructor builtins (`mlx_zeros`, `mlx_ones`, `mlx_full`, `mlx_eye`,
   `mlx_random_normal`, `mlx_random_uniform`) and for shape-preserving ops
   (`mlx_add`, `mlx_matmul`, the unary elementwise family, …), which report
   their first operand's size. `shape_lit_count()` requires every shape element
   to be a non-negative integer literal, so a shape built at runtime is
   "unknown" rather than guessed. `dtype_bytes()` mirrors the runtime's
   `IonicDtype` widths. Sizes travel in a dedicated `sv_bytes[]` column
   (`sv_set_bytes`/`sv_lookup_bytes`), separate from `sv_svs[]` which already
   carries a struct name or an array element type.
4. **Budget checking.** ✅ **Done (B4) — 2026-11-03, with decision 4 narrowed.**
   A `@gpu` fn body and a bare `gpu { }` block are both budgeted scopes:
   `budget_begin()` computes the scope's share as `pool_vram_bytes * num / den`
   in **integer** arithmetic (the fraction text is parsed as an exact ratio by
   `frac_num`/`frac_den`, so no float rounding can invent a budget), and
   `budget_charge()` sums each sized `let`-bound `tensor@gpu` local, reporting
   at the declaration that pushes the total over and naming the byte counts.

   **Decision 4 was narrowed rather than implemented literally.** "Reject
   unbounded" would reject every tensor *parameter*, since a parameter has no
   static size — and that breaks all four existing `@gpu fn(tensor@gpu x)`
   examples CI asserts must compile. So the check applies where a value is
   *created* in a budgeted scope, and exempts values whose size is unknown (a
   parameter, or the result of an op the size pass does not model). A parameter
   was allocated — and charged — in the caller's scope; counting it again here
   would double-charge it. The consequence is that the check is a **necessary
   but not sufficient** condition: it catches "I allocated 4 GiB against a 1%
   budget" and under-approximates. It cannot produce a false rejection, which
   is the property that matters for a check users cannot turn off.
5. **Use-after-move** for transfers (§4.3), per decision 1. ✅ **Done (B5).**
   `x.toGpu()` / `x.toCpu()` consumes `x`; reading the binding afterwards is
   rejected, and the diagnostic names the transfer and the line it happened on.
   The checker tracks this in the `sv_mvline` / `sv_mvhow` symbol-table columns:
   `sc_consume_operand()` marks the transferred identifier, `sv_declare()`
   clears the mark, and `sc_check_not_moved()` fires from `ND_IDENT` — so every
   reference is covered by one rule, whether it appears as an argument, an
   initialiser or a return value.

   **Only transfers move.** Every other operation borrows, which is why
   `mlx_add(g, g)` stays legal: MLX arrays are refcounted handles, so reading a
   handle twice is the normal case, and if calls consumed their operands every
   expression would need a copy of its inputs. What capacity accounting needs
   is one *owner* per value, and a transfer is the only operation that
   reassigns ownership.

   Two consequences worth knowing: a temporary has no binding to strand, so
   `mlx_ones([2, 2], 0).toGpu()` is fine; and rebinding clears the mark, so
   `let g = a.toGpu(); let a = mlx_ones(...);` is legal — the old value moved,
   the new one is a distinct value in a reused slot.

6. **Mixed-pool operands** are a compile error when both placements are known
   (§6.2). ✅ **Done** — it is also required for B1's result-pool inference to
   be well defined, so it landed with B1 rather than after it.

**Verify:** `tensor_mismatch.ionic` still fails with the same message ✅;
`04_placement_types.ionic` still compiles ✅; a malformed pool declaration or an
out-of-range `@gpu(f)` is rejected ✅ (B2); an over-budget program is rejected
with the arithmetic shown ✅ (B3/B4); reading a transferred binding is rejected
✅ (B5) — see `scripts/placement_check.sh`, whose `static sizes and the scope
budget` section covers an over-budget allocation, the same allocation under a
wider fraction, dtype affecting the size, `gpu { }` blocks being budgeted,
parameters and unsized results being exempt, and "no pool declared means no
budget", and whose `transfers move` section covers use-after-move in both
directions, freeing a moved value, and the four patterns that must stay legal
(builtins borrowing, transferring a temporary, rebinding a name).

**Risk:** the biggest checker change in the project's history. It is also the
highest-value part, because it is the part that is portable and testable
everywhere.

### Stage C — Runtime: pools as a backend interface

**Files:** `src/ionic_model_runtime.c`, `src/codegen/native.ionic`

Items 1 and 3 are **done** (the pool model is a runtime fact as of 2026-11-03).
Items 2, 4 and 5 remain, and item 6 was satisfied for the new entry points.

1. ✅ **Pool abstraction.** A tensor handle carries its pool, not just its
   array. `MlxTensor` gained `int64_t pool` and `int64_t bytes`; `pool_stream()`
   maps a pool to its stream; per-pool byte counters make occupancy answerable.
   MLX is the unified-memory implementation of this, not the concept.
2. **Backend interface.** Operations are dispatched to a backend that owns
   pools, budgets, and transfers. MLX is one implementation behind that seam;
   a portable or CUDA/ROCm backend would be another. **Not done** — today the
   pool model is expressed directly in the MLX runtime, which is enough for the
   unified-memory case and is the shape a second backend would be factored out
   of, rather than a shape it would have to be retrofitted into.
3. ✅ **Transfers become real.** `toGpu()`/`toCpu()` move a value between pools
   and adjust the counters. On the unified backend no bytes move — the array
   reference is handed over and the source pool's charge is released — while a
   discrete backend (`IONIC_POOLS_DISCRETE`) copies onto the destination stream
   before invalidating the source. Both are a move: the source stops owning its
   array either way.
4. **`gpu { }` / `@gpu fn` set the ambient pool**, and the ambient pool is a
   *runtime stack* so that a `return` from inside a `gpu {}` block cannot skip
   a restore. (`gpu { return x; }` compiles today — verified — and the codegen
   has a single-exit epilogue, so codegen-managed restore-on-exit is fragile.)
   **Not done** — the pool is carried by each handle instead, which removes the
   need for an ambient stack for the accounting half. An ambient pool would
   still be needed to *choose a stream* for an operation whose handle is
   pool-typed but whose operands disagree, and the checker already rejects that
   (§6.2.1), so there is nothing for it to do yet.
5. **Refusal without an accelerator** (§4.4a): with no accelerator backend, a
   GPU scope reports that no accelerator is present and the program does not
   proceed. It does not silently run on the host. **Not done.** Note the
   interaction with the spike outcome: on a build with no CUDA/Metal,
   `mlx_default_gpu_stream_new()` still returns a stream for a device that does
   not exist, so the refusal must be decided by asking the backend
   (`mlx_cuda_is_available` / `mlx_metal_is_available`) and treating the GPU
   pool as *absent*, not merely empty.
6. ✅ **Stub parity.** The non-MLX branch (`#if IONIC_MLX_ACTIVE`) holds the
   hand-written stubs whose header warns that a missing stub is a **link** error
   on Linux. Every new entry point (`ionic_mlx_to_gpu`, `ionic_mlx_to_cpu`,
   `ionic_mlx_pool`, `ionic_mlx_pool_bytes`, `ionic_mlx_pool_count`,
   `ionic_mlx_pools_unified`) was added there too.

**Verify:** the MLX example still produces the same numbers ✅; transfers move
bytes between pools and are charged exactly once ✅ (the `pool model at runtime`
section of `scripts/placement_check.sh` asserts both directions, the
no-double-counting property, and that a free returns the pool to its baseline);
a GPU scope on a machine with no accelerator refuses rather than running on the
host — **not yet implemented**, so not yet assertable. What that case needs is
an accelerator-free *build* of the MLX backend; a non-MLX build answers "no
backend" rather than "backend present, no device", which is a different question.

### Stage D — Capacity enforcement at runtime

Static budget checking catches what is knowable. The rest needs runtime
enforcement: an allocation that exceeds the pool must fail **at the
allocation**, not corrupt memory later.

This is where the "different capacities" part of the model becomes real rather
than advisory.

**Item 2 (the allocator backstop) is done — 2026-11-03, and so is item 5 (the
device-capacity comparison).** Items 3 and 4 remain.

What landed:

- **The declared capacity is published, not just checked.** `ionic_pool_set_capacity(pool, bytes)`
  is emitted by `ncg_main_wrapper` immediately after `ionic_runtime_init`, taking
  the number from the same `let vram = N;` the checker read. `vram` is an
  immutable integer literal, so codegen inlines it as a compile-time constant and
  it has **no runtime storage** — which is exactly why the value has to be
  emitted at startup rather than read from a global. One declaration, one number,
  both halves.
- **Pool identity and capacity live outside `#if IONIC_MLX_ACTIVE`.** The pool
  model is a language concept, not a backend detail, so `IONIC_POOL_CPU` /
  `_GPU` / `_NONE` / `_COUNT` and `g_pool_capacity[]` are declared near the top of
  `src/ionic_model_runtime.c`. Only the MLX half — streams and byte accounting —
  stays inside the conditional. Consequence: `mlx_pool_capacity()` answers
  correctly on a platform with no backend at all, because the question "what did
  the program declare" does not require an accelerator.
- **The refusal is at the charge site.** `pool_charge()` is the single place
  bytes are accounted, so the check lives there: a positive request that does not
  fit `capacity - used` calls `pool_capacity_panic()`, which reports the request,
  the pool, the current occupancy, the declared capacity, and where the capacity
  came from. The comparison is written as `bytes > cap - used` rather than
  `used + bytes > cap` so a huge request cannot overflow into a negative sum and
  slip past.
- **`exit(1)`, not a NULL handle.** Ionic has no way to observe NULL from an
  `mlx_*` call — every one returns a tensor or a scalar — so returning NULL would
  turn an over-budget program into a segfault at an unrelated line, which is the
  "corrupt memory later" this exists to prevent. `ionic_bounds_panic` makes the
  same choice for the same reason, and §4.4a's "refuse rather than fall back"
  principle is the same argument one level up.
- **`mlx_pool_capacity(pool)` is a new builtin** so a program can read back what
  it declared and compare it against the device (item 5's prerequisite). It
  returns `-1` for a pool id that does not exist and `0` for a declared-but-
  unbounded pool, so "no such pool" and "no limit" stay distinguishable.

**Item 5 (comparing the declaration against the device) is done — 2026-11-03.**
What landed:

- **`mlx_device_capacity(pool)`** answers the other half of the question: what
  the machine offers, rather than what the source asked for. For VRAM that is
  Metal's `max_recommended_working_set_size`, reached through
  `mlx_device_new_type` → `mlx_device_info_new` → `mlx_device_info_get` →
  `mlx_device_info_get_size`. It is deliberately **not** `memory_size`: on this
  8 GiB machine `memory_size` reports 8 GiB while the working set is 5.33 GiB,
  and the gap between those two numbers is exactly what the check exists to
  catch.
- **The comparison is a warning, and it is lazy.** `pool_check_declaration()`
  runs from `pool_charge()` on the first allocation that touches a pool, once
  per pool. It warns when the declaration exceeds the device and says which two
  numbers disagree. It does not refuse: a budget larger than today's machine is
  a program that will run on a bigger one, and the allocator still refuses any
  individual allocation that does not fit. Startup would have been the wrong
  site — the repo's own `vram` examples declare 24 GiB and never allocate, so a
  startup check would warn on programs that are simply not running yet. A
  declaration only becomes a claim about *this* machine when the program asks
  this machine for accelerator bytes.
- **Non-MLX stub parity.** `ionic_mlx_device_capacity` exists in both branches:
  SRAM is answerable everywhere (physical host RAM is a fact about any machine),
  and VRAM reports `0` — "unknown" — where no accelerator is linked in, rather
  than pretending.

Verified on this machine: SRAM 8.0 GiB, VRAM 5.3 GiB, and a `let vram = 24 GiB;`
program that allocates prints the warning exactly once and continues.


**Why the probe that motivated this was the right probe.** `let vram = 67108864;`
plus `mlx_zeros([n, n], 0).toGpu()` with `n = 65536` allocates 16 GiB. The static
check exempts it and warns, by design (D.0); before this change the program then
ran to completion and exited 0, so the 64 MiB declaration was pure fiction. It
now aborts with `ionic: accelerator pool exhausted: a 17179869184-byte allocation
does not fit VRAM (0 of 67108864 bytes in use)` and exit 1. Both halves are
asserted in `scripts/placement_check.sh` (`the allocator backstop` section), which
also checks that the declared number is the one the runtime reports.

**Still missing:**

- **Live-range analysis**, so a value that is provably dead stops being charged.
  Today the runtime release is driven by `mlx_free` and by a transfer, so a value
  that simply falls out of scope keeps its charge until the process ends. That is
  the same over-approximation the static check makes, and the same piece of work
  fixes both.
- **Interprocedural peak.** A callee's allocation is charged to the callee's
  frame and released on return, which is correct for a real peak but means the
  caller's budget is not consulted. A caller holding 1 GiB can call a function
  that allocates another 1 GiB against a 1 GiB pool, and neither the static check
  nor the allocator sees the combination until the *sum* exceeds the pool at the
  second allocation — which does abort, but with the inner function named rather
  than the caller.
- **A pool-agnostic charge path.** `pool_charge` currently lives in the MLX
  section because that is where `g_pool_bytes[]` is maintained. A second backend
  would need it to move out alongside `g_pool_capacity[]`, which is the Stage C
  item 2 factoring.

### D.0 What the static check does *not* catch — measured, not assumed

The goal is "pool exhaustion is a build error". The static check is a **necessary
but not sufficient** condition today, and the difference is worth stating
exactly, because every one of these was confirmed by compiling a program and
watching it succeed. None of them can produce a *false* rejection; all of them
can miss an over-budget program.

| Gap | Confirmed by | Consequence |
|---|---|---|
| **No live-range analysis.** A binding keeps its charge for the whole scope, even after `mlx_free` or after it is provably dead. | Four 256 MiB values plus a freed fifth are rejected: 5 x 256 MiB is counted, though only 4 are live. | Over-approximates the peak, so it can reject a program that would fit. Conservative direction, but it is the *opposite* error from the ones below. |
| **Interprocedural peak is not modelled.** `let n = inner()` where the caller already holds 1 GiB is not charged for what `inner` allocates. | A `@gpu` fn holding 1 GiB calls another that allocates 1 GiB; both compile, and the real peak is 2 GiB against a 1 GiB pool. | Under-approximates: the classic over-budget-at-3am case, still reachable. |
| **Runtime-sized shapes are unbounded, and only warned about.** A shape element that is not an integer literal makes the size unknown, and unknown values are exempt (decision 4, narrowed). | `mlx_zeros([n, n], 0).toGpu()` in a budgeted scope still compiles for any `n`, but now prints a `[warn]` naming the value and the scope's known total. | Under-approximates. The exemption is *reported* rather than counted as a rejection, so "this fits" and "this was never sized" are distinguishable — but nothing enforces it. |
| **The declared `vram` is never checked against the device** *statically*. The pool size is a compile-time literal and no compilation step knows what machine will run the binary. | Nothing at build time compares it to the backend's real memory. The runtime now compares it at the first allocation and warns. | A program can still declare more VRAM than exists and be "correct"; it is now told so when it runs, rather than believing it. |
| **`gpu { }` inside `@gpu(f)` resets the running total** instead of adding to it. | Noted at `budget_begin(0)` in `checker.ionic`; the nested block checks its own locals against the whole pool. | Under-approximates for nested scopes. |

Two of these are the difference between "catches obvious mistakes" and "makes
pool exhaustion a build error": **interprocedural peak** and **runtime-sized
shapes**. Until they land, the honest claim is the narrower one — the compiler
rejects statically-evident over-budget scopes, and Stage D's allocator is what
catches the rest at run time.

Ordering, if this is to become the goal it claims to be:

1. ✅ **Report unknown sizes instead of exempting them silently.** Done, in
   `budget_end()` (`src/semantic/checker.ionic`): when a budgeted scope closes,
   it emits a `[warn]` naming the first un-sized value, how many there were, and
   what the checker could actually size against the budget. It is a warning, not
   an error — an unbounded value is not provably wrong, so the exemption stays an
   exemption, it is just no longer invisible. `sc_unknown_count` /
   `sc_unknown_first` are reset per scope and saved/restored around a nested
   `gpu { }`, so one value produces one warning.
2. ✅ **Stage D's allocator.** Done — see Stage D. The program that gets past
   the checker now fails loudly at the allocation rather than at an OOM hours
   later. The ordering list below stays because items 3–5 are still open.
3. **Live-range analysis**, which converts the conservative over-approximation
   into an accurate one and makes the check *sufficient* for values it can size.
4. **Interprocedural peak** — needs call-graph summaries (what a callee allocates
   and whether it frees), which is the largest piece of work here.
5. **A compile-time device-capacity check** is not possible and never will be: a
   compile knows nothing about the machine that will run the binary. The
   *runtime* comparison landed instead (Stage D item 5) — it warns, once per
   pool, at the first allocation that touches it.

### Stage E — Tests

1. Mastery test: budget exceeded → compile error, with the arithmetic.
   ✅ `04_pool_budget.ionic` + the `static sizes and the scope budget` section of
   `scripts/placement_check.sh`.
2. Mastery test: transfer moves a value between pools; use-after-move is an
   error. ✅ **Done** — `examples/mastery/fail/use_after_move.ionic` (rejected,
   asserted in `ci.yml` and the harness) plus the `transfers move` section of
   `scripts/placement_check.sh`: 8 cases covering both directions, freeing a
   moved value, and the patterns that must stay legal.
3. Mastery test: unified backend (MLX) — transfer is free, one pool.
   ✅ **Done** — the `pool model at runtime` section of
   `scripts/placement_check.sh` asserts that a transfer on the unified backend
   moves the *accounting* between pools without double counting, that a
   constructor is charged to SRAM and not VRAM, that a free returns the pool to
   its baseline, and that builtin results keep their operands' pool. It skips
   (and says so) on a build with no MLX backend linked.
4. Mastery test: refusal — a GPU scope on a machine with no accelerator exits
   with a clear message rather than running on the host.
   **Not done**, and not assertable yet: it needs an *accelerator-free build* of
   the MLX backend (backend present, no Metal device). A non-MLX build answers
   "no backend", which is a different question and is already the case today.
5. Runtime test: an allocation exceeding the pool fails cleanly.
   ✅ **Done** — `the allocator backstop` section of `scripts/placement_check.sh`:
   a 16 GiB request against a declared 64 MiB pool aborts with
   `accelerator pool exhausted` and a non-zero exit, and a second case asserts
   that `mlx_pool_capacity(1)` returns exactly what `let vram = N;` declared (and
   `-1` for a pool id that does not exist). The harness distinguishes "refused"
   from "crashed differently" by also requiring the runtime's own message, and
   skips (loudly) on a build with no backend.
6. **CI coverage on Linux** for 1–4. The checker behaviour is testable there
   without any accelerator; the refusal path (4) is a single well-defined exit
   that CI can assert. Note that the *allocator* half (5) cannot run on Linux
   until a second backend exists — the pool charge path lives in the MLX section.

---

## 6. Decisions (resolved 2026-11-03)

| # | Question | Decision |
|---|---|---|
| 1 | Move or copy for transfers? | **Move.** Use-after-move is a checker error. Copy is an explicit, separately-named operation. |
| 2 | Budget syntax | **Top-level pool declaration + optional per-function fraction.** See §4.1. |
| 3 | Budget defaults | **A top-level declaration**, overridable at build time for a different card. |
| 4 | Unbounded GPU tensors in a budgeted scope | **Reject** — as implemented, narrowed to values *created* in the scope; a tensor parameter is exempt (see Stage B item 4). |
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
  fraction *of the declared pool*. It was previously parsed and discarded
  (§3.4); B2 gave it a meaning without adding new syntax.
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
   so the default is rejection. **As implemented (B4)**, "created in the scope"
   is the boundary: a statically-unsized value *produced* in a budgeted scope is
   counted as unknown rather than rejected outright, because a tensor
   *parameter* has no static size and rejecting it would make every function
   that takes a tensor uncompilable. See Stage B item 4.

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
correctly and, since Stage B, also checks whether a scope's values fit its
declared budget. As of Stage C the runtime matches the types: a tensor handle
carries its pool, each pool owns a stream, and a transfer is a real move that
charges the destination pool — so `tensor@cpu` and `tensor@gpu` differ at run
time and not only in the checker. Since Stage D items 2 and 5 the declaration is
both enforced and checked against reality: the runtime is told the declared
capacity at startup, refuses an allocation that would not fit, and warns — once
per pool, at the first allocation — when the declaration exceeds what the device
actually offers. So a program that gets past the static check cannot go on to
allocate ten thousand times its stated budget, and a program whose declaration is
fiction for this machine is told which two numbers disagree instead of believing
it. MLX is retained as the unified-memory backend, the point in the design space
where a transfer moves accounting but not bytes, rather than as the definition of
the semantics. Remaining: live-range analysis and interprocedural peak (which
together make the *static* check sufficient rather than necessary), the
ambient-pool runtime stack, and the refusal path for a GPU scope on a machine with
no accelerator.
