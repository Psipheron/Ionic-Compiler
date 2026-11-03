# 04 — Hardware Placement

**Goal:** understand Ionic's defining feature — memory-pool placement encoded
in the type system — both as a user and as the checker implements it.

---

## 1. The types

```
tensor@cpu   → TY_TENSOR_CPU = 9
tensor@gpu   → TY_TENSOR_GPU = 10
model@cpu    → TY_MODEL_CPU  = 11
model@gpu    → TY_MODEL_GPU  = 12
```

`tensor@cpu` and `tensor@gpu` are **distinct, invariant** types.
`ty_compat(TY_TENSOR_CPU, TY_TENSOR_GPU)` is `false`.

Source: `src/semantic/checker.ionic` (type constants, `ty_compat`).

---

## 2. Explicit transfers

```ionic
let h: tensor@cpu = ...;
let d = h.toGpu();     // tensor@cpu → tensor@gpu   [ND_TO_GPU]
let back = d.toCpu();  // tensor@gpu → tensor@cpu   [ND_TO_CPU]
```

The transfer is the *only* way to change a tensor's placement, and it is a
**move**: the source is consumed, so `h` is dead after `h.toGpu()` and using it
is an error. That is what makes a pool's occupancy meaningful — a value occupies
exactly one pool at a time, which is what lets the compiler check that the GPU
pool is not oversubscribed. If a transfer were a copy, capacity accounting would
have to count both sides, and the point of VRAM capacity would be lost to
duplicated weights.

> **Status (2026-11-03): the move semantics is enforced (stage B5).**
> `let d = h.toGpu(); h;` is now a compile error — *use after move* — and the
> diagnostic names the transfer and the line that consumed the value. So are the
> pool *declaration* and the `@gpu(f)` fraction (stage B2) and the **scope
> budget** (stages B3/B4); see §5 below.
>
> Only `.toGpu()` and `.toCpu()` move. Everything else *borrows*, which is why
> `mlx_add(g, g)` is still legal — MLX handles are refcounted, so reading one
> twice is the normal case. What the pool accounting needs is a single *owner*
> per value, and a transfer is the only operation that reassigns ownership.

Typing rules (`docs/homework_solutions.md` HW 3.2):

```
Γ ⊢ e : tensor@gpu                Γ ⊢ e : tensor@cpu
────────────────────  MOVE-TO-CPU  ────────────────────  MOVE-TO-GPU
Γ ⊢ e.toCpu() : tensor@cpu        Γ ⊢ e.toGpu() : tensor@gpu
```

The checker sets `sc_in_conversion = 1` while checking the operand so the
conversion itself isn't flagged as a cross-device use.

---

## 3. GPU contexts: `gpu { … }` and `@gpu fn`

A **GPU context** is either a `gpu { … }` block or the body of an `@gpu fn`.
Inside one, the checker enforces:

- only `tensor@gpu` values/compute are allowed — a `tensor@cpu` is rejected;
- CPU-only I/O (`print`, `file_read`, `file_write`, …) is rejected;
- calling an `@cpu`-annotated function is rejected.

`@cpu fn` bodies and unannotated functions run in **CPU context**.

A subtlety worth verifying (exercises 04.2/04.4): a `tensor@gpu` **value** may
only be *referenced* inside a GPU context. You may declare
`let d = t.toGpu();` in CPU context, but touching `d` outside a `gpu { … }`
block — **even to pass it to an `@gpu fn`** — is rejected with
`tensor@gpu 'd' used outside gpu block`. So GPU work lives inside the block.

```ionic
@gpu fn train_step(tensor@gpu x, tensor@gpu w) -> tensor@gpu {
    return model_forward(x, w);   // ok
    // print("no");               // error: CPU-only I/O in GPU context
}
```

Reference (correct): `examples/mastery/04_placement_types.ionic`.
Reference (errors): `examples/gpu_error.ionic`, `examples/hw_tag_error.ionic`.

`hw_tag_error.ionic` reports exactly three violations — the CPU-only `print`, the
`@cpu fn` call, and a `tensor@cpu` used inside the GPU context. (The last one is
only reported where the value is *used*; merely declaring it is not the error.)

---

## 4. Contradictory annotations are rejected

A parameter annotation that contradicts the function is impossible to satisfy at
any call site, so it's rejected:

```ionic
@gpu fn bad(@cpu tensor@cpu x) -> tensor@gpu { return x; }   // ERROR
```

Also rejected: an `@gpu` function returning a `tensor@cpu`, or nesting a
`tensor@cpu` inside a GPU context.

---

## 5. Function-level GPU fractions

```ionic
@gpu(0.5) fn partial(...) -> ... { ... }   // at most 50% of the accelerator pool
```

The pool size itself is a **deployment** fact — which card is installed — so it
is declared once, at top level, and every `@gpu` scope inherits it:

```ionic
let vram = 25769803776;                 // bytes of accelerator memory
```

The fraction is a share *of that pool*, not an absolute byte count, so the same
source tree serves a 16 GB card and an 80 GB card.

**What is enforced today (stages B2–B5).** The declaration is read by the checker
and the fraction is validated where it is written. A `@gpu(f)` is rejected when:

- `f` is not in `(0, 1]` — `@gpu(1.5)` and `@gpu(0)` are errors;
- no `vram` pool is declared — a share of an undeclared pool has no meaning;
- `vram` is declared twice, is not a plain integer literal, or is not positive.

On top of that, the **scope budget** is checked. A tensor's size is
`prod(shape) × dtype_bytes`, derived from the constructor (`mlx_zeros`,
`mlx_ones`, `mlx_full`, `mlx_eye`, …) or inherited from a shape-preserving op
(`mlx_add`, `mlx_matmul`, the elementwise family). Every sized `tensor@gpu` that
a `@gpu fn` body or a `gpu { }` block *creates* is summed, and the declaration
that pushes the total past the scope's share is a compile error naming both byte
counts:

```
[error] 7:47: 'c' does not fit the accelerator pool: 4194304 bytes here;
        12582912 bytes live in this scope, budget is 10737418
        (raise the `vram` pool or narrow the scope's fraction)
```

The budget is computed in **integer** arithmetic (`pool × num / den` from the
fraction's text), so no float rounding can invent a budget the author did not
write.

This is a necessary condition, not a proof. A value whose size is not statically
knowable is **exempt rather than rejected**: a tensor *parameter* (it was
allocated and charged in the caller's scope — counting it again would
double-charge), or the result of an op the size pass does not model
(`mlx_reshape`, `mlx_astype`, reductions). Rejecting those would make every
function that takes a tensor uncompilable. Live-range analysis is also not done
yet: the sum covers every sized binding in the scope, so a value that dies before
the next allocation still counts. The check therefore under-approximates — it can
miss an over-budget program, but it never falsely rejects one.

```ionic
let vram = 25769803776;
@gpu(0.25) fn helper(tensor@gpu x) -> tensor@gpu { return x; }   // ok
@gpu(1.5)  fn bad(tensor@gpu x) -> tensor@gpu { return x; }      // error: > 1
```

Reference: `examples/mastery/04_pool_budget.ionic` (must compile).
`scripts/placement_check.sh` pins the rejection cases.

---

## 6. Reality check: is it wired to hardware?

In the shipped tree, the placement rules are a **compile-time discipline**. The
`@gpu` tags, `gpu { }` blocks, transfers, pool declarations and scope budgets are
enforced by the checker; the actual GPU work happens through the backends
(`load_model`, `model_forward`, `piper_forward`, `mlx_*`). Tensor arithmetic in
pure Ionic source is not itself lowered to GPU kernels. Knowing this boundary is
part of mastery: you can rely on the checker to *keep pools separate*, to *reject
a pool fraction that cannot mean anything*, to *reject a scope whose
statically-sized allocations exceed its share of the pool* (stage B4), and to
*reject a use-after-move* once a transfer has consumed a binding (stage B5) —
but not yet to prove every possible over-budget program is caught, and you
should not assume arbitrary `tensor@gpu` math executes on an accelerator.

Nor is the *runtime* enforcing capacity yet: an over-budget program that the
static pass cannot see is not stopped at the allocation. That is stage D.

`@cpu` and `@gpu` name **memory pools** — SRAM and VRAM — not devices. On
Apple silicon the two are physically the same memory, so a transfer there does
not copy and only the accounting changes; everywhere else they are separate and
a transfer really does move bytes. MLX is the backend that ships today and
provides the unified-memory case; a portable or hardware backend implements
the same interface.

Read: `src/ionic_model_runtime.c` (the model/onnx/piper entry points) and
`src/semantic/checker.ionic` (the placement checks) to see both sides.

---

## Exercises

**04.1 — Type mismatch hint.** Construct a program that passes a `tensor@gpu`
where a `tensor@cpu` is expected. Record the diagnostic and confirm it contains
`(call .toCpu() to convert)`. Reference: `examples/mastery/fail/tensor_mismatch.ionic`.

**04.2 — GPU context violation.** Compile `examples/hw_tag_error.ionic` and list
every distinct violation the checker reports — all three are inside the `@gpu fn`:
the CPU-only `print`, the call to the `@cpu fn load`, and the `tensor@cpu` used in
the GPU context. Confirm they are reported together, not one at a time. Then
delete the *use* of that `tensor@cpu` (leaving the `let`) and observe that the
third error disappears: the rule fires on a reference, not a declaration.

**04.3 — Contradiction.** Write `@gpu fn f(@cpu tensor@cpu x) -> tensor@gpu` and
record the rejection.

**04.4 — Correct placement.** Write a program that builds a `tensor@cpu`, calls
`.toGpu()`, and uses the result inside a `gpu { }` block without error.
Reference: `examples/mastery/04_placement_types.ionic`.

**04.5 — Find the enforcement.** Locate the function in
`src/semantic/checker.ionic` that rejects CPU-only I/O inside a GPU context.
Name it and quote the condition. (Hint: search for `gpu` / `sc_in_gpu`.)

**04.6 — Exceed the budget.** Write a program with `let vram = 1073741824;` (1 GiB)
and an `@gpu(0.01)` function that binds three `mlx_zeros([1024, 1024], 0).toGpu()`
locals. Record the diagnostic. Confirm it names the third binding, its own size,
the running total, and the budget. Then raise the fraction to `0.5` and confirm it
compiles. Explain why `@gpu(0.01)` gives 10 737 418 bytes and not 10 737 418.24.

**04.7 — Why is a parameter exempt?** Take `@gpu(0.01) fn f(tensor@gpu x) -> ...`
under the same 1 GiB pool. Confirm it compiles even though `x` alone may exceed
the budget. Then explain, in one sentence, what breaks if the checker rejected it.
(Hint: how many existing examples pass a `tensor@gpu` to a function?)

**04.8 — Use after move.** Write a program that builds a `tensor@cpu`, transfers
it with `.toGpu()`, and then uses the *original* binding — say, pass it to
`mlx_free`. Confirm the diagnostic names both the binding and the line of the
transfer that consumed it. Then fix it, and confirm each of these is *accepted*:

- `mlx_add(g, g)` — a builtin reads its operand twice (it borrows);
- `mlx_ones([2, 2], 0).toGpu()` — a temporary has no binding to strand;
- `let g = a.toGpu(); let a = mlx_ones(...);` — rebinding clears the mark.

Explain why treating a *call* as a move would make the first of those illegal,
and why that would defeat the purpose of the move rule.

**04.9 — Which line is blamed?** Take the two-transfer program from exercise 04.8
and remove the use of the moved binding, but transfer the same value twice
(`let g = a.toGpu(); let h = a.toGpu();`). Confirm the error is reported on the
*second* transfer, naming the first. (Hint: `sc_move_binding` is idempotent so the
original provenance survives; `sc_check_not_moved` does the reporting.)

---

## Checkpoint 04

- [ ] You can state the two transfer rules from memory.
- [ ] You can predict whether a given `gpu { }` body is rejected, and why.
- [ ] You can explain why a tensor *parameter* is exempt from the scope budget.
- [ ] You can explain why only transfers move while builtins borrow.
- [ ] You know where placement is enforced and where it is *not* (no kernel lowering,
      no runtime capacity enforcement).
- [ ] You can produce both the mismatch hint and the contradiction error.

## [Next](05-stdlib-and-modules.md)
