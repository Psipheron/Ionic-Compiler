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
**move**: the source is consumed, so `h` is dead after `h.toGpu()` and using
it is an error. That is what makes a pool's occupancy meaningful — a value
occupies exactly one pool at a time, which is what lets the compiler check
that the GPU pool is not oversubscribed.

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

---

## 6. Reality check: is it wired to hardware?

In the shipped tree, the placement rules are a **compile-time discipline**. The
`@gpu` tags, `gpu { }` blocks, transfers, and pool budgets are enforced by the
checker; the actual GPU work happens through the backends (`load_model`,
`model_forward`, `piper_forward`, `mlx_*`). Tensor arithmetic in pure Ionic
source is not itself lowered to GPU kernels. Knowing this boundary is part of
mastery: you can rely on the checker to *keep pools separate* and to *keep the
GPU pool within budget*, but you should not assume arbitrary `tensor@gpu` math
executes on an accelerator.

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
every distinct violation the checker reports (two inside the `@gpu fn`: the
CPU-only `print`, and the call to the `@cpu fn load`). Confirm they are reported
together, not one at a time.

**04.3 — Contradiction.** Write `@gpu fn f(@cpu tensor@cpu x) -> tensor@gpu` and
record the rejection.

**04.4 — Correct placement.** Write a program that builds a `tensor@cpu`, calls
`.toGpu()`, and uses the result inside a `gpu { }` block without error.
Reference: `examples/mastery/04_placement_types.ionic`.

**04.5 — Find the enforcement.** Locate the function in
`src/semantic/checker.ionic` that rejects CPU-only I/O inside a GPU context.
Name it and quote the condition. (Hint: search for `gpu` / `sc_in_gpu`.)

---

## Checkpoint 04

- [ ] You can state the two transfer rules from memory.
- [ ] You can predict whether a given `gpu { }` body is rejected, and why.
- [ ] You know where placement is enforced and where it is *not* (no kernel lowering).
- [ ] You can produce both the mismatch hint and the contradiction error.

## [Next](05-stdlib-and-modules.md)
