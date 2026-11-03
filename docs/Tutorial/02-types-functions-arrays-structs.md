# 02 — Types, Functions, Arrays, Structs

**Goal:** understand how Ionic infers and checks types, how arrays really behave
(including a real bug in the stdlib), and how structs and methods work.

---

## 1. Function declarations

```ionic
fn add(int64 a, int64 b) -> int64 { return a + b; }
fn greet(string name) -> string { return str_concat("Hello, ", name); }
```

- Parameters are written `type name`; the return type follows `->`.
- Functions may be defined **after** their first call — the compiler does a
  full-program pass before codegen (see `src/main.ionic`: parse → `sc_program`).
- Recursion is fully supported (`examples/showcase.ionic:power`, `is_prime`).

---

## 2. Inference: the shape of the algorithm

Inference is **propagation-based**, not full Hindley–Milner. There are no
unification variables; types are monomorphic per expression. The checker
`sc_expr` in `src/semantic/checker.ionic` walks the AST and sets two globals:
`sc_ty` (the resulting type) and `sc_elem_ty` (array element type).

Key rules (see `docs/homework_solutions.md` HW 3.3 for the full trace):

- `ND_INT → int64`, `ND_FLOAT → float64`, `ND_STR → string`, `ND_BOOL → bool`.
- `ND_BINOP`: if either operand is `float64`, the result is `float64` and the **op
  constant is rewritten in the AST** (`OP_ADD → OP_FADD`, etc.). Otherwise `int64`.
- Comparison/logical ops return `bool`.
- `ND_CALL`: return type from the function's declared signature.
- Unknown names yield `TY_UNKNOWN`, which is "compatible with everything" to
  avoid error cascades.

This is the ATAPL §5 idea: inference mutates the AST so codegen needs no type
re-derivation for arithmetic.

---

## 3. Arrays — the ground truth

Arrays are dynamic (24-byte header `[len][cap][data_ptr]` + a heap data region in
`src/ionic_model_runtime.c`).

```ionic
mut a = [10, 20, 30];
a.push(40);          // grows
a.len                // 4  (property-style; NO parentheses)
len(a)               // 4  (builtin form; equivalent)
a[0]                 // 10 — INDEX 0 IS THE FIRST ELEMENT
a[0] = 99;           // element assignment
```

**Verified behaviour you must internalise:**

- **Arrays are 0-indexed.** `a[0]`..`a[len-1]`.
- **Indexing is bounds-checked at runtime.** `a[len(a)]` aborts with
  `ionic: index out of bounds: index N, length M`.
- `.len` is a *property*; `len(a)` is a *builtin*. Both work.

```ionic
mut a = [10, 20, 30];
println(str_concat("len. = ", int64_to_str(a.len)));   // 3
println(str_concat("len() = ", int64_to_str(len(a)))); // 3
println(str_concat("a[0] = ", int64_to_str(a[0])));    // 10
println(str_concat("a[2] = ", int64_to_str(a[2])));    // 30
```

Reference: `examples/mastery/02_arrays.ionic`.

### 3.1 The `lib/std/data.ionic` off-by-one (a real bug)

Several helpers in `lib/std/data.ionic` assume **1-based** arrays:

```ionic
fn mean_f([float64] a) -> float64 {
    let n = len(a) - 1;     // they treat len as "last index"
    ...
    mut i = 1;              // and start at 1, skipping index 0
    while (i < n + 1) { total = total + a[i]; i = i + 1; }
    return total / int64_to_float64(n);
}
```

Given `[1.0, 2.0, 3.0, 4.0]`, `mean_f` returns **3.0** (mean of 2,3,4), not 2.5.
It silently drops the first element. `min_f`, `max_f`, `stdev_f`, and
`to_csv_line` share the assumption. Reproduce it:

```sh
./ionic_new examples/mastery/02_data_bug.ionic -o /tmp/db && /tmp/db
```

`mean_f([1,2,3,4])` prints `3` while a correct mean is `2.5`. This is exercise
09.2's first task — decide whether the fix is the library or the array
convention.

### 3.2 Printing a string that is not a literal (was broken, now fixed)

`println` is overloaded by argument type, and the choice of overload used to be
made by pattern-matching the *shape* of the argument node: "a string literal is
a string; a call to one of sixteen named builtins is a string; a method call is
a string". Anything else fell through to `ionic_println_int64`:

```ionic
mut names = ["alice", "bob"];
println(names[0]);              // used to print a POINTER as an integer
fn show(string s) { println(s); }
show("alice");                  // used to print a POINTER as an integer
```

Both cases are fixed. The overload is now chosen by asking the type oracle
(`ncg_ty_of` in `src/codegen/native.ionic`) for the argument's static type,
which resolves an element load through the array's element type and a
parameter through its declared type. `examples/mastery/02_string_array_print.ionic`
is the regression reference.

This is the same "element loads carry no type" gap recorded in
`docs/memory-model.md §10`, and closing it was a prerequisite for the RC slice —
but it does *not* by itself make refcounting safe: `ncg_ty_of` still answers
"other/unknown" for a nested-array element and for a method call, so a release
must not be emitted on those. See units 06–07.

---

## 4. Structs and methods

```ionic
struct Point { x: float64, y: float64 }

let p = Point { x: 1.0, y: 2.0 };
p.y = 3.0;                 // field assignment — allowed even though p is `let`
```

**All struct fields are mutable regardless of the binding.** Structs are
heap-allocated (`ND_STRUCT_LIT` in `src/codegen/native.ionic`).

Methods are *sugar* parsed as `ND_METHOD_CALL` and lowered at codegen to a free
function / builtin. This is why `a.push(x)` and `push(a, x)` are the same thing,
and why `a.len` (field-style) and `len(a)` both work.

Reference: `examples/mastery/02_structs.ionic`.

---

## 5. Type errors and the tensor hint

The checker reports a type mismatch with a conversion hint for tensors. Example:

```ionic
fn apply_model(tensor@cpu w) -> tensor@cpu { return w; }
let g = some_tensor.toGpu();   // tensor@gpu
apply_model(g);                // ERROR: expected tensor@cpu, got tensor@gpu
```

Diagnostic (see `tensor_convert_hint` in `src/semantic/checker.ionic`):

```
[error] type error in arg 1 of 'apply_model': expected tensor@cpu, got tensor@gpu
        (call .toCpu() to convert)
```

---

## Exercises

**02.1 — Inference trace.** For `mut x = 42 + 1; let y = 3.14 * x;`, write the
inferred type of `x` and `y` and the op the checker rewrote. Verify with output.

**02.2 — Array semantics.** Build `[10,20,30]`, print `.len`, `len()`, `a[0]`,
`a[2]`. Then trigger the bounds check and record the exact runtime message.
Reference: `examples/mastery/02_arrays.ionic`.

**02.3 — Reproduce the data bug.** Compile `examples/mastery/02_data_bug.ionic`
and show that `mean_f` returns `3` for `[1,2,3,4]`. Predict what `max_f`
returns and verify.

**02.4 — Struct mutation.** Create a struct under a `let` binding and mutate a
field. Explain why this is legal.

**02.5 — Where string-ness was being lost.** Run
`examples/mastery/02_string_array_print.ionic`. It prints a string reached
through an array element, through a function parameter, and as a user
function's return value. Then find `ncg_ty_of` in `src/codegen/native.ionic`
and write down, for each of those three, which branch of the oracle answers the
question. Reference: `examples/mastery/02_string_array_print.ionic`.

---

## Checkpoint 02

- [ ] You can describe propagation-based inference in three sentences.
- [ ] You know arrays are 0-indexed *and* that `data.ionic` assumes 1-based.
- [ ] You can explain the difference between `.len` and `len()`.
- [ ] You can name the two places string-ness used to be lost at codegen (array
      element loads, and parameters) and say what fixed them.
## [Next](03-advanced-language.md)
