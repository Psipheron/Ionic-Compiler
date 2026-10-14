# 01 — Language Core

**Goal:** be fluent in values, types, operators, control flow, and the string
model. Everything here is checked by `examples/stdlib_test.ionic`,
`examples/showcase.ionic`, and the exercises.

---

## 1. Comments

```ionic
// line comment
/* block comment
   spanning lines (non-nesting) */
/// doc comment — attaches to the next fn/struct
```

Block comments do **not** nest (first `*/` ends them). Implemented in
`src/lexer/lexer.ionic` (`lx_skip_ws`). `///` doc comments are attached to the
following `fn`/`struct` and surface in `--dump-ast` output on the legacy path.

---

## 2. Literals and their types

| Literal | Type |
|---------|------|
| `42`, `0x2A` | `int64` |
| `3.14` | `float64` |
| `"hi"` | `string` |
| `true` / `false` | `bool` |
| `[1, 2, 3]` | `[int64]` |
| `[1.0, 2.0]` | `[float64]` |

Two non-obvious facts:

- **`bool` and `int64` are the same at runtime.** The checker's `ty_compat`
  treats them as compatible. `[1, true]` is accepted and infers `[int64]`.
- An array literal's element type is set by the **first concrete element**; a
  later incompatible element is an error (`mixed element types`).

---

## 3. Variables: `let` vs `mut`

```ionic
let x = 42;              // immutable — cannot be reassigned
mut count = 0;           // mutable
let pi: float64 = 3.14159;  // optional annotation
```

Reassigning a `let` is a compile error (`examples/mastery/fail/immutable.ionic`):

```
[error] 8:7: assignment to immutable 'x': declare with mut to allow reassignment
```

Source: `src/semantic/checker.ionic` (assignment path, ~line 1076).

---

## 4. Operators and precedence

From tightest to loosest (`src/parser/parser.ionic`; mirrored in the homework doc):

- unary `- ! ~`
- `* / %`
- `+ -`
- `<< >>`
- `< > <= >=`
- `== !=`
- `&`
- `^`
- `|`
- `&&`
- `||`

Integer division truncates; use `x - (x / y) * y` for a remainder if you don't
want `%` (both work — `%` is supported, see `OP_MOD` in
`src/codegen/native.ionic`).

**Mixed int/float arithmetic auto-widens.** `3.14 * x` where `x : int64` is
legal; the checker rewrites the op to its float form (`OP_MUL` → `OP_FMUL`) and
codegen inserts the conversion. See unit 06 for the AST mutation.

---

## 5. Control flow

```ionic
if (x > 0) { ... } else if (x == 0) { ... } else { ... }

while (i < 10) { i = i + 1; }

for (i in 0..10) { ... }     // EXCLUSIVE upper bound: 0..9

break;
continue;
```

`for (i in a..b)` iterates `a` through `b-1`. This is the single most common
off-by-one source — the checker/codegen implement it directly (no desugaring to
`while` in source).

---

## 6. Strings and the `format` builtin

Strings are immutable UTF-8 values, allocated by the C runtime (`char *`).

```ionic
let s = "hello world";
str_len(s);              // 11
str_concat("a", s);      // "ahello world"
str_eq(s, "hello world");// true  (NOT ==)
str_index(s, 0);         // 104  (ASCII of 'h')
char_to_str(65);         // "A"
int64_to_str(42);        // "42"
float64_to_str(3.14);    // "3.140000"
```

`format("{} and {}", "A", int64_to_str(42))` substitutes `{}` placeholders in
order. Note the arguments must already be strings — `format` does **not** coerce
`int64` itself, so wrap with `int64_to_str`. Reference:
`examples/mastery/01_strings.ionic`.

Escapes in string literals: `\n \t \r \" \\ \e` (ESC, 0x1B) and `\xHH` (raw
byte). `char_to_str` takes a Unicode codepoint and UTF-8 encodes it, so
`char_to_str(0x2500)` is a box-drawing glyph.

---

## 7. Output: `print` / `println`

Both are **overloaded by the inferred argument type** at codegen:

- string → `ionic_print` / `ionic_println`
- int64 → `ionic_print_int64` / `ionic_println_int64`
- float64 → `ionic_print_float64` / `ionic_println_float64` (prints with `%g`)

`print` has no trailing newline; `println` does. Reference:
`examples/mastery/01_print_variants.ionic`.

---

## 8. Input

```ionic
let line = read_line();   // builtin, returns one line (strips trailing newline)
```

`import std.io.*` adds `readline()` (same idea, buffer of 4096) and
`readline_n(max)`. See unit 05.

---

## Exercises

**01.1 — Literals.** Print one value of each literal type. Note which print
calls need a `*_to_str` wrapper and which print directly.
Reference: `examples/mastery/01_literals.ionic`.

**01.2 — Exclusive range.** Print `0..5`. Then predict and verify the output of
a loop that sums `0..n`. Confirm the bound is exclusive.
Reference: `examples/mastery/01_ranges.ionic`.

**01.3 — Widening.** Compute `3.14 * 10` and print it. Explain why it works and
which op constant the checker rewrote.

**01.4 — Operator precedence.** Write one expression that only gives the
"expected" answer if `*` binds tighter than `+`, and one that depends on `<<`
binding looser than `+`. Verify both.

**01.5 — Immutability error.** Compile `examples/mastery/fail/immutable.ionic`
and record the exact diagnostic.

---

## Checkpoint 01

- [ ] You can name the type of any literal without compiling it.
- [ ] You can state `for`'s bound behaviour and the `bool ≡ int64` rule.
- [ ] You can explain why `if (name == "ionic")` is a type error and `str_eq` isn't.
- [ ] You can produce and recognise the immutable-assignment error.

## [Next](02-types-functions-arrays-structs.md)
