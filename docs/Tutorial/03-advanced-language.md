# 03 — Advanced Language

**Goal:** the language surface beyond the basics — compound assignment, block
comments, literal edge cases, CLI arguments, file I/O, and a catalogue of the
pitfalls that bite real Ionic code.

---

## 1. Compound assignment

`+= -= *= /=` are real two-character tokens (`TOK_PLUS_EQ` … in
`src/lexer/tokens.ionic`), recognised by lookahead in the lexer, then
**desugared during parsing**: `x += e` becomes `x = x + e`. No new AST node.

```ionic
mut x = 10;
x += 5;   // 15
x -= 3;   // 12
x *= 2;   // 24
x /= 4;   // 6
```

Source: `src/parser/parser.ionic` (`is_compound_assign`, `compound_assign_op`).

---

## 2. Block comments

`/* … */`, **non-nesting**: the first `*/` terminates. Handled in `lx_skip_ws`
before token dispatch, so `"/"` inside a string is never mistaken for a comment
and a lone `/` is still division. `src/lexer/lexer.ionic`.

---

## 3. Escapes and literal edges

- String escapes: `\n \t \r \" \\ \e` (ESC) `\xHH` (raw byte).
- Hexadecimal integer literals: `0x2A` == `42`.
- Float literals: `3.14`. Floats are carried as raw IEEE-754 bits in an integer
  slot; `float64_to_str` and `println` decode them. `str_to_float64_bits` exists
  if you need the raw pattern.

```ionic
print("\e[31mred\e[0m\n");   // ANSI colour
print("\x41\x42\n");          // "AB"
print(char_to_str(0x2500));   // a box-drawing bar
```

Reference: `examples/mastery/03_escapes.ionic`.

---

## 4. Command-line arguments and exit

```ionic
fn main() -> int64 {
    let a0 = get_arg(0);     // FIRST ARG AFTER THE PROGRAM NAME (0-indexed)
    print(a0);
    exit(0);                 // set an explicit status (return value is ignored)
}
```

`get_arg` returns `""` when out of range, which is how `src/main.ionic` loops
over arguments. `exit(code)` (builtin, `-> void`) terminates immediately and *is*
how you set a non-zero status — `main`'s own return value is discarded by the
wrapper (`src/codegen/native.ionic`, `ncg_main_wrapper`). `getenv("VAR")` reads
the environment.

Reference: `examples/mastery/03_args.ionic`.

---

## 5. File I/O

```ionic
let content = file_read("data.txt");       // whole file -> string ("" if missing)
file_write("out.txt", content);            // overwrite/create
```

> ⚠️ **`file_exists` is advertised in `docs/language.md`/`guide.md` but is *not*
> registered in the shipped self-hosted compiler** (`register_builtins()` in
> `src/semantic/checker.ionic` omits it; `native.ionic` doesn't lower it). Using
> it yields `unknown function 'file_exists'`. Probe with
> `str_len(file_read(path)) > 0` instead. See the hub's known discrepancies.
> Also `file_write_binary(path, bytes)` exists.

> The runtime leaks these buffers by design today (`docs/memory-model.md §1`). A
> long-running program that reads files or builds strings grows without bound —
> that's the gap unit 07 is about.

Reference: `examples/mastery/03_file_io.ionic` (writes then reads a temp file).

---

## 6. Pitfalls catalogue

1. **Mutability.** `let` cannot be reassigned. Use `mut`.
2. **Array type syntax.** Element type goes **inside** brackets: `fn sum([int64] a)`
   is correct; `int64[] a` does not parse.
3. **String comparison.** Use `str_eq(a, b)`, never `==`.
4. **`model` is a keyword.** `let model = load_model(...)` is a parse error; use
   `let mdl = ...`.
5. **`for` is exclusive.** `0..n` visits `0`..`n-1`.
6. **`.len` has no parentheses.** `a.len` is a property; `a.len()` is wrong.
7. **`arr[i]` of a `[string]` printed directly is garbage** (unit 02 §3.2).
8. **`format` does not coerce.** Wrap numbers with `int64_to_str` /
   `float64_to_str` before passing them.
9. **`data.ionic` array helpers assume 1-based arrays** (unit 02 §3.1).

---

## 7. Diagnostics you'll live with

Errors are multi-error, colorized on a TTY, and use panic-mode recovery so one
bad statement doesn't hide the rest (`src/diagnostics.ionic`,
`p_sync_stmt`/`p_sync_top` in `src/parser/parser.ionic`). An unknown name gets a
"did you mean…" hint via Levenshtein distance.

```
[error] 5:12: unknown variable 'x': not declared in this scope
    in fn compute(int64 n) -> int64 {
        return x + 1;
               ^
    help: did you mean 'xs' (similar variable)?
```

Set `NO_COLOR=1` (or `TERM=dumb`) for clean CI logs. Reference:
`examples/mastery/fail/multi_error.ionic` (several errors on purpose; run it
through the compiler and count the diagnostics).

---

## Exercises

**03.1 — Compound assignment.** Rewrite a small arithmetic sequence using
`+= -= *= /=` and show the values match the long form.

**03.2 — Escapes.** Print an ANSI-coloured word, the bytes `AB`, and one
box-drawing glyph. Reference: `examples/mastery/03_escapes.ionic`.

**03.3 — Arguments.** Write a program that prints its first argument and returns
a non-zero exit code when no argument is given. Run it both ways.

**03.4 — Round-trip file I/O.** Write a string to a temp file, read it back, and
print `str_eq` of the two. Reference: `examples/mastery/03_file_io.ionic`.

**03.5 — Pitfall gauntlet.** For each of pitfalls 1–9, construct the failing (or
misbehaving) program and record the compiler/runtime response. Some are in
`examples/mastery/fail/`.

---

## Checkpoint 03

- [ ] You can explain how `x += e` reaches codegen without a new AST node.
- [ ] You can retrieve and validate a CLI argument and set the exit code.
- [ ] You can round-trip a file through `file_write`/`file_read`.
- [ ] You have personally triggered at least six of the nine pitfalls.

## [Next](04-hardware-placement.md)
