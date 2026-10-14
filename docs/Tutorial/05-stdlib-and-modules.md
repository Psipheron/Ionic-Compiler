# 05 — Standard Library & Modules

**Goal:** use the standard library and the import system fluently, and understand
*why* wildcard imports cost nothing.

---

## 1. The std modules

Bundled under `lib/std/`:

- `std.math` — `clamp`, `lerp`, `sign`, `min_f`, `max_f`, `degrees`, `radians`,
  `pow_i`, and constants `PI`, `TAU`, `E`.
- `std.str` — `contains`, `starts_with`, `ends_with`, `repeat`, `count_char`,
  `pad_left`, `pad_right`, `trim_left`, `trim_right`, `trim`, `str_reverse`.
- `std.array` — `arr_sum`, `arr_sum_f`, `arr_mean`, `arr_max`, `arr_min`,
  `arr_contains`, `arr_fill`, `arr_reverse`, `arr_index_of`, `arr_copy`.
- `std.io` — `readline`, `readline_n`, `eprint`, `eprintln`, `print_hr`.
- `std.data` — `lower`, `upper`, `clean`, `normalize_ws`, `split_char`,
  `split_csv`, `parse_csv`, `mean_f`, `stdev_f`, `min_f`, `max_f`, `to_csv_line`.
  ⚠️ the `*_f` stats and `to_csv_line` assume 1-based arrays — see unit 02 §3.1.
- `std.text` — `tokenize`, `words`, `count_substr`, `replace_all`, `join`.
- `std.http` — `get`, `post`, `status`, `body`, `urlencode` (raw-socket HTTP/1.1).

Full API: `docs/stdlib.md`. Reference program exercising several:
`examples/stdlib_test.ionic`.

---

## 2. Import forms

```ionic
import std.math.*;                     // wildcard — everything, selective at codegen
import std.str.{contains, trim};       // named — exactly these
import std.io;                         // module — same as wildcard today
import utils;                          // a package (no std. prefix)
import "lexer/tokens.ionic";           // relative path import
```

---

## 3. Selective compilation — the zero-overhead trick

A wildcard import is **not** a blob include. When you write
`import std.math.*;` and only call `clamp`, the compiler:

1. parses all of `lib/std/math.ionic` into a pool;
2. walks your program's AST to see which library symbols are referenced;
3. follows transitive references (BFS) from those symbols only;
4. emits IR **only** for the reachable set.

So `import std.math.*` when you use only `clamp` compiles identically to
`import std.math.{clamp}`. Dead library functions add zero bytes. See
`docs/stdlib.md` and `src/imports.ionic`.

Reference: `examples/stdlib_test.ionic` documents, in its header comment, exactly
which symbols are expected to survive.

---

## 4. Module resolution order

A dotted path like `std.math` is tried in order (first match wins):

1. `./.ionic/packages/<slash-form>.ionic` — project-local.
2. `$HOME/.ionic/packages/<slash-form>.ionic` — user-central (falls back to
   `$USERPROFILE` on Windows-style installs).
3. The bundled stdlib — **only when the first segment is `std`**: first
   `$IONIC_ROOT/lib/<slash-form>.ionic`, then repo-relative `lib/<slash-form>.ionic`.

The last layer is deliberately restricted to `std.` so user packages can never
shadow or impersonate a built-in module.

If nothing matches, the compiler exits non-zero listing every path it tried.

Source: `src/imports.ionic`.

---

## 5. Relative (path) imports — how the compiler builds itself

```ionic
import "imports.ionic";          // same dir as this file
import "lexer/tokens.ionic";     // subdirectory
import "../diagnostics.ionic";   // parent
```

These resolve against the **importing file's** directory. `.`/`..` are collapsed,
so a module reached by two paths is loaded **once** (diamond imports dedupe).
This is the mechanism behind the single-entry build: `build.sh` compiles only
`src/main.ionic`, and the whole `src/` tree is spliced in dependency order by the
`import "…"` statements themselves. See unit 08.

---

## 6. Writing your own module

Put a file at `.ionic/packages/utils.ionic` (project-local) or
`~/.ionic/packages/utils.ionic` (shared):

```ionic
// .ionic/packages/utils.ionic
fn greet(string who) -> string { return str_concat("Hello, ", who); }
```

```ionic
// your program
import utils;
print(greet("world"));
```

Reference: `examples/mastery/05_package_demo/` (a project-local package plus a
consumer).

---

## Exercises

**05.1 — Selective import.** Write a program that imports `std.math.*` but calls
only `clamp`. Then compile with `import std.math.{clamp}` and diff the two `.o`
files. They should be identical. Reference: `examples/mastery/05_selective.ionic`.

**05.2 — Named vs wildcard.** Call three functions from `std.str` via a named
import; confirm the others are absent (they won't link, since they're never
emitted).

**05.3 — Relative import.** Create `examples/mastery/05_rel/{main.ionic,helper.ionic}`
where `main.ionic` does `import "helper.ionic";` and calls a function defined
there. Compile `main.ionic` directly (no listing of `helper.ionic`).

**05.4 — Project package.** Reproduce the `utils` package example above under
`.ionic/packages/` and import it by name. Reference:
`examples/mastery/05_package_demo/`.

**05.5 — Resolution miss.** Import `std.nosuch` and capture the exact error,
including the list of tried paths.

---

## Checkpoint 05

- [ ] You can explain why `.*` is not an include-blob.
- [ ] You can list the three resolution layers and the `std.`-only restriction.
- [ ] You can build a program from a relative import without listing files.
- [ ] You can create and consume a project-local package.
- [ ] You can reproduce the resolution-failure message.

## [Next](06-compiler-architecture.md)
