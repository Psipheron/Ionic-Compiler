# Ionic

A statically-typed, self-hosting compiled language targeting native ARM64 (macOS). Ionic compiles directly to Mach-O object files — no LLVM required at runtime — and enforces hardware placement at the type level: `tensor@cpu` and `tensor@gpu` are distinct types and the compiler rejects code that crosses the boundary without an explicit transfer.

![~](assets/ionic_demo.gif)
```ionic
fn fibonacci(int64 n) -> int64 {
    if (n <= 1) { return n; }
    mut a = 0; mut b = 1; mut i = 2;
    while (i <= n) { let c = a + b; a = b; b = c; i = i + 1; }
    return b;
}

mut i = 0;
while (i <= 10) {
    println(format("fib({}) = {}", int64_to_str(i), int64_to_str(fibonacci(i))));
    i = i + 1;
}
```

```
$ ./ionic_new fib.ionic -o fib && ./fib
fib(0) = 0
fib(1) = 1
fib(2) = 1
fib(3) = 2
...
fib(10) = 55
```

---

## Features

- **Self-hosting** — `ionic_new` is compiled by itself; the entire compiler is written in Ionic
- **Direct ARM64 code generation** — emits Mach-O `.o` files directly, linked with `ld`; no LLVM at runtime
- **Static types, inferred where obvious** — `let x = 42` is `int64`; `let f = 3.14` is `float64`
- **Full float64 support** — arithmetic, `sqrt`, `pow`, `floor`, `ceil`, `fabs`, `int64_to_float64`, `float64_to_str`
- **Rich string builtins** — `format`, `str_concat`, `str_len`, `str_slice`, `str_replace`, `str_contains`, `str_starts_with`, `str_ends_with`, `int64_to_str`
- **Arrays** — `[int64]`/`[float64]`/`[string]` types, `.push`, `.len`, `arr_reset`, indexing and element assignment; element types are tracked through the checker
- **Import system** — `import std.math.*;` and selective `import std.str.{contains, trim};` pull in standard-library modules transitively with dedup; string-literal imports (`import "lexer/tokens.ionic";`) resolve relative to the importing file, and the compiler builds itself from a single entry point (`src/main.ionic`)
- **Standard library** — `std.math`, `std.str`, `std.array`, `std.io`, `std.data`, `std.text`, `std.http` live under `lib/std/`
- **Web request primitives** — `http_get`, `http_post`, `http_status`, `http_body`, `http_urlencode` (HTTP/1.1 client via raw sockets, no TLS) plus a `std.http` wrapper module
- **Hardware-aware types & placement tags** — `tensor@cpu` and `tensor@gpu` are distinct types; `@gpu fn` / `@cpu fn` pin code to a device, and the checker enforces the boundary (no accidental cross-device ops, no CPU-only I/O inside GPU code)
- **Real ML backends** — GGUF models via llama.cpp with Metal GPU; ONNX/CoreML; Piper TTS
- **Human-readable errors** — multi-error reporting, source-line carets, column tracking, panic-mode recovery, enclosing-function context, "did you mean…" suggestions, clickable `path:line:col` jump links, ANSI colors that auto-disable when not a TTY
- **Optimizer** — a self-hosted optimization pipeline (fold, inline, unroll, const-propagate) that runs on every compilation before codegen
- **Modern syntax** — compound assignment (`+=`, `-=`, `*=`, `/=`), block comments `/* ... */`, and panic-mode lexer recovery

---

## Build

Two compilers live in the tree:

- **`ionic_new`** — the self-hosted compiler. Built from the split
  `.ionic` sources in `src/`. This is what we ship in releases and what
  end users run.
- **`target/release/ionic`** — the Rust/LLVM bootstrap (cross-OS). Built
  from `src/*.rs` via `cargo build --release`. It can compile *any*
  `.ionic` program on any OS with Rust + clang installed — including the
  self-hosted sources themselves (`src/main.ionic`), because it implements
  the full operator set (bitwise `& | ^ ~ << >>`), hex literals, and the
  relative-import splicing that assembles the split `src/` tree. A fresh
  clone with Rust + clang needs no committed `ionic_self` to bootstrap.

A prebuilt **`ionic_self`** is checked in at the repo root — it's the
binary version of `ionic_new` plus any updates since the last release.
Most users (and CI) only need that.

### Install from a release

Released tarballs (`ionic-<version>-aarch64-apple-darwin.tar.gz`) are
**self-contained**: the `ionic` binary carries the compiler, and the
tarball ships the standard library and the C runtime source alongside it:

```
ionic-v0.2.1-aarch64-apple-darwin/
├── ionic                       # the compiler
├── lib/std/*.ionic             # bundled standard library
└── src/ionic_model_runtime.c   # C runtime, linked into every program you build
```

At startup `ionic` publishes its own directory as `$IONIC_ROOT` (via the C
runtime), so it finds `lib/` and `src/ionic_model_runtime.c` **relative to
the executable** — you can run it from any working directory. Homebrew
installs it the same way (`brew install ByteTheBait/ionic/ionic`).

Because the runtime is compiled with `clang` at link time, building a
program still needs a C compiler on `PATH`; nothing else is required.

### Prerequisites

- Clang / `ld` (for linking the compiled output)
- For development: a working `ionic_self` (one ships in the repo) OR
  Rust toolchain + `cargo`

### Quick start (using the prebuilt `ionic_self`)

```sh
git clone <repo>
cd AILANG
./build.sh                                  # ~3 s; produces ./ionic_new
./ionic_new hello.ionic -o hello && ./hello
```

### Rebuilding after source changes

```sh
./build.sh                                  # ~3 s; uses ./ionic_self to rebuild ./ionic_new
```

### Cross-platform, source-only bootstrap

When you don't have an `ionic_self` but do have Rust + clang:

```sh
cargo build --release                       # → target/release/ionic (Rust LLVM compiler)
./target/release/ionic src/main.ionic -o ionic_new   # → self-hosted compiler, no ionic_self needed
```

`target/release/ionic` is a complete Rust/LLVM bootstrap. It speaks the
full language the self-hosted compiler uses — bitwise operators
(`& | ^ ~ << >>`, with the same precedence as the self-hosted parser),
hex literals, and the relative `import "path.ionic";` splicing that
assembles the split `src/` tree — so it can compile **the compiler itself**
on any OS with Rust + clang (Linux ARM64, macOS, Windows). This is what
makes the tree bootstrap-capable from source alone: a fresh clone needs
only `cargo build` plus the `.ionic` sources.

### Verifying the self-hosted compiler

The compiler is reproducible at the object level: compiling the same
sources twice yields byte-identical `.o` files. The final linked
executable, however, always differs by a few dozen bytes because `ld`
stamps a random `LC_UUID` (and an ad-hoc code signature over it) into
every Mach-O — so a naive `cmp` of two *linked binaries* can never match.

`build.sh --verify-self-hosting` handles this for you: it builds
`ionic_new`, then has the freshly-built compiler recompile the same
sources and compares the two **object files** (not the linked binaries).

```sh
./build.sh --verify-self-hosting   # ✓ object files byte-identical
```

If you prefer to do it by hand, compare the compiler's emitted object
output rather than the executable. Each compile writes its object next to
the output as `<out>.o`:

```sh
./build.sh                          # ionic_self → ./ionic_new
./ionic_self src/main.ionic -o /tmp/a   # also writes /tmp/a.o
./ionic_new  src/main.ionic -o /tmp/b   # also writes /tmp/b.o
cmp -s /tmp/a.o /tmp/b.o && echo "reproducible" || echo "DIFFERS"
```

Because the object is keyed on the output name, the compiler is safe to
run concurrently — two `ionic` processes building different outputs use
different objects and cannot cross-link each other's code.

The build step on its own is enough to produce a working `./ionic_new`
tarball; releases ship that binary, and end users use it to compile their
own programs.

---

## Usage

```
./ionic_new <file.ionic> -o <output>
```

**Compile and run:**
```sh
./ionic_new hello.ionic -o hello && ./hello
```

**No `fn main` needed** — top-level statements run directly:
```ionic
let x = 6 * 7;
println(int64_to_str(x));   // prints 42
```

---

## Diagnostics

Errors are structured, multi-error (every problem in a file is reported in
one pass), and colorized when stdout is a terminal. A typical error in a
TTY:

```
    at /Users/henry/Developer/AILANG/examples/foo.ionic:5:12
[error] 5:12: unknown variable 'x': not declared in this scope
    in fn compute(int64 n) -> int64 {
        return x + 1;
               ^
    help: did you mean 'xs' (similar variable)?
1 error(s) found.
    hint: re-run `/Users/henry/Developer/AILANG/examples/foo.ionic` in your editor (hx) -- click path:line:col above to jump.
```

In a plain terminal / CI log (`NO_COLOR=1`, pipe, `TERM=dumb`, …) the escape
codes are stripped and the path is rendered as plain `at path:line:col`:

```
    at examples/foo.ionic:5:12
[error] 5:12: unknown variable 'x': not declared in this scope
    in fn compute(int64 n) -> int64 {
        return x + 1;
               ^
    help: did you mean 'xs' (similar variable)?
```

Five features make this work well:

- **Multi-error reporting** — the parser continues after the first error
  via `p_sync_stmt` / `p_sync_top` (advance to next `;` or `fn`/`let`/`struct`
  boundary), so 10 syntax errors in one file produce 10 diagnostics, not 1.
- **"Did you mean…" suggestions** — unknown names are matched against the
  registered function / variable / struct pools via Levenshtein distance
  (threshold: edits ≤ 2 for len ≥ 3, else ≤ ceil(len/3)). The closest
  candidate becomes a `help:` hint.
- **Enclosing-function context** — every error in a function body shows
  its `fn header` line in dim gray above the source line, so the caret has
  a frame of reference.
- **Clickable `path:line:col` jump links** — emitted above each `[error]`
  line. VSCode / iTerm2 / Neovim auto-detect these and make them Cmd-clickable;
  in colored TTYs the path is underlined + cyan.
- **`$EDITOR` / `$VISUAL` hint** — if either is set, the compiler prints a
  final hint line naming the editor and reminding you that the
  `path:line:col` line above is what your editor jumps on.

Color autodetection honors [no-color.org](https://no-color.org/): set
`NO_COLOR=1` (or `TERM=dumb`) to disable escape codes for CI logs.

---

## Language quick reference

### Variables

```ionic
let x = 42;           // immutable int64
mut count = 0;        // mutable int64
let pi = 3.14159;     // float64
let msg = "hello";    // string
```

### Functions

```ionic
fn gcd(int64 a, int64 b) -> int64 {
    mut aa = a; mut bb = b;
    while (bb != 0) { let t = bb; bb = aa - (aa / bb) * bb; aa = t; }
    return aa;
}
```

### Control flow

```ionic
if (x > 0) { println("positive"); }
else { println("non-positive"); }

while (i < 10) { i = i + 1; }
```

### Arrays

```ionic
mut data = [0]; arr_reset(data);
data.push(10); data.push(20); data.push(30);
println(int64_to_str(data.len));     // 3
println(int64_to_str(data[1]));      // 20
data[1] = 99;
```

### String formatting

```ionic
let s = format("x={}, y={}", int64_to_str(x), float64_to_str(y));
println(s);
```

### Float math

```ionic
let pi = 3.14159265358979;
println(float64_to_str(sqrt(2.0)));          // 1.41421
println(float64_to_str(pow(pi, 2.0)));        // 9.8696
println(float64_to_str(int64_to_float64(n))); // cast int→float
```

### Compound assignment

```ionic
mut x = 10;
x += 5;    // x = 15
x -= 3;    // 12
x *= 2;    // 24
x /= 4;    // 6
```

### Block comments

```ionic
/* line one
   line two */
let y = 1 + 2;   // 3
```

### Hardware placement

Tensors carry their device in the type: `tensor@cpu` and `tensor@gpu` are
distinct, and crossing between them requires an explicit transfer.

```ionic
let weights: tensor@gpu = load_model("model.onnx").forward(inputs);
let inputs:  tensor@cpu = load_batch(path);

let inputs_gpu = inputs.toGpu();   // explicit CPU → GPU transfer
```

`gpu { ... }` is a GPU-context block. Inside it the checker only allows
`tensor@gpu` values and tensor compute — using a `tensor@cpu` or calling CPU
I/O (`print`, `file_read`, `file_write`, …) is a compile error:

```ionic
gpu {
    let prediction = forward(inputs_gpu, weights);   // ok
    // print("hi");                                  // error: CPU-only I/O
}
```

Functions can be pinned to a device with a leading tag. An `@gpu fn` body is
checked as a GPU context (same rules as a `gpu { }` block); `@cpu fn` bodies
run on the CPU. A function tag and a contradicting parameter tag
(`@gpu fn f(@cpu tensor@cpu x)`) is rejected, as is calling an `@cpu fn` from
GPU code:

```ionic
@cpu fn load_batch(string path) -> tensor@cpu { ... }   // runs on the CPU
@gpu fn train_step(tensor@gpu x, tensor@gpu w) -> tensor@gpu {
    return forward(x, w);   // no print / CPU-only calls allowed here
}
@gpu(0.5) fn half(...) -> ... { ... }                    // use 50% of the GPU
```

### Imports & the standard library

Import a whole module with a wildcard, or select specific symbols:

```ionic
import std.math.*;                      // everything in lib/std/math.ionic
import std.str.{contains, trim};        // just those two functions (parsed but not yet filtered)
import std.array.*;
import std.io.*;
import myutils;                          // resolves against project/user packages
```

#### Module resolution

`src/imports.ionic` resolves a dotted path (e.g. `std.math`) by trying the
following locations in order, taking the first match:

1. `./.ionic/packages/<slash-form>.ionic` — project-local packages.
2. `$HOME/.ionic/packages/<slash-form>.ionic` — user-central packages
   (falls back to `$USERPROFILE` for Windows-style installs).
3. The built-in standard library, but **only when the first segment is
   `std`**. The compiler looks first in `$IONIC_ROOT/lib/<slash-form>.ionic`
   (where `$IONIC_ROOT` is the compiler's own install directory, published by
   the runtime at startup — so an installed binary works from any CWD), then
   falls back to the repo-relative `lib/<slash-form>.ionic`. Non-`std` paths
   that miss layers 1 and 2 are reported as errors with the full list of
   paths that were tried.

The dotted form `std.data.csv` is converted to slash form `std/data/csv` for
the on-disk path, so packages are stored as a regular directory hierarchy
inside `.ionic/packages/`. Each module is loaded at most once; cycles and
duplicate imports are tolerated, and transitive imports are expanded before
the dependents that need them.

Layer 3 is intentionally restricted to the `std.` prefix so user packages can
never accidentally shadow or pretend to be a built-in module.

#### Relative (path) imports

The compiler is itself assembled from a tree of modules via string-literal
imports, which resolve against the directory of the *importing* file:

```ionic
import "imports.ionic";            // same directory
import "lexer/tokens.ionic";       // subdirectory
import "../diagnostics.ionic";     // parent directory
```

`.` and `..` segments are collapsed during resolution, so a module reached by
two different paths (a "diamond" import) is loaded exactly once. This is what
lets `build.sh` compile the whole compiler from the single entry file
`src/main.ionic` — the include order lives in the source, not the build script.

#### Layer-1 example

```
project/
  .ionic/packages/utils.ionic    // contains `fn greet(string) -> string`
  examples/hello.ionic
```

```ionic
// examples/hello.ionic
import utils;
println(greet("world"));
```

#### Layer-2 example

```
~/.ionic/packages/utils.ionic    // shared across all projects on this machine
```

```ionic
import utils;                     // same source — found via $HOME
println(greet("alice"));
```

If nothing matches, the compiler exits non-zero with a diagnostic listing the
paths it tried:

```
ionic: cannot resolve import 'nosuch.module'
  tried: ./.ionic/packages/nosuch/module.ionic
  tried: /Users/henry/.ionic/packages/nosuch/module.ionic
```

Available std modules: `std.math` (constants + float helpers), `std.str`
(string utils), `std.array` (array utils over `[int64]`/`[float64]`),
`std.io` (readline, eprint, print_hr), `std.data` (lower/upper/clean,
normalize_ws, CSV parse/split, mean/stdev/min/max over `[float64]`,
to_csv_line), `std.text` (tokenize, words, count_substr, replace_all, join),
and `std.http` (web client: get / post / status / body / urlencode over the
raw-socket HTTP/1.1 runtime).

```ionic
import std.http.*;
let raw = get("http://example.com/");
if (status(raw) == 200) {
    let html = body(raw);
    println(int64_to_str(str_len(html)));
}
```

---

## Optimizations

`src/opt/opt.ionic` is a self-hosted, single-pass optimizer that runs on every
compilation between semantic checking and codegen. It rewrites the AST in place
and always preserves the stable self-hosting fixed point — the compiler is able
to optimize its own source.

- **Constant folding + algebraic simplification + DCE** — folds integer ops on
  constant operands (including comparisons and bitwise ops), simplifies
  `x + 0`, `x * 1`, `x / 1`, `x | 0`, `x << 0`, folds unary neg/not/bitnot,
  eliminates dead `if` branches on constant conditions and truncates unreachable
  code after `return`.
- **Float constant folding** — folds `+ - * /` on float (or int-promoted)
  literals and the single-arg builtins `sqrt`, `fabs`, `floor`, `ceil`, plus
  float negation, computing IEEE-754 bit patterns at compile time so the emitted
  object contains no runtime float calls.
- **Loop unrolling** — flattens `for` loops whose bounds are compile-time
  integer constants and whose trip count is small (`≤ 8`), with no
  `break`/`continue`, by deeply cloning the body per iteration and substituting
  the induction variable with its literal index.
- **Function inlining** — inlines "leaf" pure functions (a body that is exactly
  a single `return <expr>` with no calls): parameters are substituted by the
  optimized call-site arguments and the resulting expression is re-optimized.
  Recursion and multi-statement functions are left intact.
- **Scoped constant propagation (mem2reg subset)** — locally replaces references
  to names declared `let x = <literal>` and never reassigned anywhere in the
  function with their literal values, eliminating redundant store/loads so later
  expressions can fold further.

Combined, these passes work together — e.g. a constant `let` is propagated into
arguments so a leaf call inlines, and the substituted body then folds; an
inlined small loop body inside a constant-trip `for` is unrolled. Folding is
always conservative, so optimization never changes program semantics.

---

## Project layout

```
src/
  codegen/         Code generator (ARM64 Mach-O + Linux ELF emitters) — Ionic source
  parser/          Parser — Ionic source
  lexer/           Lexer — Ionic source
  semantic/        Type checker — Ionic source
  opt/             Optimizer (fold / inline / unroll / const-prop) — Ionic source
  main.ionic       Compiler entry point
  imports.ionic    Import resolver: project-local → user-central → stdlib, transitive dedup
  diagnostics.ionic  Error reporting with ANSI colors, jump links, did-you-mean
  codegen.rs       Rust LLVM backend (cross-OS bootstrap)
  lexer.rs / parser.rs / ast.rs / semantic.rs / imports.rs / resolve.rs / main.rs  Rust frontend
  ionic_model_runtime.c   Native runtime: I/O, arrays, strings, math, ML, llama.cpp
build.sh           Build script (--verify-self-hosting for regression check)
ionic_new          Primary compiler binary (self-hosted)
ionic_self         Previous-generation compiler (used to build ionic_new)
tree-sitter-ionic/ Tree-sitter grammar + highlight queries for editors
```

---

## Self-hosting cycle

```
(cargo build --release)  ──>  target/release/ionic   (Rust LLVM bootstrap, cross-OS)
target/release/ionic     ──>  ionic_new               (source-only bootstrap, any OS)
ionic_self               ──>  ionic_new               (committed-binary bootstrap)
ionic_new                ──>  ionic_new                (self-hosted, fixed point)
```

There are two independent ways to bootstrap, and both now start from the
`.ionic` sources in `src/`:

- **Rust bootstrap** — `cargo build --release`, then compile `src/main.ionic`.
  Works on any OS with Rust + clang (Linux ARM64, macOS, Windows) and needs
  no committed binary.
- **Committed binary** — the checked-in `ionic_self` rebuilds `ionic_new`
  with no toolchain at all; this is what releases and most end users use.

Once a self-hosted `ionic_new` exists it is a fixed point: it reproduces
itself (byte-identical object files). End users who install the tarball
only ever see `ionic_new`.

---
## Contributing
PR's are always welcome.
If you want a in-depth intro to the compiler and it's intricacies, look at the [Tutorial](docs/tutorial/README.md)
