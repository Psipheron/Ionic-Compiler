# 08 — Build, Bootstrap & Self-Hosting

**Goal:** master the bootstrap graph, the self-hosting fixed point, the object-key
concurrency rule, packaging, and CI. These are the invariants you must not break.

---

## 1. The two compilers in the tree

- **`ionic_new`** — the self-hosted compiler, built from the split `.ionic`
  sources in `src/` via the single entry point `src/main.ionic`. This is what
  ships and what users run.
- **`target/release/ionic`** — a Rust/LLVM **bootstrap** compiler (`cargo build
  --release`). It speaks the same language (bitwise ops, hex, relative-import
  splicing) so it can compile `src/main.ionic` on any OS with Rust + clang,
  including Linux/Windows. It is not shipped to end users.
- **`ionic_self`** — the committed previous-generation **binary**, used to
  rebuild `ionic_new` with no toolchain at all.

---

## 2. The bootstrap graph

```
cargo build --release   ──►  target/release/ionic   (Rust LLVM bootstrap, cross-OS)
target/release/ionic    ──►  ionic_new              (source-only bootstrap, any OS)
ionic_self              ──►  ionic_new              (committed-binary bootstrap)
ionic_new               ──►  ionic_new              (self-hosted fixed point)
```

There are **two independent ways** to bootstrap, both starting from the `.ionic`
sources:

- **Rust bootstrap** — `cargo build --release`, then compile `src/main.ionic`.
- **Committed binary** — `ionic_self` rebuilds `ionic_new`; no toolchain needed.

---

## 3. The self-hosting fixed point (the central invariant)

The compiler is **reproducible at the object level**: compiling the same sources
twice yields **byte-identical `.o` files**. `ionic_new` must reproduce itself.

The *linked executable* never matches byte-for-byte, because `ld` stamps a random
`LC_UUID` (and an ad-hoc code signature over it) into every Mach-O. So the
verification compares **objects**, not binaries:

```sh
./build.sh --verify-self-hosting
# ionic_self compiles src/main.ionic -> A.o
# the freshly built ionic_new compiles src/main.ionic -> B.o
# cmp A.o B.o  -> must be identical
```

The object is written next to the output as `<out>.o`. Read `build.sh` fully.

**Why keying the object on the output name matters:** a fixed `/tmp` path is not
concurrency-safe — two `ionic` processes would write the same object and each
link a *mixture* of the two. Keying on `-o` makes parallel builds isolated. Don't
"improve" this.

---

## 4. The single-entry build

`build.sh` compiles **only** `src/main.ionic`. The rest of the tree is spliced in
by the `import "…"` statements in the source (unit 05 §5), in dependency order.
The include order lives in the code, not the script. This is why the compiler's
module graph *is* its build graph.

---

## 5. Packaging & `$IONIC_ROOT`

A released tarball is self-contained:

```
ionic-vX.Y.Z-aarch64-apple-darwin/
├── ionic                       # the compiler
├── lib/std/*.ionic             # bundled stdlib
└── src/ionic_model_runtime.c   # C runtime, linked into every program
```

At startup `ionic` publishes its own directory as `$IONIC_ROOT` (via the C
runtime), so it finds `lib/` and `src/ionic_model_runtime.c` **relative to the
executable** — you can run it from any CWD. `src/main.ionic` reads
`getenv("IONIC_ROOT")` and falls back to repo-relative paths for a source
checkout. See the packaging step in `.github/workflows/ci.yml`.

---

## 6. CI is the spec

Read `.github/workflows/ci.yml`. It encodes the contract:

- Rust bootstrap: `cargo build --release`, smoke tests, then **compile
  `src/main.ionic`** and run the result (closed loop).
- Self-hosted: `./build.sh`, then `./build.sh --verify-self-hosting`, then smoke
  tests.
- Packaging: extract the tarball and compile a program importing `std.*` **from
  `/tmp`** so the repo-relative fallback can't mask a broken `$IONIC_ROOT`.
- Hardware tags: `examples/hw_tags.ionic` must compile and run; `examples/hw_tag_error.ionic`
  must be **rejected** with a `CPU-only I/O` diagnostic.
- Shadowing regression: a top-level `let n` must not defeat a parameter `n` in a
  recursive function.
- Tree-sitter: `tree-sitter generate` must not change `tree-sitter-ionic/src`,
  and every `.ionic` file must parse without `ERROR`/`MISSING`.

`multitarget.yml` additionally diffs Mach-O vs ELF output against the golden file
`examples/multitarget_test.expected`.

---

## 7. Change discipline

Before any compiler change, internalise this loop:

1. Make the change.
2. `./build.sh` — does it still build?
3. `./build.sh --verify-self-hosting` — is it still a fixed point?
4. Run the relevant `examples/` programs and CI smoke tests.
5. If you touched syntax, run `tree-sitter generate` in `tree-sitter-ionic/` and
   commit the regenerated `src/`.

A change that breaks the fixed point is not done.

---

## Exercises

**08.1 — Reproduce the fixed point by hand.** Run the recipe in `build.sh`'s
header manually: compile `src/main.ionic` with `ionic_self` and with `ionic_new`,
then `cmp` the two `.o` files. Confirm identity and record the byte size.

**08.2 — Break it deliberately.** Make a semantics-preserving-but-output-changing
edit (e.g. add a doc comment that reaches codegen, or reorder two independent
top-level statements), rebuild, and watch `--verify-self-hosting` decide. Revert.

**08.3 — Object keying.** Compile two different outputs in parallel and confirm
their `.o` files are distinct. Explain why a shared `/tmp` object path would be a
bug.

**08.4 — Bootstrap without `ionic_self`.** If you have Rust, run
`cargo build --release`, then compile `src/main.ionic` with `target/release/ionic`.
You have just bootstrapped from source alone.

**08.5 — CI map.** For each step in `.github/workflows/ci.yml`, write one line:
what invariant it protects and which file:line in the repo that invariant lives
in.

---

## Checkpoint 08

- [ ] You can draw the four-arrow bootstrap graph.
- [ ] You can explain object-level (vs binary-level) reproducibility, including
      *why* the linker prevents binary identity.
- [ ] You can state the parallel-build object-keying rule and why it exists.
- [ ] You can complete a change through build → verify → examples without
      breaking the fixed point.

## [Next](09-capstone.md)
