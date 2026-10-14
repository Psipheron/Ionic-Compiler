# 00 — Orientation

**Goal:** understand what Ionic is, how it is built, how to run it, and how to
prove it still works. By the end you can compile, run, link, and self-verify, and
you have seen the whole pipeline run once.

---

## 1. What Ionic is

Ionic is a **statically-typed, self-hosting** language that compiles directly to
**native ARM64 Mach-O** object files (Linux ELF is also emitted). There is no LLVM
in the runtime path — a legacy LLVM backend exists (`--llvm`) for debugging only.

Two properties make it unusual and define everything else:

1. **Self-hosting.** The shipped compiler, `ionic_new`, is written in Ionic and
   compiled by itself. The compiler source tree (`src/*.ionic`) is the language's
   largest program.
2. **Hardware placement is part of the type system.** `tensor@cpu` and
   `tensor@gpu` are distinct types; the checker rejects crossing the device
   boundary without an explicit `.toGpu()`/`.toCpu()`.

Read: `README.md` (top through "Features"), `docs/language.md` (skim).

---

## 2. The toolchain in the tree

- `ionic_new` — the self-hosted compiler (built from `src/*.ionic`). What you use.
- `ionic_self` — the previous-generation compiler binary, used to rebuild
  `ionic_new`. Ships so no toolchain is needed to bootstrap.
- `build.sh` — builds `ionic_new` from the split source tree via `ionic_self`.
- `target/release/ionic` — a Rust/LLVM bootstrap compiler for cross-OS bootstrap.
- `src/ionic_model_runtime.c` — the C runtime linked into every program you build.

Read: `README.md` ("Build" … "Self-hosting cycle"), `build.sh` (all of it).

---

## 3. Build, run, verify

```sh
# Build the compiler (uses the committed ionic_self). ~3 s.
./build.sh                                  # → ./ionic_new

# Compile and run a program.
./ionic_new examples/hello.ionic -o /tmp/hello && /tmp/hello

# Prove the compiler is still a fixed point (compares emitted .o files).
./build.sh --verify-self-hosting            # → "✓ object files byte-identical"
```

Note: each compile also writes an object next to the output as `<out>.o`, and
**the object path is derived from the output name on purpose** (parallel-build
safety). See `src/main.ionic` near `obj_path`.

> The legacy `compiler.ionic` accepted `--run`; `src/main.ionic` does **not**. Any
> doc snippet with `--run` should be read as "compile with `-o`, then execute".

---

## 4. Your first program, decomposed

```ionic
fn main() -> int64 {
    print("Hello, Ionic!");
    return 0;
}
```

Reference: `examples/mastery/00_hello.ionic`.

- `fn main() -> int64` is the entry point. ⚠️ Its **return value is ignored** by
  the generated `main` wrapper (which always exits 0) — use `exit(code)` to set a
  status. See the hub's discrepancy 7. Programs with **no** `fn main` run their
  top-level statements directly (`examples/hello.ionic` does this).
- `print` writes with no trailing newline; `println` appends one (see the
  discrepancy note in the hub).

Verify the exit code (this uses `exit(7)`):

```sh
./ionic_new examples/mastery/00_hello.ionic -o /tmp/h && /tmp/h; echo "exit=$?"
```

---

## 5. What you'll see internally (a taste)

Compile a program and look at the artefacts:

```sh
./ionic_new examples/mastery/00_hello.ionic -o /tmp/h
ls -l /tmp/h /tmp/h.o        # an object file and a linked binary
file /tmp/h                  # Mach-O 64-bit executable arm64
```

The compiler's stages (unit 06) are lex → parse → semantic check → optimize →
codegen → link. `src/main.ionic` is the driver that wires them together.

---

## Exercises

**00.1 — Hello, exit code.** Write `main() -> int64` that prints one line, then
prove the return-value surprise: first `return 7;` (observe exit 0), then
`exit(7);` (observe exit 7). Reference: `examples/mastery/00_hello.ionic`.

**00.2 — Top-level script.** Write a program with **no** `fn main` that uses only
top-level statements, prints three lines, and exits 0. Reference:
`examples/mastery/00_toplevel.ionic`.

**00.3 — Build and verify.** Run `./build.sh` then
`./build.sh --verify-self-hosting`. Record the byte size of the identical object.
If it fails, you have already learned the most important invariant in the repo.

**00.4 — Discrepancy hunt.** Prove or disprove each of the *seven* "Known
discrepancies" in the hub using the compiler. In particular: reproduce the
`mean_f` off-by-one, the `println(names[0])` pointer, the missing `file_exists`,
and the ignored `main` return value. Reference:
`examples/mastery/00_discrepancies.ionic`.

---

## Checkpoint 00

- [ ] `./build.sh` and `./build.sh --verify-self-hosting` both succeed.
- [ ] You can compile any `.ionic` file and explain where the `.o` goes and why.
- [ ] You reproduced at least two of the seven known discrepancies.
- [ ] You can state, in one sentence, what "self-hosting fixed point" means.

## [Next](01-language-core.md)
