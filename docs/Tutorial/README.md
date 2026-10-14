# Mastering Ionic — Curriculum

A staged, exercise-driven path from *writing Ionic* to *owning the Ionic compiler*.
It assumes you built (or helped build) the language, so it does not stop at syntax:
it treats the compiler, the type system, the memory model, and the self-hosting
fixed point as first-class study material.

Every claim in these units is grounded in this repo. Where the prose docs disagree
with each other or with the shipped behaviour, the unit says so explicitly and
tells you how to verify it yourself (see [Known discrepancies](#known-discrepancies)).

---

## How to use this curriculum

- Work the units **in order**. Each builds on the last.
- **Do the exercises.** Reading a compiler is not mastery; predicting its output,
  breaking it, and fixing it is.
- Every *exercise* that produces output has a **runnable reference** under
  `examples/mastery/`. Compile and run it, then reproduce it from scratch.
- Every *exercise* that should fail names the file under `examples/mastery/fail/`
  and the exact diagnostic you should see.
- Track yourself in [`progress.md`](progress.md). A stage is "done" only when its
  **checkpoint** passes — not when you've read it.

### Ground rules / invariants

- The compiler is a **fixed point**: `ionic_new` compiling its own sources emits a
  byte-identical `.o`. Anything you change must preserve
  `./build.sh --verify-self-hosting`. This is the single most important habit in
  the repo.
- The object is keyed on the **output name** (`<out>.o`), so parallel builds are
  isolated. Don't "fix" that.
- Ionic has **no GC**. Memory is C-style; the reclamation story is a plan, not a
  shipped feature. See unit [07](07-memory-model-and-runtime.md).
- `bool` and `int64` are the **same at runtime**. Do not assume `[1, true]` is a
  type error — it is accepted, and infers `[int64]`.

---

## Prerequisites

- A working `ionic_self` (ships in the repo) — nothing else needed for the
  language-user stages.
- `clang`/`ld` on `PATH` (used to link every program).
- For compiler-owner stages: `./build.sh` produces `./ionic_new` in ~3 s.
- Optional cross-OS bootstrap: Rust + `cargo` (`cargo build --release` →
  `target/release/ionic`).

---

## Stage map

Stage A — Language user (fluency)

- [00 — Orientation](00-orientation.md) — what Ionic is, toolchain, build, verify, first program.
- [01 — Language core](01-language-core.md) — values, types, control flow, strings, format.
- [02 — Types, functions, arrays, structs](02-types-functions-arrays-structs.md) — inference, LUB, indexing, records.
- [03 — Advanced language](03-advanced-language.md) — compound assign, block comments, literals/escapes, CLI args, file I/O, the pitfalls catalogue.
- [04 — Hardware placement](04-hardware-placement.md) — `@cpu`/`@gpu`, `toGpu`/`toCpu`, GPU contexts, the placement checker.
- [05 — Standard library & modules](05-stdlib-and-modules.md) — `std.*`, selective compilation, package layers, relative imports.

Stage B — Compiler owner (mastery)

- [06 — Compiler architecture](06-compiler-architecture.md) — pipeline, IR representations (SoA AST), SoA tokens, optimizer, codegen, diagnostics.
- [07 — Memory model & runtime](07-memory-model-and-runtime.md) — C runtime, allocation, the RC decision, and the missing type oracle.
- [08 — Build, bootstrap & self-hosting](08-build-bootstrap-selfhosting.md) — `ionic_self` vs `ionic_new`, the fixed point, the Rust bootstrap, releases/CI.
- [09 — Capstone: extend the language](09-capstone.md) — an integrated feature build plus real open bugs in the repo.

---

## Where to start depending on your goal

- **Write real Ionic programs well** → Stages A (00–05). ~1–2 weeks.
- **Own and extend the compiler** → A then B (00–09). ~4–6 weeks.
- **Fix the memory model / add reclamation** → 00–03 quickly, then 06, 07, 08 in depth, then the Slice 2 prerequisite in [07 §5](07-memory-model-and-runtime.md).

---

## Known discrepancies (verify these yourself as exercise 00.4)

These are documented *because* they are the kind of detail only a builder knows.
Confirm each with the compiler, not the prose.

1. **Array indexing is 0-based and bounds-checked at runtime.** `a[0]` is the
   first element; `a[len(a)]` aborts with `ionic: index out of bounds`. But the
   bundled `lib/std/data.ionic` (`mean_f`, `min_f`, `max_f`, `stdev_f`,
   `to_csv_line`) treats index 1 as the first element and skips index 0. On a
   direct array built as `[1.0, 2.0, 3.0, 4.0]` this silently drops the first
   element (`mean_f` returns `3.0`, not `2.5`). This is a real bug — see the
   capstone.
2. **`println(arr[i])` prints a pointer, not the element**, for `[string]`
   arrays (element loads lose string-ness at the codegen level). `println(arr[i])`
   for `[int64]` works. This is the latent bug recorded in `docs/memory-model.md §10`.
3. **CLI flags.** `src/main.ionic` accepts only `-o <name>` and source files; the
   legacy self-hosted `compiler.ionic` accepted `--run`. If a doc snippet says
   `ionic foo.ionic --run`, translate it to compile-then-execute:
   `./ionic_new foo.ionic -o foo && ./foo`.
4. **`print` vs `println`**: `print` has no trailing newline (`fputs`); `println`
   appends one (`puts`). Float params are passed as **raw IEEE-754 bit patterns
   in an integer slot**, and `println` of a float prints with `%g`.
5. **`file_exists` is not a builtin in the shipped self-hosted compiler.** It is
   implemented in the legacy `src/compiler.ionic` and in the Rust frontend, but
   is absent from `register_builtins()` in `src/semantic/checker.ionic`, and
   `src/codegen/native.ionic` does not lower it. `docs/language.md`/`guide.md`
   advertise it; the compiler rejects it with `unknown function 'file_exists'`.
   Use `str_len(file_read(path)) > 0` as a probe, or add it (capstone 09.1).
6. **A `tensor@gpu` value may only be *referenced* inside a `gpu { … }` block.**
   Declaring `let d = t.toGpu();` outside a block is fine, but touching `d` in
   CPU context — even passing it to an `@gpu fn` — is rejected with
   `tensor@gpu 'd' used outside gpu block`. Do GPU work inside the block.
7. **`main`'s return value is *not* used as the process exit code.** The generated
   `main` wrapper calls `ionic_user_main(...)` and then unconditionally loads `0`
   into the return register (`src/codegen/native.ionic`, `ncg_main_wrapper`,
   ~line 2055: `a64_load_imm(REG_X0, 0)`). `fn main() -> int64 { return 7; }`
   still exits `0`. `docs/guide.md` says otherwise. Use `exit(code)` — verified to
   work — to set a status.

---

## Conventions used in every unit

- **Learn** — the concepts, grounded in file references.
- **Read the source** — the exact files/functions to open.
- **Exercises** — numbered, with reference files where runnable.
- **Checkpoint** — pass/fail gates you record in `progress.md`.


## Start Now!
[Start](00-orientation.md)
