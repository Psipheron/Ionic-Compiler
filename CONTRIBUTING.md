# Contributing to Ionic

Thanks for taking a look. This is a self-hosting compiler, which makes a few
things different from an ordinary codebase — mostly that there is one invariant
you must not break, and one habit that makes everything else easy.

## The one invariant: the fixed point

`ionic_new` compiling its own sources must emit a **byte-identical** object
file. Everything in this repo rests on that claim, so every change must preserve
it:

```sh
./build.sh --verify-self-hosting
```

Green means `ionic_self` and the freshly built `ionic_new` agree. If you see
`✗ DIFFER — self-hosting is BROKEN`, the change is not ready — even if the
feature works, and even if every other test passes.

A related quirk you should not "fix": the object file is keyed on the **output
name** (`<out>.o`), which is what lets parallel builds coexist. It looks like a
missing feature; it is deliberate.

## Getting set up

No Rust and no toolchain is needed for the language itself — a bootstrap binary
ships in the repo.

```sh
./build.sh                        # ~3 s, builds ionic_new from src/*.ionic
./ionic_new examples/hello.ionic -o /tmp/hello && /tmp/hello
```

You do need `clang`/`ld`, because every program is linked with it.

The Rust tree (`Cargo.toml`, `src/*.rs`) is a separate, dev-only cross-OS
bootstrap:

```sh
cargo build --release             # target/release/ionic
```

It is **not** the shipped compiler. The tarball contains the self-hosted one.

## Running the gates

These are what CI runs, and now also what a release runs before it publishes:

```sh
./scripts/rc_check.sh             # every string producer returns a counted block
./scripts/arr_rc_check.sh         # array aliasing and escapes, under ASan+UBSan
./scripts/io_check.sh             # runtime conformance, both backends
./scripts/cli_check.sh            # --help/--version, version sources, tag
./scripts/model_raii_check.sh     # exactly one free per owned model
./scripts/placement_check.sh      # pool budget and transfer moves
./scripts/rss_check.sh            # leak regression, both churn shapes
```

`io_check.sh` and `cli_check.sh` also take `BACKEND=rust ./scripts/…`.

Two of these are easy to miss and worth understanding:

- **`rss_check.sh` measures two shapes on purpose.** Assignment into a `mut` slot
  must be flat in N; a `let` re-executed by a loop is bounded, because it is a
  known and documented leak. Do not "fix" the second by simply loosening the
  ceiling.
- **`cli_check.sh` compares the version to the git tag.** On a release it refuses
  to pass unless they agree. `CLI_REQUIRE_TAG=1` makes a missing tag a failure
  rather than a skip.

## Changing the compiler

The pipeline is lex → parse → check → opt → codegen, all in `src/*.ionic`.
`docs/Tutorial/06-compiler-architecture.md` walks it. Notes that save time:

- `src/codegen/native.ionic` is the real backend. The Rust `src/codegen.rs` must
  be kept working because CI bootstraps through it, but it is not what ships.
- Anything observable must match on **both** backends. `io_check.sh` exists
  because they disagreed about `print`, `file_write` and `file_exists` for a
  long time, each in a way that exited 0.
- Adding a builtin touches three places (checker, codegen, runtime) plus the
  type oracle if `print` must dispatch on its return type. `docs/Tutorial/09-capstone.md`
  Task 09.1 lists them in order.

## Adding a test

Prefer a probe over a bespoke test.

- Runtime behaviour → drop a `.ionic` file into `scripts/io_probes/`. Each
  declares its own `// Expected:` lines and `io_check.sh` picks it up with no
  script change. It runs on both backends.
- A leak shape → an example under `examples/`, wired into `rss_check.sh`.
- A new gate → `scripts/`, then add it to **both** `.github/workflows/ci.yml`
  and `.github/workflows/release.yml`. A gate that only runs in CI does not
  gate a release, which is how a stdlib bug once shipped.

## Reporting bugs

Open an issue, using the bug-report template. The two things that make a report
actionable are the **exact version** (`ionic --version`) and a **minimal
reproducing program**. A description without those usually cannot be acted on,
because the same symptom can come from three different layers.

Before filing, skim `.github/ISSUE_TEMPLATE/known-behaviour.md` — the two leak
shapes, division by zero, and the supported platforms are documented limits, not
bugs.

For a memory-safety problem, use a private security advisory instead: see
[SECURITY.md](SECURITY.md).

## Pull requests

- Keep the fixed point green; say in the PR that `--verify-self-hosting` passes.
- One logical change per PR. If a fix exposes a second bug, that is a second PR —
  it also makes the first one reviewable.
- If you change behaviour, change the documentation in the same PR. `docs/` is
  the product surface here, and it has drifted before: the language reference
  once claimed `[0]` was an empty array, which produced three real bugs in the
  bundled stdlib.
- Write the commit message for the person reading `git log` in a year: what was
  wrong, why, and how you know it is fixed.

## Licence

MIT. By contributing you agree your work is licensed under the same terms. Note
that the runtime (`src/ionic_model_runtime.c`) is compiled into users' binaries,
so its notice requirement reaches their distributed executables.
