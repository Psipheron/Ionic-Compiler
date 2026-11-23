# Security policy

Ionic is a compiler: it parses untrusted text and emits and runs native code.
The interesting failure modes are memory-safety bugs in the compiler or its C
runtime, and anything that lets a *source file* execute code the person
compiling it did not intend.

## Reporting a vulnerability

**Do not open a public issue.** Use GitHub's private advisory flow:

https://github.com/Psipheron/Ionic-Compiler/security/advisories/new

That keeps the report visible only to maintainers until a fix ships. If you
cannot use it, email the maintainer listed on the repository profile and say you
are reporting a security issue.

Please include:

- the version (`ionic --version`) and platform,
- the smallest program or input that triggers it,
- what happens, and what you think the impact is,
- whether it needs a specific backend or an installed ML library.

## What is in scope

- **Memory-safety bugs** in `src/ionic_model_runtime.c` or the emitted code that
  a program can trigger from ordinary Ionic source. A use-after-free the
  language makes reachable is a real bug; that class has already happened once
  (see `docs/memory-model-escape-bug.md`).
- **Compiler crashes on untrusted input.** A `.ionic` file should never be able
  to crash the compiler with a segfault, read outside its own bounds, or execute
  anything. Today a pathological but valid input can still crash it — deeply
  nested parentheses overflow the parser's stack. That is known and is being
  treated as a robustness bug rather than a security boundary: compiling a file
  you did not write is not yet a sandbox, so do not rely on it as one.
- **Build-time code execution** that is not the program's own doing. The
  compiler shells out to `clang`/`ld`; it must not do anything else with the
  contents of a source file.
- **Supply chain** problems in the release process: a tarball or Homebrew
  formula that does not match the tagged source.

## What is not in scope

- A compiled Ionic program misbehaving at run time. The compiler is not a
  sandbox; if you compile a program that does something harmful, that program
  does it.
- The documented leaks (two allocation shapes, cycles over mutable aggregates).
  They are limitations with known behaviour, not vulnerabilities. Filing one
  through this channel just delays real reports.
- `10 / 0` returning `0`. Undocumented and surprising, but it is consistent
  across backends and matches the hardware instruction.
- Issues in a third-party ML library (MLX, ONNX Runtime) reached through Ionic.
  Report those upstream; tell us if Ionic's binding is what makes it reachable.

## Supported versions

Only the latest stable release and the current prerelease receive fixes. There
are no long-term-support branches; the language and runtime are still moving.

| Version | Supported |
|---|---|
| Latest `vX.Y.Z` stable | Yes |
| Current `vX.Y.Z-beta.N` | Yes, as a prerelease |
| Anything older | No — please reproduce on the latest first |

## How releases are checked

A release is only published if the test gates and an artifact smoke test pass on
the exact tagged tree — the gates run in the release workflow, not only in CI.
Tarballs are published with a SHA256 alongside them, and the Homebrew formula
pins a checksum. If a published artifact does not match its checksum, treat that
as a security report.
