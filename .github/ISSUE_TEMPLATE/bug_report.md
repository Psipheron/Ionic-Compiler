---
name: Bug report
about: Something produced the wrong answer, crashed, or failed to build
title: ""
labels: bug
assignees: ""
---

<!--
Before filing, please check:
  - `ionic --version` — a fix may already be released.
  - The two documented leak shapes in docs/language.md ("Memory: what the
    compiler reclaims, and what it does not"). Those are known, not bugs.
  - Existing issues, including closed ones.
-->

## What happened

<!-- One or two sentences. "X prints Y but should print Z" is ideal. -->

## What you expected

<!-- The correct output, the compile error you wanted, or "should not crash". -->

## Reproduce

<!--
The most useful thing in this report. A minimal program beats a description.

```ionic
fn main() -> int64 {
    // ...
    return 0;
}
```

```sh
ionic repro.ionic -o repro && ./repro
```
-->

```ionic

```

## Actual output

<!--
Paste it verbatim, including diagnostics. If it crashed, say so and give the
exit status — a compiler that segfaults instead of reporting an error is worth
knowing about even when the input is extreme.
-->

```

```

## Which compiler

<!-- `ionic --version` output. Tick what you used. -->

- [ ] `ionic` from a release tarball or Homebrew
- [ ] `ionic_new` built by `./build.sh`
- [ ] `target/release/ionic` (the Rust bootstrap — `cargo build --release`)

```
version:
```

## Platform

<!-- e.g. macOS 15.2 on Apple silicon (M2). Releases are macOS arm64 only today. -->

```
OS + architecture:
```

## Anything else

<!--
Useful extras: does it reproduce on the other backend? Did it work in an older
release? A link to the program, if it is public.
-->
