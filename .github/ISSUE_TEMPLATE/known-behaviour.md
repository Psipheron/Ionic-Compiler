---
name: "Not a bug: known behaviour"
about: The two documented leak shapes, division by zero, and unsupported platforms
title: ""
labels: ""
assignees: ""
---

<!--
Delete this template and open a normal bug report if your case is genuinely one
of these being worse than described. This page exists so the known limits are
written down once instead of re-derived in every thread.
-->

## Check these first

**1. A leak in a hot loop.**
Two allocation shapes leak by design, roughly 48 bytes per iteration:

```ionic
for (i in 0..n) {
    let t = int64_to_str(i);   // leaks: a `let` re-executed by a loop
}

for (i in 0..n) {
    consume(int64_to_str(i));  // leaks: a producer temp passed to a user fn
}
```

The reclaimed form binds outside the loop and assigns inside it:

```ionic
mut t = "";
for (i in 0..n) { t = int64_to_str(i); }
```

Full explanation, including why the compiler cannot close these yet:
`docs/language.md` → "Memory: what the compiler reclaims, and what it does not",
and `docs/memory-model.md` §9. `scripts/rss_check.sh` asserts both: flat for the
assignment form, bounded for the `let` form.

**2. Integer division by zero returns 0.**
`10 / 0` is `0` on both backends; there is no trap and no diagnostic. That is
ARM64 `sdiv` behaviour, not a decision anyone made deliberately. If you were
expecting an abort, this is why you did not get one — it is a documentation gap
rather than a divergence between backends.

**3. Cycles over mutable aggregates are an accepted leak.**
`let q = p` aliases, and `a.push(b)` nests aggregates, so a cycle is
constructible from ordinary code. Reference counting reclaims acyclic graphs
only, and there is no collector by design.

**4. Platforms.**
Release tarballs are **macOS arm64 only**. Linux arm64 is built and tested from
source in CI but not shipped. Intel macOS is not supported.

## Still think it is a bug?

Open a regular bug report and include which of the above you ruled out and why.
