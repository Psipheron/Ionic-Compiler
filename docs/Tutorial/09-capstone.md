# 09 — Capstone: Own the Compiler

**Goal:** prove mastery by changing the compiler end-to-end — adding a feature,
fixing a real bug, and completing the missing type oracle that unblocks the
memory model — without breaking the self-hosting fixed point. These tasks draw on
every prior unit.

Read this unit only after 00–08. Each task is graded by the self-hosting
checkpoint plus a concrete behavioural test.

---

## Task 09.1 — Add a builtin (guided, end-to-end)

Add `str_repeat_char(int64 code, int64 n) -> string` (or any small builtin) and
wire it through every layer. This is the canonical "add to the language" drill.

Touch points, in order:

1. **Checker** — `register_builtin("str_repeat_char", 2, TY_STR, "")` in
   `register_builtins()` (`src/semantic/checker.ionic:~638`).
2. **Codegen** — map the Ionic name to a C symbol in the call-lowering switch
   (`src/codegen/native.ionic`, the `real_callee = "ionic_…"` remaps), and add it
   to any string-returning type lists so `ncg_last_ty` is set.
3. **Runtime** — implement `char *ionic_str_repeat_char(int64_t code, int64_t n);`
   in `src/ionic_model_runtime.c`.
4. **Example + smoke test** — add a program under `examples/` that exercises it.

**Grade:** `./build.sh --verify-self-hosting` passes; your example prints the
expected output; the compiler emits no new warnings.

---

## Task 09.2 — Fix the `lib/std/data.ionic` off-by-one

The stats helpers assume 1-based arrays; Ionic arrays are 0-based (unit 02 §3.1).

1. Decide the **contract**: the fix is either (a) rewrite the helpers to be
   0-based, or (b) keep the helpers and document a 1-based "statistics array"
   convention. Justify your choice in one paragraph.
2. Implement it. `mean_f([1.0,2.0,3.0,4.0])` must return `2.5`; `max_f` → `4.0`;
   `min_f` → `1.0`; `stdev_f` must be correct.
3. Check `to_csv_line` and the `min_f`/`max_f` names — note these collide with
   `std.math.min_f`/`max_f` (a real naming hazard). Consider whether importing
   both modules is even legal.

**Grade:** a new example asserts the four numeric results; the compiler still
self-hosts; `docs/stdlib.md` is updated to match.

---

## Task 09.3 — Unblock the memory model (the big one)

Implement the **type-tagging prerequisite** from `docs/memory-model.md §10` (units
06 §8, 07 §5). This is the highest-value work in the repo.

Deliverable A — the oracle: carry the semantic checker's per-expression static
type into codegen so `ncg_last_ty` is correct on **every** path, in particular
the `ND_INDEX` arm (`src/codegen/native.ionic:~1159`) and `ND_LET` with non-literal
string initializers. Replace the shadow-prone name tables
(`local_find`/`local_get_str`/`ncg_string_expr`) with this uniform oracle.

Deliverable B — the observable fix: `println(names[0])` on a `[string]` array
prints `alice`, not a pointer. Add an example asserting it.

Deliverable C (stretch): re-attempt the RC slice on top of A, using the log in
`docs/memory-model.md §10`. The churn loop must drop from ~160 MB to ~1 MB RSS.

**Grade:**
- A: `--verify-self-hosting` passes; a table of expression kinds each prove the
  oracle's classification.
- B: the string-array print example is correct.
- C: RSS measured before/after on a churn loop; no use-after-free across the full
  test suite.

---

## Task 09.4 — Give the optimizer a new pass (or the SSA design a proof)

Pick one:

- **New optimizer pass** in `src/opt/opt.ionic` — e.g. strength reduction
  (`x * 2 ** k` → `x << k`) or a second inlining tier (single-`if` bodies).
  Must be semantics-preserving and keep the fixed point.
- **SSA design** — extend the SSA exercise in `docs/homework_solutions.md` HW 4.1
  into an implemented design doc, then implement a minimal SSA construction for a
  single straight-line function and prove the emitted object is unchanged for a
  suite of programs.

**Grade:** a before/after metric (object size, or number of emitted instructions
for a benchmark); fixed point preserved.

---

## Task 09.5 — Harden an invariant

Choose one CI invariant and add a regression test plus, if warranted, a compiler
fix:

- The **shadowing** regression (top-level `let n` vs parameter `n`) — CI tests it;
  find the root cause in `opt.ionic`'s const-propagation and harden it.
- **Bounds-check** message stability.
- The **packaging** `$IONIC_ROOT` path from an unrelated CWD.

**Grade:** your test fails against the pre-fix compiler and passes after; CI would
catch a regression.

---

## Task 09.6 — Extend the tree-sitter grammar

If you added syntax in 09.1–09.4, update `tree-sitter-ionic/grammar.js`, run
`tree-sitter generate`, commit the regenerated `src/`, and add a highlight query
if relevant. CI (`ci.yml`) fails if `src/` is stale or any file has a parse error.

**Grade:** `tree-sitter generate` is a no-op on a clean tree; all `.ionic` files
parse cleanly.

---

## Mastery rubric

You have mastered Ionic when you can truthfully say:

- [ ] I can write idiomatic Ionic and predict its output from the types alone.
- [ ] I can trace a program through lex → parse → check → opt → codegen and name
      every IR representation.
- [ ] I can add a language feature and wire it through all three layers
      (checker, codegen, runtime) without breaking self-hosting.
- [ ] I understand the memory model's decision, its shipped slice, and the exact
      prerequisite for the next slice.
- [ ] I can fix a real bug in the compiler, the stdlib, or the runtime, and prove
      it fixed.
- [ ] I preserve the object-level self-hosting fixed point on every change.
