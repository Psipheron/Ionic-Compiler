# Progress Tracker

Copy this file's checkboxes into your own notes, or edit it in place (don't
commit personal progress to the repo). Tick a box only when you can do the thing
**from memory**, not while reading.

Suggested cadence:

- **Days 1–2** — Unit 00. Build, verify, reproduce two discrepancies.
- **Days 3–5** — Units 01–02. Language core + types/arrays/structs.
- **Days 6–7** — Units 03–04. Advanced language + placement.
- **Day 8** — Unit 05. Stdlib & modules.
- **Day 9** — Review + write a small program using everything so far.
- **Days 10–12** — Unit 06. Read the whole pipeline in `src/`.
- **Days 13–14** — Unit 07. Memory model; write the type-oracle plan.
- **Day 15** — Unit 08. Build, bootstrap, CI, fixed point.
- **Days 16+** — Unit 09 capstone, one task at a time.

A realistic full pass is 4–6 weeks at a few focused hours a week.

---

## Stage A — Language user

### Unit 00 — Orientation
- [ ] `./build.sh` succeeds
- [ ] `./build.sh --verify-self-hosting` succeeds
- [ ] Exercise 00.1 — hello with exit code 7
- [ ] Exercise 00.2 — top-level script, no `main`
- [ ] Exercise 00.3 — recorded identical-object byte size: ______
- [ ] Exercise 00.4 — reproduced ≥2 known discrepancies
- [ ] I can state what "self-hosting fixed point" means in one sentence

### Unit 01 — Language core
- [ ] Exercise 01.1 — literals
- [ ] Exercise 01.2 — exclusive ranges
- [ ] Exercise 01.3 — int/float widening
- [ ] Exercise 01.4 — operator precedence
- [ ] Exercise 01.5 — immutable-assignment error reproduced
- [ ] I can state the `bool ≡ int64` rule and the `str_eq` requirement

### Unit 02 — Types, functions, arrays, structs
- [ ] Exercise 02.1 — inference trace for `42 + 1` / `3.14 * x`
- [ ] Exercise 02.2 — array semantics + bounds error
- [ ] Exercise 02.3 — reproduced the `mean_f` off-by-one
- [ ] Exercise 02.4 — struct field mutation under `let`
- [ ] Exercise 02.5 — string-array print bug reproduced
- [ ] I can describe propagation-based inference in three sentences

### Unit 03 — Advanced language
- [ ] Exercise 03.1 — compound assignment
- [ ] Exercise 03.2 — escapes
- [ ] Exercise 03.3 — CLI arguments + exit code
- [ ] Exercise 03.4 — file I/O round trip
- [ ] Exercise 03.5 — triggered ≥6 of the 9 pitfalls
- [ ] I can explain how `x += e` reaches codegen

### Unit 04 — Hardware placement
- [ ] Exercise 04.1 — tensor mismatch hint reproduced
- [ ] Exercise 04.2 — both `@gpu fn` violations found
- [ ] Exercise 04.3 — contradictory annotation rejected
- [ ] Exercise 04.4 — correct placement program runs
- [ ] Exercise 04.5 — located the CPU-only-I/O enforcement
- [ ] I know where placement is enforced and where it is *not*

### Unit 05 — Stdlib & modules
- [ ] Exercise 05.1 — wildcard vs named `.o` are identical
- [ ] Exercise 05.2 — named import emits only used symbols
- [ ] Exercise 05.3 — relative import builds
- [ ] Exercise 05.4 — project-local package works
- [ ] Exercise 05.5 — resolution-miss message captured
- [ ] I can list the three resolution layers

---

## Stage B — Compiler owner

### Unit 06 — Compiler architecture
- [ ] Exercise 06.1 — driver traced
- [ ] Exercise 06.2 — token/AST kinds for `+=` and `tensor@gpu`
- [ ] Exercise 06.3 — compound-assign desugaring quoted
- [ ] Exercise 06.4 — unrolling shown (and prevented by `break`)
- [ ] Exercise 06.5 — the missing `ncg_last_ty` site pinpointed
- [ ] Exercise 06.6 — named the files a new builtin must touch
- [ ] I can name all five optimizer passes

### Unit 07 — Memory model & runtime
- [ ] Exercise 07.1 — measured the string leak (RSS growth)
- [ ] Exercise 07.2 — quoted the intentional-leak comment + `IonicArray`
- [ ] Exercise 07.3 — quoted the bounds check
- [ ] Exercise 07.4 — traced the four `model` RAII sites
- [ ] Exercise 07.5 — wrote the type-oracle plan
- [ ] I can explain why RC was reverted (soundness, not coverage)

### Unit 08 — Build, bootstrap & self-hosting
- [ ] Exercise 08.1 — reproduced the fixed point by hand
- [ ] Exercise 08.2 — deliberately broke and reverted
- [ ] Exercise 08.3 — proved parallel object isolation
- [ ] Exercise 08.4 — bootstrapped without `ionic_self`
- [ ] Exercise 08.5 — mapped every CI step to an invariant
- [ ] I can draw the four-arrow bootstrap graph

### Unit 09 — Capstone
- [ ] Task 09.1 — added a builtin end-to-end
- [ ] Task 09.2 — fixed the `data.ionic` off-by-one
- [ ] Task 09.3A — type oracle: `ncg_last_ty` correct on all paths
- [ ] Task 09.3B — `println(names[0])` prints the string
- [ ] Task 09.3C — RC slice: churn RSS ~160 MB → ~1 MB (stretch)
- [ ] Task 09.4 — new optimizer pass or SSA proof
- [ ] Task 09.5 — hardened an invariant with a regression test
- [ ] Task 09.6 — tree-sitter grammar in sync

---

## Notes / open questions

_(log things you couldn't explain or that surprised you — they are the best
study targets)_

-
-
-
