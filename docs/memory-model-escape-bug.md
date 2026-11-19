# Array RC stage 2 — the escape bug, in full

## The claim

Array RC stage 1 was proved unsound by ASan. The repair is retain-on-copy,
which is implemented. But retain-on-copy alone is *not* enough: the escape path
has to change with it, and the escape path has the same shape in the string pass,
which has been shipping.

**This is the record of that repair. It is fixed**: the escape now retains and
keeps the source slot's claim, and `scripts/arr_rc_check.sh` gates it under
ASan and UBSan.

## The bug

An escape — storing a value into a container the compiler cannot follow —
used to **drop the slot's claim**:

```
ncg_note_str_escape(n):  local_set_own_str_name(name, 0)
ncg_note_arr_escape(n):  local_set_own_arr_name(name, 0)
```

The reasoning was "the container holds the pointer now, so the slot must not
release it; leaking is the safe direction".

That reasoning is wrong in two ways.

**1. The count is now short by one.** The slot's claim was one reference. The
container is a *second* name for the same block. Dropping the claim without
retaining means the container's pointer has no count behind it: the block has
one reference, the slot's, and the slot no longer releases it — so the count
sits at 1 forever and the block is never freed. The leak is real, but it is
not the leak that was documented.

**2. The next assignment frees the container's value.** This is the
use-after-free. The escape clears the claim; a *later* assignment to the same
slot raises a fresh claim over whatever the slot holds — and on every path
where that assignment did not run, the slot still holds the value the
container is pointing at. The exit block then releases it.

Concretely, for `s = str_concat("he","llo"); outer.push(s);` with a
conditional reassignment:

```
let s = str_concat("he", "llo");   // rc = 1, slot claims it
outer.push(s);                     // escape: claim dropped, rc still 1
// ... the container holds `s`, and rc = 1 is the *container's* count now,
// but the compiler thinks the slot owns nothing
s = str_concat("wor", "ld");        // on the path where this DOES NOT run,
//                                    the slot still holds "hello"
// exit block: the slot claims nothing, so nothing is released here
```

The string case that failed is `escape_conditional_reassign.ionic`: the
container's element is freed by the *assignment path* when the assignment *does*
run, and by the *exit block* when it does not. Both are use-after-free.

The array cases are the same shape, and ASan found three of them:
`escape_loop`, `escape_alias`, and `global_store_array`.

## The fix

An escape **transfers** a reference. The correct bookkeeping is therefore:

- the slot **keeps** its claim, and
- the escape site **retains** the value, so the count goes up by one.

The extra count belongs to the container, and nothing ever releases it — that
is the accepted leak for a container escape (docs/memory-model.md §4.2). It is
a leak, but it is a *correct* leak: the container's pointer has a count behind
it, so no release can free it out from under the container.

With the retain in place, the slot's claim is still correct, so a later
assignment releases the slot's value exactly as it should, and the exit block
releases it too. The container's count is untouched by either.

## Why the retain has to be at the escape site

The retain cannot be folded into the `let` or `=` paths, because those do not
know whether the value is going into a container. `push`, an array-literal
element, a struct field, an indexed assignment and a global store are the five
places a value enters a container this pass cannot follow, and each has to
retain the value it stores.

The retain is emitted only for a value this frame *owns*. A borrow — a
parameter, a global, an element load — is not ours to count, and the container
that stores it is the caller's problem, exactly as it is for the claim.

## What this changes in the docs

§11.4 and §11.6 of docs/memory-model.md described the old behaviour. Both are
now corrected:

- §11.4 said "There is deliberately **no retain-on-copy**: `let b = a` poisons
  `b` instead of claiming it." That was false as of the repair: the copy
  retains, and the claim is raised.
- §11.4 said the escape "drops its claim". That was false as of the repair:
  the escape retains and the claim is kept. §11.4 now has a "The escape rule"
  subsection stating the rule and recording why the old one was wrong.
- §11.6 said "Array stage 1 does not retain on copy (§11.4). `ring.push(ring)`
  therefore stores a borrowed pointer and raises no count, so the array's count
  still walks 1 → 0 ... A refcount cycle over an array cannot be constructed
  while copies do not retain". With retain-on-copy, a cycle *can* be
  constructed, so the cyclic-benchmark obligation this section declared
  "unnecessary" became live. It is now met: §11.6 records the measured growth
  (a self-referential array leaks, by design) and the two halves of the
  obligation — a cycle-free benchmark must go flat, and a deliberately cyclic
  one must *not* crash.
- `scripts/probes/cyclic_array.ionic` carries the array half of that second
  obligation. It asserts *safety*, not reclamation: a cycle that was reclaimed
  would be the bug, because the count would have reached zero with the array
  still reachable.

## The gate that keeps it fixed

`scripts/arr_rc_check.sh` compiles each probe with the real compiler, relinks
it against the real runtime under AddressSanitizer and UndefinedBehaviorSanitizer,
and runs it. A failure is a real memory error in the emitted code, not a model
of one.

Leak detection is off in that harness on purpose: the accepted leak for a
container escape is a *decision*, and a leak report would flag it as a bug.
