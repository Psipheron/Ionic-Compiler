# 07 — Memory Model & Runtime

**Goal:** understand exactly how Ionic allocates and (doesn't) free memory today,
why the design chose deterministic ownership, the shipped `model` RAII slice, and
the one prerequisite blocking the reference-counting slice. This unit is the
heart of becoming a compiler *owner* rather than a compiler *user*.

Primary source: `docs/memory-model.md`. Secondary: `src/ionic_model_runtime.c`,
`src/codegen/native.ionic`.

---

## 1. The shipped reality

Ionic today has **no GC, no reference counting, no lifetimes**. It is C-style:

- **Strings** are `malloc`'d C strings (`char *`), treated as immutable. Every
  `str_concat`, `int64_to_str`, `float64_to_str`, `str_slice`, `format`,
  `char_to_str`, `file_read`, `read_line` allocates a fresh buffer and never frees
  it. The runtime says so:

  ```c
  return buf; /* caller owns — but Ionic treats strings as const; leak is acceptable for now */
  ```

- **Arrays** have a header `struct IonicArray { i64 len; i64 cap; void *data; }`
  (`src/ionic_model_runtime.c:53`) with a separate `malloc`/`realloc`'d data region.
- **Structs** are `malloc`'d at construction (`ND_STRUCT_LIT`).
- The only explicit free is `model_free(m)`.

Consequence: any long-running program that churns strings/arrays/structs grows
without bound. The compiler itself leaks for this reason.

---

## 2. Runtime entry points to know

Read `src/ionic_model_runtime.c`:

- I/O: `ionic_print(ln)(_int64)(_float64)`, `ionic_get_arg`, `ionic_file_read`,
  `ionic_file_write`, `ionic_read_line`, `ionic_getenv`, `ionic_system`.
- Strings: `ionic_str_concat`, `ionic_int64_to_str`, `ionic_float64_to_str`,
  `ionic_str_slice`, `ionic_format`, `ionic_char_to_str`, `ionic_str_hash`.
- Arrays: `ionic_array_alloc`, `ionic_array_get`, `ionic_array_set`,
  `ionic_array_len`, `ionic_arr_reset`, `ionic_array_push`.
- Floats: everything uses an **int64 bit-pattern convention** — float values and
  float args/returns are passed as raw IEEE-754 bits in integer slots.
- Models: `ionic_load_model`, `ionic_model_forward`, `ionic_model_free`,
  `gguf_*`, `piper_forward`, `write_wav`.

`ionic_array_get` (`:472`) is the bounds check behind `ionic: index out of bounds`.

---

## 3. The decision: deterministic ownership

`docs/memory-model.md §3` adopts three mechanisms, all deterministic and local
(nothing walks the heap, nothing pauses):

1. **Non-atomic reference counting** for shared immutable data — strings, array
   data, tensor storage.
2. **Unique ownership + RAII** for scarce, device-affine resources — `model`,
   GPU buffers, file/socket handles.
3. **Arena scopes** for compiler-internal temporaries (one per phase).

And the key theorem (§6): **cycles are structurally impossible**, because shared
data is immutable and mutable ownership is unique/affine. Therefore every
reference graph is a DAG and refcounting alone suffices — no cycle collector,
ever.

---

## 4. Mechanism details (from the design)

- Header: 8-byte refcount immediately before the payload; `rc_alloc(n)` returns
  `payload + 8`. `rc_retain`/`rc_release` are single-threaded, non-atomic.
- Literals are wrapped (`ionic_str_lit`) as pinned managed copies keyed by rodata
  address, so `release` on a literal is a safe no-op.
- Insertion points (compile time, using the type oracle):
  - **retain** on copy into another owning slot (assignment, field store, array
    store/`push`, `return`);
  - **release** at end of block, and on reassignment (retain-new-first so `x = x`
    is safe);
  - **release** caller temporaries after a call unless ownership transferred.

---

## 5. The blocker — and why it is the same bug as unit 06 §8

The RC slice (`docs/memory-model.md §10`) was implemented, measured working, and
**reverted**. The churn loop dropped from **160 MB to 1.6 MB** peak RSS — then it
was backed out because:

> the codegen cannot reliably classify a value as a managed string.

`ionic_rc_release` was firing on non-managed pointers (stack/rodata addresses).
The insertion logic gates on "is this expression a string?", and that predicate
is not knowable from the current backend state:

1. `local_find` returned the *last* slot for a name (shadowing) while
   `local_get_str` returned the *first* — a shadowed integer could be treated as
   a string.
2. `ncg_last_ty` is set only on some paths — `ND_LET` with a non-literal string
   initializer, and array/struct element **loads**, never set it.
3. The observable symptom is the pre-existing `println(arr[0])` pointer bug
   (unit 02 §3.2).

A single mis-placed `release` turns a benign leak into a use-after-free — worse
than the leak. So the requirement is soundness, not coverage.

**Prerequisite:** a **type-tagging pass** that carries the semantic checker's
static types into codegen and makes `ncg_last_ty` correct on **every** expression
path, especially element loads. This also fixes the string-array print bug. The
shadow-prone name tables (`local_find`/`local_get_str`/`ncg_string_expr`) should be
replaced by that uniform oracle.

---

## 6. The `model` RAII slice — shipped

Unlike RC, the `model` half needs no value-type oracle because handles are
released explicitly and deterministically. It is implemented in
`src/codegen/native.ionic`:

- `local_owns_model(name)` tracks which locals own a `model` (`:690`);
- at function end, owned `model` locals are freed (`:733`–`:751`);
- on reassignment, the old model is freed first (`:1571`–`:1588`);
- `return` **transfers** ownership — the source is not freed (`:1660`);
- explicit `model_free(m)` clears the local's ownership so it isn't double-freed.

Programs that don't use `model` are byte-identical in output — a hard constraint
worth internalising (unit 08).

---

## Exercises

**07.1 — Measure the leak.** Write a loop that concatenates strings ~1e6 times
and observe RSS growth (e.g. `/usr/bin/time -l`). Confirm strings are never
freed. Reference: `examples/mastery/07_leak.ionic`.

**07.2 — Read the allocators.** In `src/ionic_model_runtime.c`, quote the line
comment that documents the intentional string leak, and quote the `IonicArray`
struct. Explain the 24-byte header.

**07.3 — Find the bounds check.** Quote `ionic_array_get` and show where the
`index out of bounds` message is produced.

**07.4 — Trace the RAII.** In `src/codegen/native.ionic`, list the four sites
the `model` ownership logic fires (function end, reassignment, return transfer,
explicit free). Quote one line from each.

**07.5 — Write the type-oracle plan.** In your own words, design the
type-tagging pass that would unblock RC: what information flows from
`checker.ionic` to codegen, and how `ncg_last_ty` gets set on the `ND_INDEX` arm.
Cross-check against `docs/memory-model.md §10`. This is capstone task 09.3.

---

## Checkpoint 07

- [ ] You can explain the three ownership mechanisms and the "cycles impossible" theorem.
- [ ] You can name the single reason RC was reverted and why it's a soundness issue.
- [ ] You can point to the four sites of `model` RAII in the codegen.
- [ ] You can articulate a concrete plan for the type-tagging prerequisite.

## [Next](08-build-bootstrap-selfhosting.md)
