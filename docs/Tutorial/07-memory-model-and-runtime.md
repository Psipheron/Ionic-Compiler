# 07 — Memory Model & Runtime

**Goal:** understand exactly how Ionic allocates and (doesn't) free memory today,
why the design chose deterministic ownership, the shipped `model` RAII slice, and
the allocation audit and type oracle that gate the reference-counting slice. This
unit is the heart of becoming a compiler *owner* rather than a compiler *user*.

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
  (`src/ionic_model_runtime.c:77`) with a separate `malloc`/`realloc`'d data region.
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
  `piper_forward`, `write_wav`.

`ionic_array_get` (`:554`) is the bounds check behind `ionic: index out of bounds`.

---

## 3. The decision: deterministic ownership

`docs/memory-model.md §3` adopts three mechanisms, all deterministic and local
(nothing walks the heap, nothing pauses):

1. **Non-atomic reference counting** for shared immutable data — strings, array
   data, tensor storage.
2. **Unique ownership + RAII** for scarce, device-affine resources — `model`,
   GPU buffers, file/socket handles.
3. **Arena scopes** for compiler-internal temporaries (one per phase).

And the key claim (§6): **cycles are meant to be structurally impossible**,
because shared data is immutable and mutable ownership would be unique/affine.
If that held, every reference graph would be a DAG and refcounting alone would
suffice — no cycle collector, ever.

It does not hold. `let q = p` copies a pointer rather than moving it (for both
structs and arrays), so two names alias one mutable aggregate, and `a.push(b)`
lets an aggregate contain an aggregate. A cycle is therefore constructible from
ordinary Ionic (`docs/memory-model.md §11.6`). Keep the *intent* in mind, but do
not rely on the theorem.

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

## 5. The blocker — and what has been done about it

The RC slice (`docs/memory-model.md §10`) was implemented, measured working, and
**reverted**. The churn loop dropped from **160 MB to 1.6 MB** peak RSS — then it
was backed out because:

> the codegen cannot reliably classify a value as a managed string.

`ionic_rc_release` was firing on non-managed pointers (stack/rodata addresses).
The insertion logic gates on "is this expression a string?", and that predicate
was not knowable from the backend state at the time:

1. `local_find` returned the *last* slot for a name (shadowing) while
   `local_get_str` returned the *first* — a shadowed integer could be treated as
   a string.
2. `ncg_last_ty` is set only on some paths — `ND_LET` with a non-literal string
   initializer, and array element **loads**, never set it.
3. Parameters were never typed at all: they do not enter the checker's symbol
   table, so `fn show(string s) { println(s); }` printed a pointer.

A single mis-placed `release` turns a benign leak into a use-after-free — worse
than the leak. So the requirement is soundness, not coverage.

### 5.1 What shipped (the first half of the prerequisite)

`ncg_ty_of(n)` / `ncg_elem_ty(n)` in `src/codegen/native.ionic` is a **pure**
type oracle over the AST plus the symbol tables. It answers in its own
vocabulary (`OTY_INT`/`OTY_STR`/`OTY_FLOAT`/`OTY_OTHER`), emits nothing, mutates
nothing, and is what `print`/`println` dispatch now consults. Supporting it:

- `local_get_str`'s scan direction is now irrelevant to printing, because print
  dispatch no longer reads it — it asks the oracle.
- A `local_elem` column records the element type of an array-typed local and
  parameter, so an element load no longer loses its type.
- Parameters record their declared `string`/array element type when they are
  defined, closing case 3 above.

This fixed the two observable symptoms (`println(names[0])` and
`println(s)` for a parameter). Regression reference:
`examples/mastery/02_string_array_print.ionic`.

### 5.2 The real blocker: an allocation audit, not a type question

The oracle is deliberately conservative — it answers `OTY_OTHER` ("I cannot
prove it") for:

- a **nested-array element** (`[[string]][i][j]`), because the oracle's vocabulary
  has no code for an array-valued element;
- a **method call** result, because `fn_ret_ty` does not describe methods;
- anything reached through a **struct field** load.

`OTY_OTHER` must be treated as "unknown", never as "int64". A refcount pass may
emit a `retain`/`release` only where the oracle returns `OTY_STR`; everywhere else
it must fall back to the current no-op behaviour.

**But that is not sufficient, and assuming it was is what broke the first
attempt.** A refcount word at `p - 8` only works if `p` is the *start* of a
`rc_alloc` payload. `docs/memory-model.md §11` audits all 23 string-returning
runtime entry points against that requirement. Two findings came out of it:

- **Alias returns (§11.1) — fixed.** `ionic_file_read` (failed open),
  `ionic_str_replace` / `ionic_http_body` (a NULL argument or a missing separator)
  and `ionic_http_get` / `ionic_http_post` (transport failure) returned either a
  rodata `""` or the caller's own input. Releasing one was a free of static data
  or a double free. Every return path now allocates, so that class is empty.
- **Interior pointers (§11.2) — fixed by copying.** `ionic_http_path` /
  `ionic_http_method` returned pointers into `HttpRequest`, whose `method[16]`
  and `path[1024]` are **inline arrays**. `p - 8` lands inside the struct: a
  release decremented the tail of `method`, and when the count reached zero
  `free()` took an interior address and aborted. `ionic_http_req_body` was worse
  — it returned a pointer the request also frees, so releasing it was a double
  free. All three now return a copy, so the value is ownable.

So the ordering is: audit (§11, done) → make every string return path allocate
(§11.1, done) → resolve the interior/static accessors (§11.2, done) → *then* the
retain/release pass. Nothing in the audit blocks it now.

The lesson worth carrying: no amount of type-oracle work fixes §11.2, because
the oracle is **right** — the value really is a string — and it is still not
ownable. Ownership is a property of the allocator, not of the type.

Also read §11.6: cycles **are** constructible (`let q = p` aliases; `a.push(b)`
nests aggregates), so refcounting reclaims acyclic graphs only.

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

**07.5 — Audit the oracle's blind spots.** `ncg_ty_of` already exists (§5.1).
Read it and list every node kind it answers `OTY_OTHER` for. For each, say
whether an RC pass could safely emit a `release` there, and what it would take to
answer the question precisely. Cross-check against `docs/memory-model.md §10`.

**07.6 — Audit the runtime's return paths.** Read
`docs/memory-model.md §11`, then repeat the exercise independently for one
function of each surviving class (`str_concat`, `http_path`, `getenv`,
`fgets_stdin`, `mlx_last_error`). Classify each as fresh / interior / foreign /
caller-buffer, and say what a `release` on it would do.

Then take a function from a class that is now *empty* — `file_read`,
`str_replace`, `http_body` (§11.1), or `http_path` / `mlx_weights_key_at`
(§11.2) — and check the *current* source to confirm every return path allocates
or copies. The exercise is to verify that for yourself rather than trust the
section: `http_path` is the interesting one, because the fix is a one-word
difference (`rt_dup_str(...)` instead of `r->path`) and nothing in the type
system would have caught the original. The original §11 conclusion took an hour
with a C probe; try to reach it yourself before reading the answers.

---

## Checkpoint 07

- [ ] You can explain the three ownership mechanisms.
- [ ] You can say whether cycles are actually impossible, and what makes that so.
- [ ] You can name the single reason RC was reverted and why it's a soundness issue.
- [ ] You can point to the four sites of `model` RAII in the codegen.
- [ ] You can name three queries `ncg_ty_of` still cannot answer, and say why
      "unknown" must not be treated as "not managed".
- [ ] You can explain why `ionic_http_path` *used to* be un-refcountable, why
      that was not a type-inference problem, and what fix was chosen.
- [ ] You can state the ownership contract a string-returning runtime function
      must satisfy, and name a function that used to violate it.
- [ ] You can say which of the four foreign/static sources still must never be
      instrumented, and why copying was not applied to them.

## [Next](08-build-bootstrap-selfhosting.md)
