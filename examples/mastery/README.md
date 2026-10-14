# Mastery Exercise Programs

Runnable references for the exercises in `docs/mastery/`. Each one compiles with
the self-hosted compiler and prints labelled, deterministic output.

```sh
# from the repo root
./ionic_new examples/mastery/00_hello.ionic -o /tmp/x && /tmp/x; echo "exit=$?"
```

## Runnable references

- `00_hello.ionic` — `main` + exit code (ex. 00.1)
- `00_toplevel.ionic` — no `main`, top-level statements (ex. 00.2)
- `00_discrepancies.ionic` — known bugs, reproduced (ex. 00.4)
- `01_literals.ionic` — one value of each type (ex. 01.1)
- `01_ranges.ionic` — exclusive range (ex. 01.2)
- `01_strings.ionic` — string builtins + `format` (ex. 01.1)
- `01_print_variants.ionic` — `print` vs `println` overloads (units 01/03)
- `02_arrays.ionic` — `.len`, `len()`, indexing, element set (ex. 02.2)
- `02_data_bug.ionic` — the `std.data` off-by-one (ex. 02.3)
- `02_structs.ionic` — struct fields mutable under `let` (ex. 02.4)
- `03_escapes.ionic` — `\e`, `\xHH`, `char_to_str` (ex. 03.2)
- `03_args.ionic` — CLI argument + exit code (ex. 03.3)
- `03_file_io.ionic` — `file_write`/`file_read` round trip (ex. 03.4)
- `04_placement_types.ionic` — `@cpu`/`@gpu` functions that compile (ex. 04.4)
- `05_selective.ionic` — wildcard import, selective at codegen (ex. 05.1)
- `05_rel/` — relative import demo (ex. 05.3); compile `main.ionic`
- `07_leak.ionic` — string churn, unbounded RSS (ex. 07.1)

## Intentionally-failing programs (`fail/`)

These are meant to be **rejected** (or, for the string-array one, to misbehave).
They are references for the error-reproduction exercises; they are not part of
any build.

- `fail/immutable.ionic` — assign to a `let` (ex. 01.5)
- `fail/multi_error.ionic` — several errors in one pass (unit 03 §7)
- `fail/string_array_print.ionic` — compiles+runs but prints a pointer (ex. 02.5)
- `fail/tensor_mismatch.ionic` — tensor@gpu where tensor@cpu expected (ex. 04.1)
- `fail/gpu_cpu_io.ionic` — contradicting `@gpu fn (@cpu …)` annotation (ex. 04.3)
- `fail/resolution_miss.ionic` — unresolvable `import std.nosuch` (ex. 05.5)
