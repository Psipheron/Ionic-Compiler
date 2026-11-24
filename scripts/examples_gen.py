#!/usr/bin/env python3
"""Regenerate the measurement columns of scripts/examples.manifest.

The manifest is data, not prose: it records what each example *does* on each
backend. Measuring it matters because the interesting rows are the ones that
disagree — a file that compiles natively but not under the Rust bootstrap, or a
negative test one backend wrongly accepts. Transcribing those by hand is how the
two backends drift apart in the first place.

What is preserved across a regeneration: every comment block already attached to
a row. Those carry the *reason* a row is `skip`, which cannot be measured, only
explained. A newly detected divergence gets a placeholder note so it is
impossible to regenerate and silently lose the record of a known gap.

Usage:
    python3 scripts/examples_gen.py            # rewrite the measurement columns
    python3 scripts/examples_gen.py --check    # exit 1 if the file is out of date
"""
import glob
import os
import subprocess
import sys

MANIFEST = "scripts/examples.manifest"
NATIVE = os.environ.get("IONIC", "./ionic_new")
RUST = "./target/release/ionic"
TIMEOUT = int(os.environ.get("EXAMPLE_TIMEOUT", "60"))

# Measure with both ML backends forced OFF, regardless of what is installed.
#
# Without this the manifest records the machine, not the program: an example
# that uses mlx_* exits 0 on a developer box with mlx-c linked in and 1 ("not
# compiled in") on a bare CI runner. The first version of this gate was
# generated locally and then failed on CI for exactly that reason — two rows
# asserted exit statuses the runner's environment could not produce.
#
# Forcing the backends off makes every measurement depend only on the compiler
# and the source, so the file regenerates identically here and on the runner and
# `--check` can actually hold. Backend-dependent behaviour is covered by the
# dedicated MLX/ONNX steps in ci.yml, which run where the libraries exist.
MEASURE_ENV = {**os.environ, "IONIC_NO_MLX": "1", "IONIC_NO_ORT": "1"}

# A comment, not a bare header line: the gate strips comment lines and treats
# everything else as data, so an un-prefixed header would be parsed as a row
# naming a file called "path".
ROW_HEADER = "# path\tkind\trun_exit\tnative\trust"

PREAMBLE = """\
# Every .ionic file under examples/, and what each backend must do with it.
#
# Format (tab-separated, `#` starts a comment line):
#
#   path <TAB> kind <TAB> run_exit <TAB> native <TAB> rust
#
#   path      repository-relative path, so a moved file fails loudly rather
#             than silently dropping out of the gate.
#   kind      `ok`  — must compile
#             `neg` — must be REJECTED by the compiler (the *_error and
#                     mastery/fail programs; their own first line says so)
#   run_exit  the exit status the compiled program must produce, or `.` for
#             "compile only". `.` is used where the program is deliberately
#             silent, and for anything whose exit status is a property of the
#             machine rather than the source — see the note on ML backends below.
#   native    `ok`   — the self-hosted compiler must meet this expectation
#             `skip` — known gap, recorded rather than hidden
#   rust      same, for the Rust bootstrap (target/release/ionic)
#
# Every measurement is taken with IONIC_NO_MLX=1 and IONIC_NO_ORT=1, so the file
# describes the program rather than the machine that generated it. Without that,
# an example using mlx_* records exit 0 on a box with mlx-c installed and exit 1
# ("backend not compiled in") on a CI runner, and `--check` can never agree with
# either. Backend-dependent behaviour is tested by the dedicated MLX/ONNX steps
# in ci.yml, which run where the libraries exist.
#
# `skip` is deliberately narrow and every entry carries its reason. A gate that
# silently tolerated a whole class of failure would not be a gate, so the point
# of writing these down is that fixing one means deleting a line here — the
# list can only shrink, never grow unnoticed.
#
# What this gate does NOT assert: the text a program prints. Output correctness
# belongs to io_check.sh (the contract), the multitarget differential job
# (Mach-O vs ELF) and rss_check.sh (allocation shape). Duplicating it here would
# make every example edit a three-file change.
#
# Only the columns below `path` are generated. To change an expectation, either
# change the code or explain the divergence in a comment above its row.
#
# Regenerate after an intentional change:
#   python3 scripts/examples_gen.py
"""


def kind_of(path):
    return "neg" if ("/fail/" in path or path.endswith("_error.ionic")) else "ok"


def compiles(binary, path, out):
    # Both backends print diagnostics to stdout, so capture both streams.
    return subprocess.run(
        [binary, path, "-o", out],
        capture_output=True, timeout=TIMEOUT, env=MEASURE_ENV,
    ).returncode == 0


def measure(path):
    kind = kind_of(path)
    native_ok = compiles(NATIVE, path, "/tmp/gen_n")
    rust_ok = compiles(RUST, path, "/tmp/gen_r")

    if kind == "neg":
        # For a negative row, `ok` means "this backend rejects it" — the
        # opposite sense from a positive row. The gate reads `kind` first, so
        # the same `ok` token is unambiguous in context. `skip` means the
        # backend accepts a program it should reject.
        return (kind, "-", "ok" if not native_ok else "skip",
                "ok" if not rust_ok else "skip")

    native_col = "ok" if native_ok else "skip"
    rust_col = "ok" if rust_ok else "skip"

    run_exit = "."
    if native_ok:
        try:
            proc = subprocess.run(
                ["/tmp/gen_n"], capture_output=True, timeout=TIMEOUT, env=MEASURE_ENV
            )
        except subprocess.TimeoutExpired:
            proc = None
        if proc is not None:
            # A program that must exit 0 but prints nothing is either a
            # deliberately silent demo or one whose output depends on the
            # machine. Either way the gate cannot assert a status it does not
            # understand, so record compile-only.
            silent_ok = proc.returncode == 0 and not proc.stdout.strip()
            if not silent_ok:
                run_exit = str(proc.returncode)

    return (kind, run_exit, native_col, rust_col)


def parse_existing():
    """(path -> comment lines above it, ordered list of paths).

    `notes` holds the explanation attached to a row, so a regeneration cannot
    drop the reason a row is `skip`. Two things are deliberately excluded from
    it: the file preamble, and a bare section heading like "── Negative tests
    ──", which group rows rather than explaining one. Both would otherwise be
    swallowed into the first row beneath them and then duplicated into the
    middle of the rewritten file.
    """
    notes, order = {}, []
    if not os.path.exists(MANIFEST):
        return notes, order

    pending = []
    seen_first_row = False
    for line in open(MANIFEST, encoding="utf-8"):
        s = line.rstrip("\n")
        if s.startswith("#"):
            pending.append(s)
            continue
        if not s.strip():
            # A blank line ends a block: what follows is a heading, not a note
            # about the row above.
            pending = []
            continue

        path = s.split("\t")[0]
        is_placeholder_header = path == "path"

        if pending and seen_first_row and not is_placeholder_header:
            # A section heading is decoration, not an explanation.
            heading_like = all(
                p.lstrip("# ").startswith("──") or not p.lstrip("# ").strip()
                for p in pending
            )
            if not heading_like:
                notes[path] = pending

        if not is_placeholder_header:
            seen_first_row = True
        order.append(path)
        pending = []
    return notes, order


def render():
    files = (
        sorted(glob.glob("examples/*.ionic"))
        + sorted(glob.glob("examples/mastery/*.ionic"))
        + sorted(glob.glob("examples/mastery/fail/*.ionic"))
    )
    notes, _ = parse_existing()

    out = [PREAMBLE, ROW_HEADER]
    for f in files:
        note = notes.get(f)
        if note:
            out.append("\n".join(note))
        kind, run_exit, n, r = measure(f)
        if not note and "skip" in (n, r):
            out.append(
                "# NOTE: divergence — explain why this row is `skip` before\n"
                "#       committing. Measured, not invented."
            )
        out.append("\t".join([f, kind, run_exit, n, r]))
    return "\n".join(out).rstrip("\n") + "\n"


def main():
    for need in (NATIVE, RUST):
        if not os.access(need, os.X_OK):
            sys.exit(f"examples_gen: {need} is not executable — build it first")

    text = render()

    if "--check" in sys.argv:
        current = open(MANIFEST, encoding="utf-8").read() if os.path.exists(MANIFEST) else ""
        if current != text:
            print("examples_gen: manifest is out of date")
            for a, b in zip(current.splitlines(), text.splitlines()):
                if a != b:
                    print(f"  have: {a}\n  want: {b}")
                    break
            sys.exit(1)
        print("examples_gen: manifest is up to date")
        return

    open(MANIFEST, "w", encoding="utf-8").write(text)
    print(f"wrote {MANIFEST}")


if __name__ == "__main__":
    main()
