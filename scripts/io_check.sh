#!/bin/sh
# File I/O conformance gate.
#
# The three defects this pins down were all invisible from the outside:
#
#   file_write / file_write_binary returned a status, but the checker declared
#     them TY_VOID. So `if (file_write(p, s) == 0)` was a compile error and the
#     result could only be ignored — a program whose write failed exited 0
#     having silently done nothing. The runtime was right; the declaration was
#     not.
#
#   file_exists existed only in the legacy LLVM backends. On the self-hosted
#     path it was "unknown function", so docs/language.md told readers to probe
#     with `file_size(path) >= 0` instead.
#
#   the two backends disagreed about the same call. The Rust backend emitted
#     `access(path, F_OK)` — which is true for a directory — and returned a
#     constant 1 from file_write on every path, including the one where fopen
#     failed.
#
# The fix is not just "make it compile": a status that cannot be observed and a
# predicate that means something different on each backend are the kind of bug
# that reads as working code. So each probe below runs a real program and its
# output is compared against the `// Expected:` lines in its own header. Adding
# a case means adding a file; nothing here needs editing.
#
# Why output comparison rather than exit codes: the failure being guarded
# against is a *wrong answer that exits 0*. A gate that only checked "ran
# without crashing" would pass against every version of this bug.
#
# The probes share a prefix in /tmp and the harness clears it first, so a
# previous run cannot leave a "missing" file behind for a later one to find.
#
# Usage:
#   scripts/io_check.sh              # gate on the native (self-hosted) backend
#   BACKEND=rust scripts/io_check.sh # gate on the Rust bootstrap, same probes
#
# BACKEND=rust runs the same probe set through target/release/ionic. Both
# backends must produce identical output: that is the cross-backend assertion
# the file_exists defect needed and did not have.
set -u
case "$0" in
  */*) cd "$(dirname "$0")/.." || exit 1 ;;
  *)   cd . || exit 1 ;;
esac

BACKEND="${BACKEND:-native}"
PROBE_DIR=scripts/io_probes
WORK="${TMPDIR:-/tmp}/ionic_io_probe_build"

if [ "$BACKEND" = "rust" ]; then
    IONIC="${IONIC:-./target/release/ionic}"
else
    IONIC="${IONIC:-./ionic_new}"
fi

if [ ! -x "$IONIC" ]; then
    echo "io_check: no compiler at '$IONIC'; set IONIC= to override" >&2
    exit 2
fi

rm -rf "$WORK"
mkdir -p "$WORK" || exit 2

echo "==> File I/O probes ($BACKEND backend)"

fail=0
for src in "$PROBE_DIR"/*.ionic; do
    name=$(basename "$src" .ionic)
    out="$WORK/$name"
    expected="$WORK/$name.expected"
    actual="$WORK/$name.actual"

    # The probes write under this prefix, and one of them asserts that a path
    # is *absent*. Leftovers from an earlier run would make that assertion fail
    # for a reason that has nothing to do with the compiler, so the prefix is
    # cleared per probe.
    rm -rf /tmp/ionic_io_probe_*

    if ! "$IONIC" "$src" -o "$out" > "$WORK/$name.compile.log" 2>&1; then
        echo "    ✗ FAIL: $name did not compile"
        sed 's/^/        /' "$WORK/$name.compile.log"
        fail=1
        continue
    fi

    if ! "$out" > "$actual" 2>&1; then
        echo "    ✗ FAIL: $name exited non-zero"
        sed 's/^/        /' "$actual"
        fail=1
        continue
    fi

    # The expected output is the probe's own header, so a case and its
    # assertion cannot drift apart into two files.
    sed -n 's|^// Expected: ||p' "$src" > "$expected"

    if [ ! -s "$expected" ]; then
        echo "    ✗ FAIL: $name declares no '// Expected:' lines"
        fail=1
        continue
    fi

    if ! diff -u "$expected" "$actual" > "$WORK/$name.diff"; then
        echo "    ✗ FAIL: $name output does not match its expectations"
        sed 's/^/        /' "$WORK/$name.diff"
        fail=1
        continue
    fi

    echo "    ✓ $name: $(tr '\n' ' ' < "$actual")"
done

rm -rf /tmp/ionic_io_probe_*

if [ "$fail" -ne 0 ]; then
    echo "    ✗ FAIL: file I/O does not behave as documented (see above)."
    exit 1
fi

echo "    ✓ PASS: status, existence and read semantics all match the documentation."
exit 0
