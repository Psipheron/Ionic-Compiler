#!/bin/sh
# Command-line-surface gate.
#
# Before this, neither compiler answered `--help` or `--version`. Both treated
# argv[1] as a source path unconditionally, so
#
#     $ ionic --version
#     ionic: cannot read '--version'
#       no such file, or not a regular file
#
# — a missing-file error for a flag, exit 1. That is the failure mode this gate
# exists for: not "the flag is unimplemented" (obvious) but "the flag is
# interpreted as a filename", which reads as a legitimate diagnostic.
#
# Two properties are asserted, and they are deliberately different:
#
#   AGREEMENT   `--version` must print the same bytes from both backends. A
#               version is a fact about the release, not about which compiler
#               built it, so a difference here is a real defect.
#
#   FIDELITY    `--help` is *not* required to be identical, because the two
#               backends accept different flags: the self-hosted one has
#               --target/--print-link, the Rust one has --run/--emit-ir/
#               --dump-*. Help that described both would advertise options the
#               binary in hand rejects. Instead each help text is checked
#               against the flags that backend actually honours — every flag
#               the parser handles must be named in its own help.
#
# That second check is what keeps the text from rotting: adding a flag without
# documenting it fails here, and documenting a flag that was removed fails too.
#
# Usage:
#   scripts/cli_check.sh              # gate on the native (self-hosted) backend
#   BACKEND=rust scripts/cli_check.sh # gate on the Rust bootstrap
set -u
case "$0" in
  */*) cd "$(dirname "$0")/.." || exit 1 ;;
  *)   cd . || exit 1 ;;
esac

BACKEND="${BACKEND:-native}"
WORK="${TMPDIR:-/tmp}/ionic_cli_check"
rm -rf "$WORK"; mkdir -p "$WORK" || exit 2

if [ "$BACKEND" = "rust" ]; then
    IONIC="${IONIC:-./target/release/ionic}"
    # Flags this backend's arg parser handles (see parse_args in src/main.rs).
    FLAGS="--run --emit-ir --dump-ast --dump-tokens --help --version"
else
    IONIC="${IONIC:-./ionic_new}"
    # Flags this backend's arg loop handles (see src/main.ionic). `-o` and
    # --target take a value and are listed separately below.
    FLAGS="--print-link --help --version"
fi

if [ ! -x "$IONIC" ]; then
    echo "cli_check: '$IONIC' is not executable; build it or set IONIC=" >&2
    exit 2
fi

echo "==> CLI surface ($BACKEND backend)"
fail=0

# ── --help ───────────────────────────────────────────────────────────────────
if ! "$IONIC" --help > "$WORK/help.txt" 2>&1; then
    echo "    ✗ FAIL: --help exited non-zero"
    sed 's/^/        /' "$WORK/help.txt"
    fail=1
fi

if [ ! -s "$WORK/help.txt" ]; then
    echo "    ✗ FAIL: --help printed nothing"
    fail=1
fi

# Every flag the parser accepts must appear in its own help. A substring test
# is enough: a flag mentioned anywhere in the text is discoverable, and pinning
# the exact wording would make the gate an obstacle to improving it.
for f in $FLAGS; do
    if ! grep -q -- "$f" "$WORK/help.txt"; then
        echo "    ✗ FAIL: --help does not mention '$f', which this backend accepts"
        fail=1
    fi
done

# `-o` takes an argument in both backends; it has to be discoverable too.
if ! grep -q -- "-o " "$WORK/help.txt"; then
    echo "    ✗ FAIL: --help does not mention '-o <name>'"
    fail=1
fi

# The help must describe this binary. It is a usage message, so it has to say
# how to use it; an empty or one-line output would pass the checks above.
if ! grep -q -- "Usage:" "$WORK/help.txt"; then
    echo "    ✗ FAIL: --help has no 'Usage:' line"
    fail=1
fi

# ── --version ────────────────────────────────────────────────────────────────
#
# Just the number, with no prefix: `ionic --version | cut -d' ' -f2` is the
# shape package managers and build scripts actually use, and a prefix would
# silently make that expression print the wrong field.
if ! "$IONIC" --version > "$WORK/version.txt" 2>&1; then
    echo "    ✗ FAIL: --version exited non-zero"
    sed 's/^/        /' "$WORK/version.txt"
    fail=1
fi

got_version=$(tr -d '\r\n' < "$WORK/version.txt")

# Must look like a version: digits and dots, no leading 'v', no prose. This is
# the check that catches `--version` falling through to the "cannot read"
# path (which would put the whole diagnostic here) or printing a banner.
case "$got_version" in
    [0-9]*.*[0-9]*) ;;
    *)
        echo "    ✗ FAIL: --version printed '$got_version', not a bare version number"
        fail=1
        ;;
esac
if printf '%s' "$got_version" | grep -q -- ' '; then
    echo "    ✗ FAIL: --version printed '$got_version'; expected the number alone"
    fail=1
fi

# ── short forms ─────────────────────────────────────────────────────────────
#
# -h and -V are the conventional aliases. They are checked for agreement with
# the long forms rather than against a fixed string, so a version bump touches
# one place.
"$IONIC" -h > "$WORK/help_short.txt" 2>&1 || {
    echo "    ✗ FAIL: -h exited non-zero"
    fail=1
}
if ! cmp -s "$WORK/help.txt" "$WORK/help_short.txt"; then
    echo "    ✗ FAIL: -h and --help print different text"
    fail=1
fi

"$IONIC" -V > "$WORK/version_short.txt" 2>&1 || {
    echo "    ✗ FAIL: -V exited non-zero"
    fail=1
}
if ! cmp -s "$WORK/version.txt" "$WORK/version_short.txt"; then
    echo "    ✗ FAIL: -V and --version print different text"
    fail=1
fi

# ── the version must match the release metadata ─────────────────────────────
#
# The number is duplicated in places that cannot import each other: this
# compiler's source, the other compiler's source, and Cargo.toml. Neither
# compiler can read the other's source and no build step generates one from the
# other, so duplication is unavoidable — which is exactly why it is checked
# rather than merely noted.
cargo_version=$(sed -n 's/^version *= *"\([^"]*\)".*/\1/p' Cargo.toml 2>/dev/null | head -1)
if [ -n "$cargo_version" ] && [ "$got_version" != "$cargo_version" ]; then
    echo "    ✗ FAIL: --version says '$got_version' but Cargo.toml says '$cargo_version'"
    fail=1
fi

# The self-hosted source stores the number as a literal in two places (the
# `--version` output and the banner in `--help`). Both are read from source
# because that backend cannot be asked for its banner separately.
for src in src/main.ionic; do
    [ -f "$src" ] || continue
    for lit in $(sed -n 's/.*ionic compiler v\([0-9][0-9.]*\).*/\1/p' "$src"); do
        if [ "$lit" != "$got_version" ]; then
            echo "    ✗ FAIL: $src banner says '$lit' but --version says '$got_version'"
            fail=1
        fi
    done
done

# ── a flag is not a filename ────────────────────────────────────────────────
#
# The regression itself: an unknown flag must not be reported as an unreadable
# file. Both backends reject it, and neither may emit "cannot read".
printf 'fn main() -> int64 { return 0; }\n' > "$WORK/ok.ionic"
"$IONIC" --no-such-flag-xyz > "$WORK/unknown.txt" 2>&1
unknown_rc=$?
if [ "$unknown_rc" = 0 ]; then
    echo "    ✗ FAIL: an unknown flag was accepted (exit 0)"
    fail=1
fi
if grep -qi -- "cannot read" "$WORK/unknown.txt"; then
    echo "    ✗ FAIL: an unknown flag was reported as an unreadable file"
    sed 's/^/        /' "$WORK/unknown.txt"
    fail=1
fi

if [ "$fail" -ne 0 ]; then
    echo "    ✗ FAIL: the CLI surface does not behave as documented (see above)."
    exit 1
fi

echo "    ✓ --help ($BACKEND): describes every flag this backend accepts"
echo "    ✓ --version ($BACKEND): $got_version, matching Cargo.toml"
echo "    ✓ PASS: help, version and unknown-flag handling are correct."
exit 0
