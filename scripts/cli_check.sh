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
    [0-9]*.*[0-9]*) ;;                             # 1.2.3
    [0-9]*.*[0-9]*-[0-9a-z]*) ;;                   # 1.0.0-beta.1
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
    for lit in $(sed -n 's/.*ionic compiler v\([0-9][0-9.a-z-]*\).*/\1/p' "$src"); do
        if [ "$lit" != "$got_version" ]; then
            echo "    ✗ FAIL: $src banner says '$lit' but --version says '$got_version'"
            fail=1
        fi
    done
done

# Every emitted IR file is stamped with the compiler version. That was a fourth,
# unchecked copy of the number until the 0.3.0 bump: it is written by codegen,
# not by the code that prints --version, so `--help`, `--version`, Cargo.toml
# and src/main.ionic could all agree while `.ll` files claimed something else.
#
# Checked against the *emitted file* rather than grepping the source, so it
# stays honest if codegen ever formats the banner differently — and it fails if
# the banner goes missing entirely, which a source grep would not.
#
# Only the Rust backend can emit IR; the native one skips this by design.
printf 'fn main() -> int64 { return 0; }\n' > "$WORK/ir_probe.ionic"
if "$IONIC" "$WORK/ir_probe.ionic" --emit-ir > "$WORK/ir_probe.ll" 2>/dev/null; then
    ir_version=$(sed -n 's/^; Generated by ionic v\([0-9][0-9.a-z-]*\)$/\1/p' "$WORK/ir_probe.ll")
    if [ -z "$ir_version" ]; then
        echo "    ✗ FAIL: emitted IR has no '; Generated by ionic v<version>' banner"
        fail=1
    elif [ "$ir_version" != "$got_version" ]; then
        echo "    ✗ FAIL: emitted IR banner says '$ir_version' but --version says '$got_version'"
        fail=1
    fi
fi

# ── the version must match the release tag ─────────────────────────────────
#
# This is the check that would have caught the state this file was written in:
# the tarball is named from the git tag (`release.yml` uses
# `${{ github.ref_name }}`), but the binary inside takes its version from the
# three sources above. Tag `v0.3.0` therefore shipped an `ionic` whose
# `--version` printed `0.1.0`, and nothing noticed, because every check
# compared the sources to each other and never to the tag.
#
# Only applies when HEAD is exactly a release tag. On a branch, or with local
# commits on top of a tag, there is no tag to agree with and the check is
# skipped — this is a release gate, not a branch gate.
#
# `CLI_REQUIRE_TAG=1` turns that skip into a failure. release.yml sets it,
# because a release job runs on a tag by definition: if the tag cannot be found
# there, the check has not run and the build must not proceed. Skipping is the
# right default in CI (a detached build has no tag) and the wrong one during a
# release, where a silent skip would reintroduce the drift this check exists to
# catch.
# A prerelease tag has to agree with a prerelease version, and a release tag with
# a release version. `v1.0.0-beta.1` and `1.0.0-beta.1` match; `v1.0.0` and
# `1.0.0-beta.1` do not. Comparing the strings as-is is exactly that rule, so it
# needs no special case -- but the sed above had to learn to read the suffix, or
# it would have compared `1.0.0` (truncated at `-`) against the tag and reported
# a mismatch that is not there.
tag=$(git describe --tags --exact-match 2>/dev/null || true)
if [ -n "$tag" ]; then
    tag_version=${tag#v}
    if [ "$got_version" != "$tag_version" ]; then
        echo "    ✗ FAIL: release tag is '$tag' but --version says '$got_version'"
        echo "        release.yml names the tarball from the tag, so this would"
        echo "        ship a binary that misreports its own version."
        fail=1
    fi
elif [ "${CLI_REQUIRE_TAG:-0}" = "1" ]; then
    echo "    ✗ FAIL: CLI_REQUIRE_TAG=1 but HEAD is not exactly a tag."
    echo "        Either the tag was not fetched, or this is running off-tag."
    echo "        A release build must not proceed without this check."
    fail=1
fi

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
echo "    ✓ --version ($BACKEND): $got_version, matching Cargo.toml and the source"
if [ -n "${tag:-}" ]; then
    echo "    ✓ release tag ($tag) agrees with --version"
fi
echo "    ✓ PASS: help, version and unknown-flag handling are correct."
exit 0
