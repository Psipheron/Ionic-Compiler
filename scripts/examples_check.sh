#!/bin/sh
# Every example must compile, run, or fail for the reason it was written to.
#
# Why this gate exists
# --------------------
# `examples/` is the surface a new user meets first, and it is the part of the
# tree with no owner. Before this, 24 of 35 example files were referenced by no
# CI job at all, so an example could rot silently: a language change would break
# a demo and nothing would notice until someone tried to teach from it.
#
# It also covers the cross-backend case, which is the one that actually bit:
# `examples/mlx_matmul.ionic` uses `c += 1`, and the Rust bootstrap has no
# compound assignment, so the same file compiles on the self-hosted compiler and
# fails to parse on the bootstrap. Nothing compared the two.
#
# What is asserted
# ----------------
# For each row in scripts/examples.manifest:
#
#   ok   compile must succeed; if run_exit is a number, the program must run
#        and exit with exactly that status
#   neg  compile must FAIL. A negative test that starts compiling is a
#        regression in the checker, not a fixed example.
#
# Both backends are checked against their own column, because they legitimately
# differ today (see the `skip` rows and their reasons).
#
# What is NOT asserted: printed text. That belongs to io_check.sh (the I/O
# contract), the multitarget differential job and rss_check.sh. Duplicating it
# here would make every example edit a three-file change.
#
# The manifest is also checked for *coverage*: every .ionic file under
# examples/ must have a row, and every row must point at a file that exists.
# That is the half that stops the gate from quietly shrinking — deleting an
# example without deleting its row fails, and adding one without a row fails.
#
# Usage:
#   scripts/examples_check.sh                 # both backends
#   BACKEND=native scripts/examples_check.sh   # self-hosted only (no cargo needed)
#   BACKEND=rust   scripts/examples_check.sh   # Rust bootstrap only
#   IONIC=./some/binary scripts/examples_check.sh
set -u
# Resolve through symlinks: a script linked into ~/bin or /usr/local/bin would
# otherwise `cd` to the link's directory and fail to find the repo. Other gates
# in scripts/ have the same limitation, but this one is worth getting right
# because it is the kind of thing people wire into an editor's save hook.
self="$0"
while [ -L "$self" ]; do
    target=$(readlink "$self")
    case "$target" in
      /*) self="$target" ;;
      *)  self="$(dirname "$self")/$target" ;;
    esac
done
case "$self" in
  */*) cd "$(dirname "$self")/.." || exit 1 ;;
  *)   cd . || exit 1 ;;
esac

MANIFEST="scripts/examples.manifest"
BACKEND="${BACKEND:-both}"
WORK="${TMPDIR:-/tmp}/ionic_examples_check"

# Per-program timeout. The slowest example runs in well under a second; this is
# only here so a regression that turns one into an infinite loop fails the gate
# instead of hanging CI until the job's own timeout.
LIMIT="${EXAMPLE_TIMEOUT:-60}"

# Measure with both ML backends forced OFF, matching scripts/examples_gen.py.
#
# The manifest records what the *program* does, not what this machine happens to
# have installed. An example that uses mlx_* exits 0 with mlx-c linked in and 1
# ("not compiled in") without it, so measuring on whatever box is running turns
# a portable expectation into a machine-specific one. The first version of this
# gate was generated on a developer machine with MLX and then failed on CI for
# exactly that reason. Backend behaviour is covered by the dedicated MLX/ONNX
# steps in ci.yml, which run where the libraries actually exist.
export IONIC_NO_MLX=1
export IONIC_NO_ORT=1

rm -rf "$WORK"; mkdir -p "$WORK" || exit 2

[ -f "$MANIFEST" ] || { echo "examples_check: missing $MANIFEST" >&2; exit 2; }

# Both backends emit coloured diagnostics. Escaped into a log they are noise, so
# strip the CSI sequences before echoing anything a human has to read. `perl` is
# already a build dependency (build.sh uses it for timeouts); `sed` alone cannot
# match an unterminated escape portably.
if command -v perl >/dev/null 2>&1; then
    strip_ansi() { perl -pe 's/\e\[[0-9;]*[A-Za-z]//g'; }
else
    strip_ansi() { cat; }
fi

# ── Resolve which backends to run ────────────────────────────────────────────
run_native=0
run_rust=0
case "$BACKEND" in
  both)   run_native=1; run_rust=1 ;;
  native) run_native=1 ;;
  rust)   run_rust=1 ;;
  *) echo "examples_check: BACKEND must be both|native|rust" >&2; exit 2 ;;
esac

NATIVE="${IONIC:-./ionic_new}"
RUST="./target/release/ionic"

# A backend that was asked for but is absent is a hard error, not a skip:
# silently testing one backend while believing you tested two is the failure
# this gate exists to prevent.
check_backend_present() {
    name="$1"; bin="$2"
    if [ ! -x "$bin" ]; then
        if [ -n "${IONIC:-}" ] && [ "$name" = "native" ]; then
            echo "examples_check: IONIC='$bin' is not executable" >&2
            exit 2
        fi
        echo "examples_check: $name backend '$bin' is not executable." >&2
        echo "  native: run ./build.sh          rust: run cargo build --release" >&2
        exit 2
    fi
}
[ "$run_native" = 1 ] && check_backend_present native "$NATIVE"
[ "$run_rust"   = 1 ] && check_backend_present rust   "$RUST"

# ── Parse the manifest ───────────────────────────────────────────────────────
# Read into a temp file of clean rows so the loop can use plain `read`.
grep -v '^[[:space:]]*#' "$MANIFEST" | grep -v '^[[:space:]]*$' > "$WORK/rows"

if [ ! -s "$WORK/rows" ]; then
    echo "examples_check: $MANIFEST has no rows" >&2
    exit 2
fi

echo "==> Examples: compile, run and reject across backends"
echo "    (ML backends forced off: the manifest describes the program, not this machine)"

fail=0
n_ok=0
n_neg=0
n_run=0
n_skip=0

# ── Coverage: manifest ↔ filesystem, both directions ─────────────────────────
for f in examples/*.ionic examples/mastery/*.ionic examples/mastery/fail/*.ionic; do
    [ -f "$f" ] || continue
    if ! cut -f1 "$WORK/rows" | grep -qx "$f"; then
        echo "    ✗ FAIL: $f exists but has no row in $MANIFEST"
        echo "        A new example must be declared, or the gate does not cover it."
        fail=1
    fi
done

while IFS='	' read -r path kind run_exit native_ok rust_ok; do
    if [ ! -f "$path" ]; then
        echo "    ✗ FAIL: $MANIFEST lists '$path', which does not exist"
        echo "        Deleting an example must delete its row too."
        fail=1
        continue
    fi

    # ── native backend ───────────────────────────────────────────────────────
    if [ "$run_native" = 1 ]; then
        if [ "$native_ok" = "skip" ]; then
            n_skip=$((n_skip + 1))
        else
            # Diagnostics go to *stdout* from both backends (they print
            # coloured errors and exit), so both streams must be captured —
            # redirecting only stderr let every negative test's error text
            # scroll past as if it were gate output.
            out="$WORK/native.err"
            if "$NATIVE" "$path" -o "$WORK/prog_n" >"$out" 2>&1; then
                compiled=1
            else
                compiled=0
            fi

            if [ "$kind" = "neg" ]; then
                n_neg=$((n_neg + 1))
                if [ "$compiled" = 1 ]; then
                    echo "    ✗ FAIL: $path compiled, but it is a negative test"
                    echo "        The checker stopped rejecting it — that is a regression."
                    fail=1
                fi
                continue
            fi

            if [ "$compiled" = 0 ]; then
                echo "    ✗ FAIL: $path does not compile (native backend)"
                strip_ansi < "$out" | head -8 | sed 's/^/        /'
                fail=1
                continue
            fi

            if [ "$run_exit" != "." ]; then
                started=$(date +%s)
                "$WORK/prog_n" >"$WORK/native.out" 2>&1
                got=$?
                elapsed=$(( $(date +%s) - started ))
                if [ "$elapsed" -ge "$LIMIT" ]; then
                    echo "    ✗ FAIL: $path did not finish within ${LIMIT}s"
                    fail=1
                    continue
                fi
                n_run=$((n_run + 1))
                if [ "$got" != "$run_exit" ]; then
                    echo "    ✗ FAIL: $path exited $got, expected $run_exit"
                    strip_ansi < "$WORK/native.out" | head -6 | sed 's/^/        /'
                    fail=1
                    continue
                fi
                # A program that must exit 0 with no output at all is almost
                # certainly producing nothing where it used to print something.
                if [ "$run_exit" = "0" ] && [ ! -s "$WORK/native.out" ]; then
                    echo "    ✗ FAIL: $path printed nothing, and is expected to"
                    fail=1
                    continue
                fi
            fi
            n_ok=$((n_ok + 1))
        fi
    fi

    # ── Rust bootstrap ───────────────────────────────────────────────────────
    if [ "$run_rust" = 1 ]; then
        if [ "$rust_ok" = "skip" ]; then
            n_skip=$((n_skip + 1))
        else
            out="$WORK/rust.err"
            if "$RUST" "$path" -o "$WORK/prog_r" >"$out" 2>&1; then
                compiled=1
            else
                compiled=0
            fi

            if [ "$kind" = "neg" ]; then
                if [ "$compiled" = 1 ]; then
                    echo "    ✗ FAIL: $path compiled under the Rust bootstrap, but is a negative test"
                    fail=1
                fi
            elif [ "$compiled" = 0 ]; then
                # The bootstrap is expected to parse the whole language; a file
                # it cannot compile is a real gap, so say which one.
                echo "    ✗ FAIL: $path does not compile (Rust bootstrap)"
                strip_ansi < "$out" \
                    | grep -vE '^ld: warning|^warning:|^ *[0-9]+ warning' \
                    | head -6 | sed 's/^/        /'
                fail=1
            fi
        fi
    fi
done < "$WORK/rows"

if [ "$fail" -ne 0 ]; then
    echo "    ✗ FAIL: the example corpus does not behave as declared (see above)."
    exit 1
fi

detail=""
[ "$n_ok"   -gt 0 ] && detail="$detail$n_ok compiled"
[ "$n_run"  -gt 0 ] && detail="$detail, $n_run ran and exited as declared"
[ "$n_neg"  -gt 0 ] && detail="$detail, $n_neg rejected as intended"
[ "$n_skip" -gt 0 ] && detail="$detail, $n_skip known gap(s)"
echo "    ✓${detail}"
echo "    ✓ PASS: every example compiles, runs or fails as $MANIFEST declares."
exit 0
