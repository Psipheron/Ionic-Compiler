#!/usr/bin/env bash
# RSS regression gate.
#
# The language's stated priority is the smallest possible resident set. This
# script guards that property by compiling a string-churn benchmark at two
# sizes and checking that peak RSS does NOT grow with the iteration count.
#
# Why scaling, not an absolute number:
#   A leaking allocator grows RSS ~linearly with N. A correct memory model
#   reclaims each iteration's garbage as it dies, so peak RSS is flat in N.
#   The startup baseline (linked runtime, libc, page table) is a constant that
#   cancels when we subtract, making the gate independent of machine, malloc
#   implementation, and OS version -- unlike a raw byte budget, which would be
#   brittle across CI images.
#
# A fixed allocator yields delta ~= 0. The current leaky runtime yields
# delta ~= (bytes_per_iter * (N2 - N1)) ~= 48 * 600000 ~= 29 MB. The default
# 8 MB threshold sits an order of magnitude away from both, so it flags a
# real leak without flapping.
#
# Usage:
#   scripts/rss_check.sh                     # build via ./ionic_new
#   IONIC=./some_binary scripts/rss_check.sh
#
# Env overrides:
#   IONIC           compiler to use           (default: ./ionic_new)
#   RSS_N           small N                   (default: 200000)
#   RSS_FACTOR      large N = N * RSS_FACTOR  (default: 4)
#   RSS_MAX_GROWTH  max allowed growth, bytes (default: 8388608 = 8 MB)

set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
IONIC="${IONIC:-$ROOT/ionic_new}"
N1="${RSS_N:-200000}"
FACTOR="${RSS_FACTOR:-4}"
MAX_GROWTH="${RSS_MAX_GROWTH:-8388608}"
TEMPLATE="$ROOT/examples/rss_churn.ionic"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

if [ ! -x "$IONIC" ]; then
    echo "rss_check: compiler not found at '$IONIC' (run ./build.sh first)" >&2
    exit 2
fi

N2=$(( N1 * FACTOR ))

# Peak RSS (bytes) of a program, from macOS/BSD `time -l`.
peak_rss() {
    /usr/bin/time -l "$1" 2>&1 >/dev/null | awk '/maximum resident set size/ { print $1; exit }'
}

# Materialise the benchmark at a given N from the checked-in template.
build_variant() {
    local n="$1"
    local src="$WORK/churn_$n.ionic"
    local bin="$WORK/churn_$n"
    sed "s/let n = [0-9]*;/let n = $n;/" "$TEMPLATE" > "$src"
    if ! "$IONIC" "$src" -o "$bin" >/dev/null 2>&1; then
        echo "rss_check: failed to compile the benchmark at N=$n" >&2
        exit 2
    fi
    printf '%s' "$bin"
}

echo "==> RSS regression gate (N=$N1 vs N=$N2, max growth $(( MAX_GROWTH / 1048576 )) MB)"
BIN1="$(build_variant "$N1")"
BIN2="$(build_variant "$N2")"

RSS1="$(peak_rss "$BIN1")"
RSS2="$(peak_rss "$BIN2")"
DELTA=$(( RSS2 - RSS1 ))

printf '    N=%-8s peak RSS = %s bytes\n' "$N1" "$RSS1"
printf '    N=%-8s peak RSS = %s bytes\n' "$N2" "$RSS2"
printf '    growth        = %s bytes\n' "$DELTA"

if [ "$DELTA" -gt "$MAX_GROWTH" ]; then
    echo "    ✗ FAIL: peak RSS grows with iteration count — memory is not reclaimed."
    echo "      ${DELTA} bytes of growth over $(( N2 - N1 )) extra iterations"
    echo "      (~$(( DELTA / (N2 - N1) )) bytes leaked per iteration)."
    echo "      A correct memory model keeps this flat (delta near 0)."
    exit 1
fi

echo "    ✓ PASS: peak RSS is flat in N — churn is reclaimed, not leaked."
