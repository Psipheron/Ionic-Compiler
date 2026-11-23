#!/usr/bin/env bash
# RSS regression gate.
#
# The language's stated priority is the smallest possible resident set. This
# script compiles string-churn benchmarks at two sizes and checks what peak RSS
# does as the iteration count grows.
#
# Why scaling, not an absolute number:
#   A leaking allocator grows RSS ~linearly with N. A correct memory model
#   reclaims each iteration's garbage as it dies, so peak RSS is flat in N.
#   The startup baseline (linked runtime, libc, page table) is a constant that
#   cancels when we subtract, making the gate independent of machine, malloc
#   implementation, and OS version -- unlike a raw byte budget, which would be
#   brittle across CI images.
#
# ── Two shapes, two assertions ───────────────────────────────────────────────
#
# The memory model treats these differently, so a single benchmark was telling
# only half the story:
#
#   examples/rss_churn.ionic      `s = str_concat(...)` — assign into a `mut`
#                                 slot. The slot carries a claim column and loop
#                                 re-entry releases the previous value. This is
#                                 FIXED, so the assertion is the strong one:
#                                 growth must be near zero.
#
#   examples/rss_churn_let.ionic  `let t = ...` inside the loop body. This is a
#                                 KNOWN LEAK — releasing on re-entry is unsound
#                                 under the single-exit design because the slot
#                                 may hold a borrow the program never owned (see
#                                 docs/memory-model.md §9). The assertion is a
#                                 CEILING instead: the leak must not get worse,
#                                 and when it is fixed this gate starts
#                                 reporting the flat result without being
#                                 edited.
#
# Before this split, the gate ran only the assignment shape and passed, which
# read as "the memory model reclaims loop churn" when a whole allocation shape
# was still leaking. Measuring the leaking shape is what makes the pass mean
# something.
#
# Usage:
#   scripts/rss_check.sh                     # build via ./ionic_new
#   IONIC=./some_binary scripts/rss_check.sh
#
# Env overrides:
#   IONIC           compiler to use           (default: ./ionic_new)
#   RSS_N           small N                   (default: 200000)
#   RSS_FACTOR      large N = N * RSS_FACTOR  (default: 4)
#   RSS_MAX_GROWTH  max growth, assignment shape, bytes (default: 8388608 = 8 MB)
#   RSS_LET_MAX_BPI max bytes/iter, let shape           (default: 64)

set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
IONIC="${IONIC:-$ROOT/ionic_new}"
N1="${RSS_N:-200000}"
FACTOR="${RSS_FACTOR:-4}"
MAX_GROWTH="${RSS_MAX_GROWTH:-8388608}"
LET_MAX_BPI="${RSS_LET_MAX_BPI:-64}"
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

# Materialise a benchmark at a given N from its checked-in template, compile it,
# and print the path to the binary.
build_variant() {
    local template="$1" n="$2" tag="$3"
    local src="$WORK/${tag}_$n.ionic"
    local bin="$WORK/${tag}_$n"
    sed "s/let n = [0-9]*;/let n = $n;/" "$template" > "$src"
    if ! "$IONIC" "$src" -o "$bin" >/dev/null 2>&1; then
        echo "rss_check: failed to compile $tag at N=$n" >&2
        exit 2
    fi
    printf '%s' "$bin"
}

echo "==> RSS regression gate (N=$N1 vs N=$N2)"
fail=0

# ── shape 1: assignment into a mut slot — must be flat ──────────────────────
BIN1="$(build_variant "$ROOT/examples/rss_churn.ionic" "$N1" assign)"
BIN2="$(build_variant "$ROOT/examples/rss_churn.ionic" "$N2" assign)"
RSS1="$(peak_rss "$BIN1")"
RSS2="$(peak_rss "$BIN2")"
DELTA=$(( RSS2 - RSS1 ))

echo "    assignment shape (rss_churn.ionic), max growth $(( MAX_GROWTH / 1048576 )) MB"
printf '        N=%-8s peak RSS = %s bytes\n' "$N1" "$RSS1"
printf '        N=%-8s peak RSS = %s bytes\n' "$N2" "$RSS2"
printf '        growth        = %s bytes\n' "$DELTA"

if [ "$DELTA" -gt "$MAX_GROWTH" ]; then
    echo "    ✗ FAIL: peak RSS grows with iteration count — churn is not reclaimed."
    echo "        ${DELTA} bytes of growth over $(( N2 - N1 )) extra iterations"
    echo "        (~$(( DELTA / (N2 - N1) )) bytes leaked per iteration)."
    echo "        This shape is fixed; growth here is a regression."
    fail=1
fi

# ── shape 2: let inside the loop — documented leak, ceiling only ────────────
LBIN1="$(build_variant "$ROOT/examples/rss_churn_let.ionic" "$N1" let)"
LBIN2="$(build_variant "$ROOT/examples/rss_churn_let.ionic" "$N2" let)"
LRSS1="$(peak_rss "$LBIN1")"
LRSS2="$(peak_rss "$LBIN2")"
LDELTA=$(( LRSS2 - LRSS1 ))
LBPI=$(( LDELTA / (N2 - N1) ))

echo "    let-in-loop shape (rss_churn_let.ionic), ceiling $LET_MAX_BPI bytes/iter"
printf '        N=%-8s peak RSS = %s bytes\n' "$N1" "$LRSS1"
printf '        N=%-8s peak RSS = %s bytes\n' "$N2" "$LRSS2"
printf '        growth        = %s bytes (%s bytes/iter)\n' "$LDELTA" "$LBPI"

if [ "$LBPI" -gt "$LET_MAX_BPI" ]; then
    echo "    ✗ FAIL: the let-in-loop leak grew past its recorded ceiling."
    echo "        ${LBPI} bytes/iter, ceiling is ${LET_MAX_BPI}."
    echo "        Known leak, but this is worse than documented — investigate."
    fail=1
elif [ "$LBPI" -le 1 ]; then
    # Reaching here means the leak was fixed. Say so, so the ceiling can be
    # tightened to a flat assertion rather than silently staying lenient.
    echo "    ✓ let-in-loop churn is now reclaimed too — tighten this gate to a"
    echo "      flat assertion and delete the note in docs/memory-model.md §9."
fi

if [ "$fail" -ne 0 ]; then
    exit 1
fi

echo "    ✓ PASS: assignment churn is flat; the known let-in-loop leak is bounded."
