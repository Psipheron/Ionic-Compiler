#!/bin/sh
# Reference-counting probe gate (memory-model §8 step 4, stage 1).
#
# Stages 1 and 2 of step 4 are deliberately separable. Stage 1 routes every
# string producer through ionic_rc_alloc but emits no retain/release, so a
# running Ionic program never looks at the header. That is what makes stage 1
# safe, and also what makes it invisible: nothing exercises the contract, so a
# producer that silently kept a raw malloc would compile, run, and pass every
# other check in this repo.
#
# Stage 2 emits releases. At that point a missed producer is not a missed
# optimisation — it is `free()` on a pointer eight bytes into a malloc block, on
# the first call. The probe is therefore written before the thing that needs it,
# and this script makes it possible to run it without remembering the sanitizer
# flags.
#
# What it checks (all in scripts/rcprobe.c, which #includes the runtime TU):
#   every string producer returns rc==1 ... including its failure paths
#   a literal copy is pinned and interned ... and survives a release
#   retain/release balance 1 -> 3 -> 1 ......... and frees at zero
#   rc_realloc preserves header and contents ... grows and shrinks
#   rc_realloc on a pinned block ............... copies into a live block,
#                                                leaving the interned one intact
#
# It runs under ASan and UBSan because the failure mode it guards against is
# memory corruption, which a plain build reports as "mostly worked".
#
# Deliberately NOT checked: double release. Once the count hits zero the block is
# freed, so a second release reads reclaimed memory — the header load is the
# use-after-free, before any guard could look at the count. There is no runtime
# check that can catch it; only one-release-per-owning-slot can. See the
# ionic_rc_release comment.
#
# Usage:
#   scripts/rc_check.sh
set -u
case "$0" in
  */*) cd "$(dirname "$0")/.." || exit 1 ;;
  *)   cd . || exit 1 ;;
esac

CC="${CC:-clang}"
RUNTIME=src/ionic_model_runtime.c
PROBE=scripts/rcprobe.c
BIN="${TMPDIR:-/tmp}/ionic_rcprobe"

for f in "$RUNTIME" "$PROBE"; do
    if [ ! -f "$f" ]; then
        echo "rc_check: missing $f" >&2
        exit 2
    fi
done

if ! command -v "$CC" >/dev/null 2>&1; then
    echo "rc_check: no C compiler ('$CC'); set CC= to override" >&2
    exit 2
fi

echo "==> RC probe (ASan + UBSan)"
# -O1 keeps the sanitizers honest about inlining without making the probe slow.
# Leak detection is off on purpose: several producers below (the pinned literal
# table, the interned copies) are leak-by-design for the process lifetime, and a
# leak report would be noise about a decision, not a bug.
if ! ASAN_OPTIONS=detect_leaks=0 UBSAN_OPTIONS=halt_on_error=1 \
     "$CC" -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer \
     -Wall -Wextra -Wno-unused-parameter \
     "$PROBE" -o "$BIN"; then
    echo "    ✗ FAIL: probe did not compile (a producer changed signature?)"
    exit 1
fi

# stdin is /dev/null on purpose: one case calls ionic_read_line, which would
# block forever on an interactive terminal. On /dev/null it takes the EOF path,
# which is the failure path the probe wants anyway.
if ASAN_OPTIONS=detect_leaks=0 UBSAN_OPTIONS=halt_on_error=1 "$BIN" < /dev/null; then
    echo "    ✓ PASS: every producer returns a counted block; release reclaims it."
else
    echo "    ✗ FAIL: the producer contract is broken (see above)."
    exit 1
fi
