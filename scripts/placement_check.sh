#!/bin/sh
# Regression harness for hardware-placement checks (Stages B2–B5).
#
# `@cpu` / `@gpu` name memory pools, not devices: SRAM (host memory) and VRAM
# (accelerator memory). VRAM is the scarce one, so the interesting question is
# not "which device is this on" but "does it fit". Stage B2 gives the pool a
# declared size and gives `@gpu(f)` a meaning: f is a share of that pool.
# Stage B3 sizes a tensor from its constructor, and B4 charges that against the
# enclosing scope's share. B5 makes a transfer a move, so that a value has
# exactly one owner — without which capacity accounting counts the same bytes
# twice under two names.
#
# Before B2, `@gpu(0.5)` was parsed and then discarded — a claim the compiler
# never looked at — and a top-level `let vram = ...` was an ordinary global
# nothing read. This script pins down the checks that changed:
#
#   declared pool + valid fraction      -> compiles
#   fraction above 1                    -> rejected
#   fraction with no pool declared      -> rejected
#   pool declared as zero / negative    -> rejected
#   pool declared twice                 -> rejected
#   pool declared as a non-byte-count   -> rejected
#   GPU allocation over the budget      -> rejected
#   same allocation, wider fraction     -> compiles
#   dtype counted in the size           -> F16 fits where F32 does not
#   gpu { } blocks budgeted too         -> rejected
#   tensor parameters / unsized results -> exempt (never a false rejection),
#                                          and unsized ones are reported (warn)
#   no declared pool                    -> no budget in force
#   using a transferred value           -> rejected (B5)
#   builtins borrow / temporaries move  -> compiles (B5)
#   mastery/04_placement_types.ionic    -> still compiles (the tutorial's reference)
#   mastery/04_pool_polymorphism.ionic  -> still compiles (transfers, done correctly)
#   mastery/fail/use_after_move.ionic   -> rejected (the tutorial's fail-case)
#   over-budget allocation at run time  -> refused by the allocator (Stage D)
#   declared pool readable at run time  -> matches what was written
#   device capacity readable at run time-> the other half of the comparison
#   a copied handle outlives its source -> reference counted at run time (Stage D)
#
# Everything above the "runtime" sections is decided statically, so it runs on
# any OS with no accelerator present. That is the point: the checker's behaviour
# is portable even when the backend is not. The runtime sections do the opposite
# on purpose — they execute the binary, because a budget the checker merely
# *believes* is not a budget. They report "skip" rather than "ok" when no backend
# is linked in, so a platform without one cannot look like it verified something
# it never ran.
set -u
# Run from the repo root regardless of where the script is invoked from, but
# derive it from the script's own location: a hard-coded absolute path made this
# script pass locally and fail in CI, where the checkout lives elsewhere.
case "$0" in
  */*) cd "$(dirname "$0")/.." || exit 1 ;;
  *)   cd . || exit 1 ;;
esac

COMPILER="${COMPILER:-./ionic_new}"
WORK=/tmp/ionic_placement
mkdir -p "$WORK"
FAIL=0

# expect_ok <name> <source>
expect_ok() {
  name=$1; src=$2
  printf '%s\n' "$src" > "$WORK/t.ionic"
  rm -f "$WORK/t" "$WORK/t.o"
  if "$COMPILER" "$WORK/t.ionic" -o "$WORK/t" >"$WORK/err.txt" 2>&1; then
    echo "ok    $name"
  else
    echo "FAIL  $name  (expected to compile)"
    sed -n '1,6p' "$WORK/err.txt"
    FAIL=1
  fi
}

# expect_reject <name> <needle> <source>
# `needle` must appear in the diagnostic, so a rejection for the wrong reason
# does not pass silently.
expect_reject() {
  name=$1; needle=$2; src=$3
  printf '%s\n' "$src" > "$WORK/t.ionic"
  rm -f "$WORK/t" "$WORK/t.o"
  if "$COMPILER" "$WORK/t.ionic" -o "$WORK/t" >"$WORK/err.txt" 2>&1; then
    echo "FAIL  $name  (expected rejection, but it compiled)"
    FAIL=1
    return
  fi
  if NO_COLOR=1 grep -q -- "$needle" "$WORK/err.txt"; then
    echo "ok    $name  (rejected: $needle)"
  else
    echo "FAIL  $name  (rejected, but not because of '$needle')"
    sed -n '1,6p' "$WORK/err.txt"
    FAIL=1
  fi
}

# expect_warns <name> <needle> <source>
# The program must COMPILE (a warning is not an error) and the compiler must
# print `needle`. Both halves matter: an exempted value has to stay compilable,
# and the exemption has to be visible.
expect_warns() {
  name=$1; needle=$2; src=$3
  printf '%s\n' "$src" > "$WORK/t.ionic"
  rm -f "$WORK/t" "$WORK/t.o"
  if ! "$COMPILER" "$WORK/t.ionic" -o "$WORK/t" >"$WORK/err.txt" 2>&1; then
    echo "FAIL  $name  (expected a warning, but it failed to compile)"
    sed -n '1,6p' "$WORK/err.txt"
    FAIL=1
    return
  fi
  if NO_COLOR=1 grep -q -- "$needle" "$WORK/err.txt"; then
    echo "ok    $name  (warned: $needle)"
  else
    echo "FAIL  $name  (compiled, but did not warn about '$needle')"
    sed -n '1,6p' "$WORK/err.txt"
    FAIL=1
  fi
}

# expect_silent <name> <needle> <source>
# The program must compile AND must not print `needle`. This is the half that
# keeps the warning worth reading: a diagnostic that fires on every program is
# one readers learn to skip.
expect_silent() {
  name=$1; needle=$2; src=$3
  printf '%s\n' "$src" > "$WORK/t.ionic"
  rm -f "$WORK/t" "$WORK/t.o"
  if ! "$COMPILER" "$WORK/t.ionic" -o "$WORK/t" >"$WORK/err.txt" 2>&1; then
    echo "FAIL  $name  (expected to compile)"
    sed -n '1,6p' "$WORK/err.txt"
    FAIL=1
    return
  fi
  if NO_COLOR=1 grep -q -- "$needle" "$WORK/err.txt"; then
    echo "FAIL  $name  (warned about '$needle' when it should be silent)"
    sed -n '1,6p' "$WORK/err.txt"
    FAIL=1
  else
    echo "ok    $name"
  fi
}

# expect_warn_count <name> <needle> <count> <source>
# The program must compile and print `needle` exactly `count` times. `expect_warns`
# cannot catch a diagnostic emitted twice at the same line: the needle is present
# either way. Counting is what makes a duplicate a failure.
expect_warn_count() {
  name=$1; needle=$2; want=$3; src=$4
  printf '%s\n' "$src" > "$WORK/t.ionic"
  rm -f "$WORK/t" "$WORK/t.o"
  if ! "$COMPILER" "$WORK/t.ionic" -o "$WORK/t" >"$WORK/err.txt" 2>&1; then
    echo "FAIL  $name  (expected a warning, but it failed to compile)"
    sed -n '1,6p' "$WORK/err.txt"
    FAIL=1
    return
  fi
  got=$(NO_COLOR=1 grep -c -- "$needle" "$WORK/err.txt" || true)
  if [ "$got" = "$want" ]; then
    echo "ok    $name  ($want x $needle)"
  else
    echo "FAIL  $name  (warned $got times, expected $want)"
    sed -n '1,10p' "$WORK/err.txt"
    FAIL=1
  fi
}

echo "=== pool declaration + fraction (Stage B2) ==="

expect_ok "declared pool, valid fraction" \
'let vram = 1073741824;
@gpu(0.25) fn f(tensor@gpu x) -> tensor@gpu { return x; }
fn main() -> int64 { println("ok"); return 0; }'

# 1.0 is the whole pool — the largest fraction that still means something.
expect_ok "fraction of exactly 1.0" \
'let vram = 1073741824;
@gpu(1.0) fn f(tensor@gpu x) -> tensor@gpu { return x; }
fn main() -> int64 { return 0; }'

# A bare @gpu with no fraction does not need a pool: it inherits the whole of
# one only if a pool exists, and makes no claim otherwise.
expect_ok "@gpu with no fraction needs no pool" \
'@gpu fn f(tensor@gpu x) -> tensor@gpu { return x; }
fn main() -> int64 { return 0; }'

# The tutorial's "correct placement" reference (unit 04) uses `@gpu(0.5)`, so it
# must declare a pool. It is the file readers are pointed at, which makes it the
# one a fraction rule must never quietly break.
expect_ok "the placement-types reference still compiles" \
"$(cat examples/mastery/04_placement_types.ionic)"

expect_reject "fraction above 1" "must be greater than 0 and at most 1" \
'let vram = 1073741824;
@gpu(1.5) fn f(tensor@gpu x) -> tensor@gpu { return x; }
fn main() -> int64 { return 0; }'

expect_reject "fraction of zero" "must be greater than 0 and at most 1" \
'let vram = 1073741824;
@gpu(0) fn f(tensor@gpu x) -> tensor@gpu { return x; }
fn main() -> int64 { return 0; }'

expect_reject "fraction with no pool declared" "no pool is declared" \
'@gpu(0.5) fn f(tensor@gpu x) -> tensor@gpu { return x; }
fn main() -> int64 { return 0; }'

expect_reject "pool declared as zero" "must be positive" \
'let vram = 0;
fn main() -> int64 { return 0; }'

expect_reject "pool declared twice" "declare it once" \
'let vram = 100;
let vram = 200;
fn main() -> int64 { return 0; }'

expect_reject "pool declared as a float" "plain byte count" \
'let vram = 1.5;
fn main() -> int64 { return 0; }'

echo
echo "=== static sizes and the scope budget (Stages B3/B4) ==="

# Stage B3 sizes a tensor from its constructor: prod(shape) x dtype_bytes.
# Stage B4 charges that against the scope's share of the pool. The failure has
# to name the allocation responsible, so the diagnostic carries the byte count.
expect_reject "a GPU allocation over the budget is rejected" "does not fit the accelerator pool" \
'let vram = 1073741824;
@gpu(0.01) fn alloc() -> int64 {
    let a = mlx_zeros([1024, 1024], 0).toGpu();
    let b = mlx_zeros([1024, 1024], 0).toGpu();
    let c = mlx_zeros([1024, 1024], 0).toGpu();
    return 1;
}
fn main() -> int64 { return 0; }'

# The same allocations under a wide-enough fraction must compile: the budget is
# a share of the pool, not an absolute cap, so widening it is the fix.
expect_ok "same allocations fit a wider fraction" \
'let vram = 1073741824;
@gpu(0.5) fn alloc() -> int64 {
    let a = mlx_zeros([1024, 1024], 0).toGpu();
    let b = mlx_zeros([1024, 1024], 0).toGpu();
    let c = mlx_zeros([1024, 1024], 0).toGpu();
    return 1;
}
fn main() -> int64 { return 0; }'

# dtype is part of the size: F16 is 2 bytes an element, F32 is 4. A shape that
# blows the budget in F32 fits in F16, so a checker that ignored dtype would
# reject this. 20000 x 20000 is 400M elements: 800 MB in F16, 1.6 GB in F32.
expect_ok "dtype is counted in the size" \
'let vram = 1073741824;
@gpu(1.0) fn alloc() -> int64 {
    let a = mlx_zeros([20000, 20000], 1).toGpu();
    return 1;
}
fn main() -> int64 { return 0; }'

expect_reject "the same shape in F32 does not fit" "does not fit the accelerator pool" \
'let vram = 1073741824;
@gpu(1.0) fn alloc() -> int64 {
    let a = mlx_zeros([20000, 20000], 0).toGpu();
    return 1;
}
fn main() -> int64 { return 0; }'

# ── The budget follows live values, not declarations ─────────────────────────
#
# Each of the following was a *false rejection* before the accounting learned
# that a value can stop being live: the program's real peak fits, and rejecting
# it made the pool look smaller than the machine it describes. They are asserted
# because the over-approximation was easy to reintroduce — any change that
# charges on declaration and never releases brings all of them back.
#
# 4 x 256 MiB live = exactly 1 GiB. Freeing one makes room for a fifth.
expect_ok "mlx_free releases the charge it took" \
'let vram = 1073741824;
@gpu fn work() -> int64 {
    let a = mlx_zeros([8192, 8192], 0).toGpu();
    let b = mlx_zeros([8192, 8192], 0).toGpu();
    let c = mlx_zeros([8192, 8192], 0).toGpu();
    let d = mlx_zeros([8192, 8192], 0).toGpu();
    mlx_free(a);
    let e = mlx_zeros([8192, 8192], 0).toGpu();
    return 0;
}
fn main() -> int64 { return 0; }'

# The same, twice over: two frees make room for two allocations.
expect_ok "two frees release two charges" \
'let vram = 1073741824;
@gpu fn work() -> int64 {
    let a = mlx_zeros([8192, 8192], 0).toGpu();
    let b = mlx_zeros([8192, 8192], 0).toGpu();
    let c = mlx_zeros([8192, 8192], 0).toGpu();
    let d = mlx_zeros([8192, 8192], 0).toGpu();
    mlx_free(a);
    mlx_free(b);
    let e = mlx_zeros([8192, 8192], 0).toGpu();
    let f = mlx_zeros([8192, 8192], 0).toGpu();
    return 0;
}
fn main() -> int64 { return 0; }'

# `.toCpu()` moves the value out of VRAM, so its bytes stop counting against the
# accelerator pool. Two moved back, one allocated: peak stays at 1 GiB.
expect_ok "toCpu releases the VRAM charge" \
'let vram = 1073741824;
@gpu fn work() -> int64 {
    let a = mlx_zeros([8192, 8192], 0).toGpu();
    let b = mlx_zeros([8192, 8192], 0).toGpu();
    let c = mlx_zeros([8192, 8192], 0).toGpu();
    let d = mlx_zeros([8192, 8192], 0).toGpu();
    let a2 = a.toCpu();
    let b2 = b.toCpu();
    let e = mlx_zeros([8192, 8192], 0).toGpu();
    return 0;
}
fn main() -> int64 { return 0; }'

# Sibling `if` branches do NOT accumulate a charge twice... but they DO both
# count. A branch's locals go out of scope at its `}`, yet Ionic handles are not
# RAII-freed: no runtime release happens, so a program whose branches both run
# keeps every allocation. Releasing a branch's charges at `}` made the checker
# accept the program below against a 1 GiB pool, and it then aborted at run time
# on the eighth 256 MiB allocation. The over-approximation is the correct
# direction here: a false rejection is a compile error with a line number, a
# false acceptance is a runtime abort somewhere else.
expect_reject "sibling if branches both count" "does not fit the accelerator pool" \
'let vram = 1073741824;
@gpu fn work(int64 c) -> int64 {
    if (c > 0) {
        let a = mlx_zeros([8192, 8192], 0).toGpu();
        let b = mlx_zeros([8192, 8192], 0).toGpu();
        let d = mlx_zeros([8192, 8192], 0).toGpu();
        let e = mlx_zeros([8192, 8192], 0).toGpu();
    }
    if (c > 1) {
        let f = mlx_zeros([8192, 8192], 0).toGpu();
        let g = mlx_zeros([8192, 8192], 0).toGpu();
        let h = mlx_zeros([8192, 8192], 0).toGpu();
        let i = mlx_zeros([8192, 8192], 0).toGpu();
    }
    return 0;
}
fn main() -> int64 { return 0; }'

# The same program with mlx_free() calls in each branch does fit, because the
# bytes really are returned before the second branch allocates. This is the
# difference the checker is trying to express: scope exit is not a release, but
# mlx_free() is.
expect_ok "freeing inside each branch makes them fit" \
'let vram = 1073741824;
@gpu fn work(int64 c) -> int64 {
    if (c > 0) {
        let a = mlx_zeros([8192, 8192], 0).toGpu();
        let b = mlx_zeros([8192, 8192], 0).toGpu();
        let d = mlx_zeros([8192, 8192], 0).toGpu();
        let e = mlx_zeros([8192, 8192], 0).toGpu();
        mlx_free(a);
        mlx_free(b);
        mlx_free(d);
        mlx_free(e);
    }
    if (c > 1) {
        let f = mlx_zeros([8192, 8192], 0).toGpu();
        let g = mlx_zeros([8192, 8192], 0).toGpu();
        let h = mlx_zeros([8192, 8192], 0).toGpu();
        let i = mlx_zeros([8192, 8192], 0).toGpu();
    }
    return 0;
}
fn main() -> int64 { return 0; }'

# `let b = a` copies the *handle*, not the allocation. At run time the pool
# holds those bytes once, so five names for one 256 MiB buffer is 256 MiB.
expect_ok "aliases of one handle are charged once" \
'let vram = 1073741824;
@gpu fn work() -> int64 {
    let a = mlx_zeros([8192, 8192], 0).toGpu();
    let b = a;
    let c = a;
    let d = a;
    let e = a;
    return 0;
}
fn main() -> int64 { return 0; }'

# An alias of an alias. The chain `a -> e -> f` names one buffer three times, so a
# charge that followed only one hop would bill 512 MiB as 768 MiB and reject it.
expect_ok "a chain of aliases is charged once" \
'let vram = 1073741824;
@gpu fn work() -> int64 {
    let a = mlx_zeros([8192, 8192], 0).toGpu();
    let b = mlx_zeros([8192, 8192], 0).toGpu();
    let c = mlx_zeros([8192, 8192], 0).toGpu();
    let d = mlx_zeros([8192, 8192], 0).toGpu();
    let e = a;
    let f = e;
    return 0;
}
fn main() -> int64 { return 0; }'

# ...and freeing through the far end of the chain releases the bytes once, which
# is what makes room for a fifth allocation.
expect_ok "freeing through an alias chain releases the charge" \
'let vram = 1073741824;
@gpu fn work() -> int64 {
    let a = mlx_zeros([8192, 8192], 0).toGpu();
    let b = mlx_zeros([8192, 8192], 0).toGpu();
    let c = mlx_zeros([8192, 8192], 0).toGpu();
    let d = mlx_zeros([8192, 8192], 0).toGpu();
    let e = a;
    let f = e;
    mlx_free(f);
    let g = mlx_zeros([8192, 8192], 0).toGpu();
    return 0;
}
fn main() -> int64 { return 0; }'

# ...while a released value that is still referenced elsewhere must not free the
# bytes twice. `mlx_free(b)` releases the charge `a` carries; that is one release
# for one allocation, and the total returns to exactly zero.
expect_ok "freeing an alias releases the charge once" \
'let vram = 1073741824;
@gpu fn work() -> int64 {
    let a = mlx_zeros([8192, 8192], 0).toGpu();
    let b = a;
    let c = a;
    let d = a;
    mlx_free(b);
    let e = mlx_zeros([8192, 8192], 0).toGpu();
    return 0;
}
fn main() -> int64 { return 0; }'

# A nested `gpu { }` adds to the enclosing scope's total rather than restarting
# from zero. 256 MiB outside plus 4 x 256 MiB inside is 1.25 GiB, over a 1 GiB
# pool, so this must be rejected — it used to compile because the block was
# checked against the whole pool instead of what was left of it.
expect_reject "a nested gpu block adds to the enclosing total" \
  "does not fit the accelerator pool" \
'let vram = 1073741824;
@gpu fn work() -> int64 {
    let outer = mlx_zeros([8192, 8192], 0).toGpu();
    gpu {
        let a = mlx_zeros([8192, 8192], 0).toGpu();
        let b = mlx_zeros([8192, 8192], 0).toGpu();
        let c = mlx_zeros([8192, 8192], 0).toGpu();
        let d = mlx_zeros([8192, 8192], 0).toGpu();
    }
    return 0;
}
fn main() -> int64 { return 0; }'

# The same block one value smaller fits, so the rejection above is about the
# total and not about nesting as such.
expect_ok "a nested gpu block that fits is accepted" \
'let vram = 1073741824;
@gpu fn work() -> int64 {
    let outer = mlx_zeros([8192, 8192], 0).toGpu();
    gpu {
        let a = mlx_zeros([8192, 8192], 0).toGpu();
        let b = mlx_zeros([8192, 8192], 0).toGpu();
        let c = mlx_zeros([8192, 8192], 0).toGpu();
    }
    return 0;
}
fn main() -> int64 { return 0; }'

# The block's charge survives the block. A `gpu { }` does not release its values
# at `}`, so a value allocated after the block is live alongside them and must be
# measured against them. Restoring the pre-block total on exit accepted this
# against a 1 GiB pool.
expect_reject "a value allocated after a gpu block still counts against it" \
  "does not fit the accelerator pool" \
'let vram = 1073741824;
@gpu fn work() -> int64 {
    gpu {
        let a = mlx_zeros([8192, 8192], 0).toGpu();
        let b = mlx_zeros([8192, 8192], 0).toGpu();
        let c = mlx_zeros([8192, 8192], 0).toGpu();
        let d = mlx_zeros([8192, 8192], 0).toGpu();
    }
    let after = mlx_zeros([8192, 8192], 0).toGpu();
    return 0;
}
fn main() -> int64 { return 0; }'

# A bare `gpu { }` block has no fraction, so it inherits the whole pool. The
# budget is charged there too, not only in @gpu functions.
expect_reject "gpu blocks are budgeted as well" "does not fit the accelerator pool" \
'let vram = 1073741824;
fn main() -> int64 {
    let t = mlx_zeros([64, 64], 0);
    gpu {
        let a = mlx_zeros([8192, 8192], 0).toGpu();
        let b = mlx_zeros([8192, 8192], 0).toGpu();
        let c = mlx_zeros([8192, 8192], 0).toGpu();
        let d = mlx_zeros([8192, 8192], 0).toGpu();
        let e = mlx_zeros([8192, 8192], 0).toGpu();
        let f = mlx_zeros([8192, 8192], 0).toGpu();
    }
    return 0;
}'

# A parameter has no static size — it was allocated, and charged, in the
# caller's scope. It must not be rejected here, or every function that takes a
# tensor becomes uncompilable (see decision 4 in docs/plan-placement.md).
expect_ok "tensor parameters are exempt from the budget" \
'let vram = 1073741824;
@gpu(0.01) fn uses(tensor@gpu x, tensor@gpu y) -> tensor@gpu {
    return mlx_add(x, y);
}
fn main() -> int64 { return 0; }'

# An unsized result (a reshape, whose shape this pass does not model) is exempt
# for the same reason: the check is a necessary condition, never a false
# rejection.
#
# "Exempt" used to mean "silently skipped", which is the one thing a capacity
# check must not do: a program compiled whether its GPU values were measured or
# merely ignored. It still compiles — an unbounded value is not provably
# wrong — but the exemption is now reported, so "this fits" and "this was never
# sized" are distinguishable.
expect_warns "unsized results are exempt from the budget, but reported" "could not be sized statically" \
'let vram = 1073741824;
@gpu(0.01) fn reshapes(tensor@gpu x) -> tensor@gpu {
    let r = mlx_reshape(x, [4, 4]);
    return r;
}
fn main() -> int64 { return 0; }'

# A shape built at runtime — `[n, n]` from an argument — is the case that
# motivated the warning: `mlx_zeros([n, n], 0).toGpu()` allocates an amount the
# checker cannot know, and it used to compile against any budget with no hint
# that nothing was measured.
expect_warns "a runtime-sized allocation is reported" "could not be sized statically" \
'let vram = 1073741824;
@gpu(0.01) fn grow(int64 n) -> int64 {
    let big = mlx_zeros([n, n], 0).toGpu();
    return mlx_size(big);
}
fn main() -> int64 { return 0; }'

# The warning applies inside a bare gpu { } block too, not only in @gpu fns.
expect_warns "a gpu block reports its unsized values" "could not be sized statically" \
'let vram = 1073741824;
fn main() -> int64 {
    let t = mlx_zeros([4, 4], 0);
    gpu {
        let big = mlx_reshape(t.toGpu(), [8, 8]);
        let m = mlx_size(big);
    }
    return 0;
}'

# ...and it must stay silent when everything is sized, or a clean program would
# carry a warning it cannot act on.
expect_silent "a fully-sized GPU scope does not warn" "could not be sized statically" \
'let vram = 1073741824;
@gpu(1.0) fn alloc() -> int64 {
    let a = mlx_zeros([1024, 1024], 0).toGpu();
    let b = mlx_zeros([1024, 1024], 0).toGpu();
    return mlx_size(b);
}
fn main() -> int64 { return 0; }'

# A `gpu { }` nested inside an @gpu fn is its own budgeted scope. It must report
# its own unsized value exactly once: the counters have to be saved and restored
# around the block, or the fn's own close reports the same value a second time
# and the user sees one problem printed twice with the same line and column.
expect_warn_count "a nested gpu block reports once, not at both closes" "could not be sized statically" 1 \
'let vram = 1073741824;
@gpu(1.0) fn outer(int64 n) -> int64 {
    let a = mlx_zeros([1024, 1024], 0).toGpu();
    gpu {
        let big = mlx_zeros([n, n], 0).toGpu();
        let s = mlx_size(big);
    }
    return mlx_size(a);
}
fn main() -> int64 { return 0; }'

# Reassignment is the second way to create a GPU-resident value. Only `let` was
# charged, so a tensor too big for the pool could be assigned into an existing
# binding and the check never ran:
#
#     mut slot = mlx_zeros([1024, 1024], 0).toGpu();
#     slot = mlx_zeros([20000, 20000], 0).toGpu();   // 1.6 GiB, pool is 1 GiB
#
# That compiled. This pins the fix.
expect_reject "an over-budget tensor reassigned into a slot is rejected" "does not fit the accelerator pool" \
'let vram = 1073741824;
@gpu(1.0) fn alloc() -> int64 {
    mut slot = mlx_zeros([1024, 1024], 0).toGpu();
    slot = mlx_zeros([20000, 20000], 0).toGpu();
    return 1;
}
fn main() -> int64 { return 0; }'

# ...and the fix must not double-count a reused slot. Each assignment releases
# the previous value, so five assignments of 256 MiB through one binding peak at
# 256 MiB, not 1.25 GiB, and must stay accepted.
expect_ok "a reused slot releases the old value before recharging" \
'let vram = 1073741824;
@gpu(1.0) fn alloc() -> int64 {
    mut slot = mlx_zeros([8192, 8192], 0).toGpu();
    slot = mlx_zeros([8192, 8192], 0).toGpu();
    slot = mlx_zeros([8192, 8192], 0).toGpu();
    slot = mlx_zeros([8192, 8192], 0).toGpu();
    slot = mlx_zeros([8192, 8192], 0).toGpu();
    return 1;
}
fn main() -> int64 { return 0; }'

# A binding reassigned to a runtime-shaped value must stop being sized. The size
# record feeds `expr_bytes`, so leaving the old count in place charges every
# value derived from it:
#
#     mut a = mlx_zeros([8192, 8192], 0).toGpu();  // 256 MiB
#     a = mlx_reshape(a, [n, n]);                  // unknown, NOT 256 MiB
#     let b = mlx_add(a, a);                       // must be exempt, not 256 MiB
#
# Before the fix the five derived values were charged a stale 256 MiB each and
# the program was rejected — a false rejection, the one thing this check
# promises never to produce.
expect_warns "a value derived from an unsized binding is exempt, not charged stale bytes" "could not be sized statically" \
'let vram = 1073741824;
@gpu(1.0) fn f(int64 n) -> int64 {
    mut a = mlx_zeros([8192, 8192], 0).toGpu();
    a = mlx_reshape(a, [n, n]);
    let b = mlx_add(a, a);
    let c = mlx_add(a, a);
    let d = mlx_add(a, a);
    let e = mlx_add(a, a);
    let g = mlx_add(a, a);
    return mlx_size(g);
}
fn main() -> int64 { return 0; }'

# Releasing a reassigned binding's bytes is only valid in the scope that counted
# them. A nested `gpu { }` starts its running total at zero, so subtracting an
# enclosing scope's charge from it drives the total below what the body actually
# holds:
#
#     gpu { mut a = ...256 MiB...;      // charged to the OUTER scope
#       gpu { a = ...1 GiB + 64 KiB...; // inner total must start at 0, not -256 MiB
#       } }
#
# That compiled with the unconditional subtraction. The 16385 makes the inner
# value exceed the pool by a margin, so the assertion discriminates rather than
# relying on the two byte counts coincidentally matching.
expect_reject "a nested scope does not release an outer scope's charge" "does not fit the accelerator pool" \
'let vram = 1073741824;
fn main() -> int64 {
    gpu {
        mut a = mlx_zeros([8192, 8192], 0).toGpu();
        gpu {
            a = mlx_zeros([16384, 16385], 0).toGpu();
            let u = mlx_size(a);
        }
    }
    return 0;
}'

# The warning's total wording switches when the scope held nothing sizeable.
# `the 0 bytes it could size are only the known total` would read as noise, so
# that case says so plainly instead.
expect_warns "a scope that sized nothing says so" "nothing in this scope could be sized" \
'let vram = 1073741824;
@gpu(1.0) fn f(int64 n) -> int64 {
    let a = mlx_zeros([n, n], 0).toGpu();
    return mlx_size(a);
}
fn main() -> int64 { return 0; }'

# ...and the other wording appears when the scope did size something, so the
# report distinguishes "0 of a 1 GiB budget" from "256 MiB of a 1 GiB budget".
expect_warns "a partly-sized scope reports the known total" "bytes it could size are only the known total" \
'let vram = 1073741824;
@gpu(1.0) fn f(int64 n) -> int64 {
    let a = mlx_zeros([8192, 8192], 0).toGpu();
    let b = mlx_zeros([n, n], 0).toGpu();
    return mlx_size(b);
}
fn main() -> int64 { return 0; }'

# No pool declared means no budget in force: a program that never restricts a
# GPU scope is not asked to declare one.
expect_ok "no pool means no budget" \
'@gpu fn alloc() -> int64 {
    let a = mlx_zeros([8192, 8192], 0).toGpu();
    let b = mlx_zeros([8192, 8192], 0).toGpu();
    return 1;
}
fn main() -> int64 { return 0; }'

echo
echo "=== existing placement rules still hold ==="

expect_reject "user functions stay placement-strict" "expected tensor@cpu, got tensor@gpu" \
'@cpu fn consume(tensor@cpu x) -> int64 { return 1; }
@gpu fn produce(tensor@gpu x) -> tensor@gpu { return x; }
fn main() -> int64 {
    let a = mlx_ones([2, 2], 0);
    let d = a.toGpu();
    return consume(d);
}'

# Two separate bindings: `a.toGpu()` moves `a`, so reusing it here would trip
# the B5 rule below and this case would pass for the wrong reason.
expect_reject "mixed-pool operands rejected" "mixed memory pools" \
'fn main() -> int64 {
    let a = mlx_ones([2, 2], 0);
    let b = mlx_ones([2, 2], 0);
    let d = b.toGpu();
    gpu { let bad = mlx_add(a, d); }
    return 0;
}'

echo
echo "=== transfers move (Stage B5) ==="

# `x.toGpu()` consumes `x`: the value belongs to exactly one binding afterwards,
# which is what makes capacity accounting meaningful. Using the old name is a
# use-after-move, and the diagnostic says which transfer took it and where.
expect_reject "using a value after transferring it" "use after move" \
'fn main() -> int64 {
    let a = mlx_ones([2, 2], 0);
    let g = a.toGpu();
    gpu {
        let s = mlx_add(a.toGpu(), g);
    }
    return 0;
}'

expect_reject "freeing a value that was transferred away" "use after move" \
'fn main() -> int64 {
    let a = mlx_ones([2, 2], 0);
    let g = a.toGpu();
    gpu { let u = mlx_add(g, g); }
    mlx_free(a);
    return 0;
}'

# The reverse direction moves too: `.toCpu()` consumes the GPU binding.
expect_reject "using a value after transferring it back" "use after move" \
'fn main() -> int64 {
    let a = mlx_ones([2, 2], 0);
    let g = a.toGpu();
    let c = g.toCpu();
    gpu { let u = mlx_add(g, g); }
    return 0;
}'

# Builtins BORROW. Reading the same handle twice in one call is the normal case
# and must stay legal — if a call consumed its operands, `mlx_add(g, g)` would
# be an error and every expression would need a copy of its inputs.
expect_ok "builtins borrow, so a handle can be read twice" \
'fn main() -> int64 {
    let a = mlx_ones([2, 2], 0);
    let s = mlx_add(a, a);
    let g = a.toGpu();
    gpu {
        let u = mlx_add(g, g);
        println(str_concat("ndim: ", int64_to_str(mlx_ndim(u))));
    }
    return 0;
}'

# A temporary has no binding to strand: `f(...).toGpu()` moves nothing that a
# later line could name.
expect_ok "transferring a temporary consumes no binding" \
'fn main() -> int64 {
    let g = mlx_ones([2, 2], 0).toGpu();
    gpu { let u = mlx_add(g, g); }
    return 0;
}'

# Rebinding clears the mark. The old value was moved; the new one is a distinct
# value in a reused slot, and naming it is not a use-after-move.
expect_ok "rebinding a name clears the moved mark" \
'fn main() -> int64 {
    let a = mlx_ones([2, 2], 0);
    let g = a.toGpu();
    let a = mlx_ones([2, 2], 0);
    let h = a.toGpu();
    gpu { let u = mlx_add(g, h); }
    return 0;
}'

# The tutorial's pool-polymorphism example has to keep compiling: it is the one
# place showing that a transfer is the only way to enter the accelerator pool.
expect_ok "the pool-polymorphism reference still compiles" \
"$(cat examples/mastery/04_pool_polymorphism.ionic)"

# The tutorial's fail-case reference for this rule (exercise 04.8).
expect_reject "the use-after-move fail-case example is rejected" "use after move" \
"$(cat examples/mastery/fail/use_after_move.ionic)"

# ── Runtime: the pool model has to hold at run time, not just typecheck ──────
#
# Everything above is decided statically, which is exactly the gap this section
# closes. Until the pool became a runtime fact, `.toGpu()` compiled to an
# identity (the operand's value returned unchanged) and every `mlx_*` call site
# ran on the GPU stream regardless of the handle's declared pool — so a program
# could be "correctly placed" and still do all its work in one pool. A
# compile-only harness could not tell the difference.
#
# These run the binary and ask it. They are skipped when the compiler produced a
# program with no MLX backend linked in (any non-Apple platform today), because
# there is nothing to observe there — but the skip is reported, not silent.

# expect_runtime <name> <source>
#
# A pass means the program ran and every assertion in it held. A build with no
# MLX backend linked in prints "SKIP:" and exits 0 — that is reported as a skip,
# not a pass, so a Linux CI run cannot look like it verified something it never
# executed.
expect_runtime() {
  name=$1; src=$2
  printf '%s\n' "$src" > "$WORK/r.ionic"
  rm -f "$WORK/r" "$WORK/r.o"
  if ! "$COMPILER" "$WORK/r.ionic" -o "$WORK/r" >"$WORK/err.txt" 2>&1; then
    echo "FAIL  $name  (did not compile)"
    sed -n '1,6p' "$WORK/err.txt"
    FAIL=1
    return
  fi
  if "$WORK/r" >"$WORK/out.txt" 2>&1; then
    if grep -q '^SKIP:' "$WORK/out.txt"; then
      echo "skip  $name  ($(sed -n '1p' "$WORK/out.txt"))"
    else
      echo "ok    $name"
    fi
  else
    echo "FAIL  $name  (runtime assertions failed)"
    sed -n '1,40p' "$WORK/out.txt"
    FAIL=1
  fi
}

# expect_runtime_refuses <name> <needle> <source>
#
# The program must compile, and then must FAIL at run time with `needle` on its
# output. This is the Stage D backstop: the static checker exempts what it cannot
# size, so an over-budget allocation has to be caught where the bytes are
# charged. A pass here means the program did NOT run to completion — the mirror
# image of expect_runtime, and the reason both exist.
#
# The same SKIP convention applies: without a backend there is nothing to
# allocate, so the program reports the skip and exits 0.
expect_runtime_refuses() {
  name=$1; needle=$2; src=$3
  printf '%s\n' "$src" > "$WORK/r.ionic"
  rm -f "$WORK/r" "$WORK/r.o"
  if ! "$COMPILER" "$WORK/r.ionic" -o "$WORK/r" >"$WORK/err.txt" 2>&1; then
    echo "FAIL  $name  (did not compile)"
    sed -n '1,6p' "$WORK/err.txt"
    FAIL=1
    return
  fi
  "$WORK/r" >"$WORK/out.txt" 2>&1
  rc=$?
  if [ "$rc" = 0 ] && grep -q '^SKIP:' "$WORK/out.txt"; then
    echo "skip  $name  ($(sed -n '1p' "$WORK/out.txt"))"
    return
  fi
  if [ "$rc" = 0 ]; then
    echo "FAIL  $name  (ran to completion; the pool backstop did not fire)"
    sed -n '1,20p' "$WORK/out.txt"
    FAIL=1
    return
  fi
  if NO_COLOR=1 grep -q -- "$needle" "$WORK/out.txt"; then
    echo "ok    $name  (refused: $needle)"
  else
    echo "FAIL  $name  (failed, but not because of '$needle')"
    sed -n '1,20p' "$WORK/out.txt"
    FAIL=1
  fi
}

echo
echo "=== pool model at runtime (Stage C) ==="
RUNTIME_TEST='
fn check(string name, bool ok) -> int64 {
    if (ok) {
        println(str_concat("  ok   ", name));
        return 0;
    }
    println(str_concat("  FAIL ", name));
    return 1;
}

fn main() -> int64 {
    if (mlx_available() == 0) {
        println("SKIP: no MLX backend compiled in");
        return 0;
    }
    mut fails = 0;
    fails = fails + check("two pools exist", mlx_pool_count() == 2);

    let cpu_base = mlx_pool_bytes(0);
    let gpu_base = mlx_pool_bytes(1);

    // A constructor is SRAM-resident and charged there, never to VRAM.
    let a = mlx_ones([1024, 1024], 0);
    fails = fails + check("constructor is SRAM", mlx_pool(a) == 0);
    fails = fails + check("SRAM grew by 4 MiB", mlx_pool_bytes(0) - cpu_base == 4194304);
    fails = fails + check("constructor leaves VRAM alone", mlx_pool_bytes(1) == gpu_base);

    // A transfer MOVES the bytes rather than copying them under a second name.
    // "charged exactly once" is the assertion that catches double counting.
    let g = a.toGpu();
    fails = fails + check("SRAM released the bytes", mlx_pool_bytes(0) == cpu_base);
    fails = fails + check("VRAM charged exactly once", mlx_pool_bytes(1) - gpu_base == 4194304);
    gpu { fails = fails + check("transfer lands in VRAM", mlx_pool(g) == 1); }

    let c = g.toCpu();
    fails = fails + check("return lands in SRAM", mlx_pool(c) == 0);
    fails = fails + check("VRAM released the bytes", mlx_pool_bytes(1) == gpu_base);
    fails = fails + check("SRAM charged exactly once again", mlx_pool_bytes(0) - cpu_base == 4194304);

    mlx_free(c);
    fails = fails + check("free returns SRAM to baseline", mlx_pool_bytes(0) == cpu_base);

    // Pool-polymorphic builtins keep their operand pool; they do not silently
    // produce SRAM results from VRAM operands.
    let p = mlx_ones([2, 2], 0);
    let q = mlx_multiply(p, p);
    fails = fails + check("builtin result is SRAM", mlx_pool(q) == 0);
    let pg = p.toGpu();
    gpu {
        let qg = mlx_multiply(pg, pg);
        fails = fails + check("gpu builtin result is VRAM", mlx_pool(qg) == 1);
    }
    mlx_free(q);
    mlx_free(pg.toCpu());

    if (fails == 0) { return 0; }
    return 1;
}'

expect_runtime "transfers move bytes between pools, accounting once" "$RUNTIME_TEST"

echo
echo "=== handle reference counting (Stage D) ==="

# `let b = a;` names one handle twice. Before the handle carried a reference
# count, `mlx_free(a)` destroyed the struct that `b` still pointed at, so the
# next use of `b` was a use-after-free: `size(b)` returned -1 and the process
# carried on from a freed pointer. The compiler accepted that program, which is
# the worst of the three possible outcomes.
#
# The count is on the handle, not on the name, because the alternative — a
# checker rule banning `let b = a` — could not cover the two places a handle is
# really duplicated in ordinary code: passing a tensor to a function, and
# returning one. Those need the runtime answer, so they are asserted here too.
REFCNT_TEST='
fn check(string name, bool ok) -> int64 {
    if (ok) {
        println(str_concat("  ok   ", name));
        return 0;
    }
    println(str_concat("  FAIL ", name));
    return 1;
}

fn size_of(tensor@cpu t) -> int64 { return mlx_size(t); }

fn make_one() -> tensor@cpu { return mlx_zeros([3, 3], 0); }

fn main() -> int64 {
    if (mlx_available() == 0) {
        println("SKIP: no MLX backend compiled in");
        return 0;
    }
    mut fails = 0;

    // ── a copy keeps the allocation alive ──────────────────────────────────
    let base = mlx_pool_bytes(0);
    let a = mlx_zeros([2, 2], 0);
    let b = a;
    fails = fails + check("a copy charges nothing extra", mlx_pool_bytes(0) - base == 16);
    mlx_free(a);
    fails = fails + check("freeing one name keeps the bytes", mlx_pool_bytes(0) - base == 16);
    fails = fails + check("the other name is still readable", size_of(b) == 4);
    mlx_free(b);
    fails = fails + check("freeing the last name releases the bytes", mlx_pool_bytes(0) == base);

    // ── a chain of copies is still one allocation ──────────────────────────
    let c = mlx_zeros([4, 4], 0);
    let d = c;
    let e = d;
    fails = fails + check("a chain charges once", mlx_pool_bytes(0) - base == 64);
    mlx_free(c);
    mlx_free(d);
    fails = fails + check("the chain survives two frees", size_of(e) == 16);
    mlx_free(e);
    fails = fails + check("the chain releases at the end", mlx_pool_bytes(0) == base);

    // ── passing a tensor borrows it ────────────────────────────────────────
    // The callee reads the handle; it does not acquire a reference. So the
    // caller can still free afterwards, and the pool falls back to baseline.
    let p = mlx_zeros([2, 2], 0);
    fails = fails + check("a parameter sees the value", size_of(p) == 4);
    mlx_free(p);
    fails = fails + check("passing did not leak a reference", mlx_pool_bytes(0) == base);

    // ── returning a tensor moves it ────────────────────────────────────────
    // No retain on the way out and no release of the source local, so the
    // returned value arrives owning exactly the one reference it was built with.
    let r = make_one();
    fails = fails + check("a returned value is usable", mlx_size(r) == 9);
    fails = fails + check("a return charges once", mlx_pool_bytes(0) - base == 36);
    mlx_free(r);
    fails = fails + check("a returned value is released", mlx_pool_bytes(0) == base);

    if (fails == 0) { return 0; }
    return 1;
}'

expect_runtime "a copied handle outlives the name it was copied from" "$REFCNT_TEST"

echo
echo "=== the allocator backstop (Stage D) ==="

# The static check above is a necessary condition, not a sufficient one: it
# exempts every value it cannot size, and it warns while doing so. That leaves a
# hole a declaration-only pool cannot survive — the declared `vram` becomes a
# fiction and a program allocates whatever the machine will give it.
#
#     let vram = 67108864;              // 64 MiB
#     @gpu(1.0) fn grow(int64 n) -> int64 {
#         let big = mlx_zeros([n, n], 0).toGpu();   // 16 GiB when n = 65536
#     }
#
# Before the backstop this compiled (with the unsized warning) and ran to
# completion at exit 0, having allocated 16 GiB against a 64 MiB declaration.
# The allocation is refused now, at the line that made it.
#
# The program is written so that "did not fit" is distinguishable from "crashed
# for another reason": it prints the pool total first and never reaches the
# second line, so the harness sees a non-zero exit AND the runtime's own message.
BACKSTOP_TEST='
let vram = 67108864;                 // 64 MiB declared
@gpu(1.0) fn grow(int64 n) -> int64 {
    let big = mlx_zeros([n, n], 0).toGpu();   // n = 65536 -> 16 GiB
    return mlx_pool_bytes(1);
}
fn main() -> int64 {
    if (mlx_available() == 0) {
        println("SKIP: no MLX backend compiled in");
        return 0;
    }
    println(str_concat("pool capacity: ", int64_to_str(mlx_pool_capacity(1))));
    let used = grow(65536);
    println(str_concat("UNREACHED used: ", int64_to_str(used)));
    return 0;
}'

expect_runtime_refuses "an over-budget allocation is refused at run time" \
  "accelerator pool exhausted" "$BACKSTOP_TEST"

# The declaration must be *published*, not merely checked: a program that asks
# the runtime what it declared has to be told. This is what stops the checker and
# the allocator from disagreeing about the number — each keeping its own copy is
# exactly how a 64 MiB declaration ends up compared against a 24 GiB assumption.
#
# No allocation happens here, so this runs (rather than skipping) wherever MLX
# is absent too: capacity is a language-level fact and is answered on every
# platform.
CAPACITY_TEST='
let vram = 67108864;
fn main() -> int64 {
    let cap = mlx_pool_capacity(1);
    if (cap != 67108864) {
        println(str_concat("FAIL: declared 67108864, runtime reports ",
                           int64_to_str(cap)));
        return 1;
    }
    if (mlx_pool_capacity(2) != -1) {
        println("FAIL: a non-existent pool did not report -1");
        return 1;
    }
    println("ok: the declared pool size is readable at run time");
    return 0;
}'

expect_runtime "the declared pool size is published to the runtime" "$CAPACITY_TEST"

# The other half of the comparison: what the machine actually offers. A
# declaration is only meaningful against a real number, and before this the
# runtime could only ever report what the program *asked* for.
#
# The assertions are deliberately shape-only rather than numeric — the host's
# RAM and the GPU's working set differ on every machine, so pinning either would
# make the suite pass only on the author's laptop. What must hold everywhere:
#
#   * SRAM has a positive capacity (physical RAM is a fact of every host, and
#     the value is answered even where no accelerator exists);
#   * VRAM is non-negative — 0 means "unknown", which is the honest answer on a
#     build without a backend, so it is allowed but a negative is not;
#   * a pool id that does not exist reports 0, not a stray number.
DEVICE_TEST='
let vram = 67108864;
fn main() -> int64 {
    let sram = mlx_device_capacity(0);
    if (sram <= 0) {
        println(str_concat("FAIL: host memory reported as ", int64_to_str(sram)));
        return 1;
    }
    let vram_cap = mlx_device_capacity(1);
    if (vram_cap < 0) {
        println(str_concat("FAIL: VRAM capacity reported as ", int64_to_str(vram_cap)));
        return 1;
    }
    if (mlx_device_capacity(2) != 0) {
        println("FAIL: a non-existent pool reported a device capacity");
        return 1;
    }
    println(str_concat("ok: device capacities readable (SRAM ",
                       int64_to_str(sram)));
    return 0;
}'

expect_runtime "the device capacity is readable alongside the declaration" "$DEVICE_TEST"

echo
if [ "$FAIL" = 0 ]; then
  echo "RESULT: PASS"
else
  echo "RESULT: FAIL"
fi
exit "$FAIL"
