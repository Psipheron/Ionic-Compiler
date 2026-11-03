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
#   tensor parameters / unsized results -> exempt (never a false rejection)
#   no declared pool                    -> no budget in force
#   using a transferred value           -> rejected (B5)
#   builtins borrow / temporaries move  -> compiles (B5)
#   mastery/04_placement_types.ionic    -> still compiles (the tutorial's reference)
#   mastery/04_pool_polymorphism.ionic  -> still compiles (transfers, done correctly)
#   mastery/fail/use_after_move.ionic   -> rejected (the tutorial's fail-case)
#
# Everything here is decided statically, so it runs on any OS with no
# accelerator present. That is the point: the checker's behaviour is portable
# even when the backend is not.
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
expect_ok "unsized results are exempt, not rejected" \
'let vram = 1073741824;
@gpu(0.01) fn reshapes(tensor@gpu x) -> tensor@gpu {
    let r = mlx_reshape(x, [4, 4]);
    return r;
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
if [ "$FAIL" = 0 ]; then
  echo "RESULT: PASS"
else
  echo "RESULT: FAIL"
fi
exit "$FAIL"
