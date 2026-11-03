#!/bin/sh
# Regression harness for model RAII ownership in the native ARM64 codegen.
#
# Asserts how many times the generated object references ionic_model_free for
# a given source shape. The correct counts are:
#
#   load_model, no explicit model_free    -> 1   (scope-exit destructor only)
#   load_model + explicit model_free      -> 1   (explicit only; implicit suppressed)
#   two independent models, no model_free -> 2
#   `return m` from a fn that owns it     -> 0   (ownership transferred to caller)
#
# Historically the first two shapes emitted 2 and 3 respectively, because the
# destructor was emitted at both the scope-exit path and the fall-through
# epilogue. That is a double/triple free at runtime.
#
# Note that the count is what the *object* references, not what runs: a function
# with several `return`s still emits one destructor, in the shared exit block.
# The one place that breaks down is transferring a different model from each of
# two branches — see the KNOWN LIMITATION note on ND_RETURN in
# src/codegen/native.ionic. That shape is not asserted here because it is a
# documented gap, not a guarantee.
#
# The runtime section also pins down *return values* from functions that own a
# model: `ionic_model_free` takes the handle in X0 and clobbers every
# caller-saved register, so the returning value has to be parked in the frame
# across the destructor. Without that the caller sees a heap address instead of
# the value it returned.
set -u
cd /Users/henry/Developer/Ionic || exit 1

COMPILER="${COMPILER:-./ionic_new}"
WORK=/tmp/ionic_raii
mkdir -p "$WORK"
FAIL=0

# count_frees <name> <expected> <body> [preamble]
count_frees() {
  name=$1; expected=$2; body=$3; pre=${4:-}
  printf '%s\nfn main() -> int64 {\n%s\n    return 0;\n}\n' "$pre" "$body" > "$WORK/t.ionic"
  rm -f "$WORK/t" "$WORK/t.o"
  if ! "$COMPILER" "$WORK/t.ionic" -o "$WORK/t" >"$WORK/err.txt" 2>&1; then
    echo "COMPILE-FAIL  $name"
    sed -n '1,5p' "$WORK/err.txt"
    FAIL=1
    return
  fi
  actual=$(nm -u "$WORK/t.o" | grep -c '_ionic_model_free')
  if [ "$actual" = "$expected" ]; then
    echo "ok    $name  (frees=$actual)"
  else
    echo "FAIL  $name  (expected=$expected, got=$actual)"
    FAIL=1
  fi
}

echo "=== model ownership: reference counts ==="
count_frees "implicit RAII only"      1 '    let m = load_model("x.onnx");
    println("loaded");'

count_frees "explicit model_free"     1 '    let m = load_model("x.onnx");
    model_free(m);'

count_frees "two models, implicit"    2 '    let a = load_model("a.onnx");
    let b = load_model("b.onnx");
    println("two");'

count_frees "no model at all"         0 '    println("hi");'

# `give` returns its model, which transfers ownership to the caller. The callee
# therefore emits no destructor, and the only ionic_model_free in the program is
# the caller's own — the count of 1 proves the transfer suppressed the implicit
# free rather than merely moving it.
count_frees "returned model transferred" 1 '    let t = give("x.onnx");
    println("got");
    model_free(t);' 'fn give(string p) -> model {
    let m = load_model(p);
    return m;
}'

echo
echo "=== runtime behaviour (real ONNX model) ==="
run_case() {
  name=$1; expected_exit=$2; body=$3
  printf 'fn main() -> int64 {\n%s\n    return 0;\n}\n' "$body" > "$WORK/r.ionic"
  rm -f "$WORK/r"
  if ! "$COMPILER" "$WORK/r.ionic" -o "$WORK/r" >"$WORK/e2.txt" 2>&1; then
    echo "COMPILE-FAIL  $name"
    FAIL=1
    return
  fi
  "$WORK/r" >/dev/null 2>&1
  rc=$?
  if [ "$rc" = "$expected_exit" ]; then
    echo "ok    $name  (exit=$rc)"
  else
    echo "FAIL  $name  (expected exit=$expected_exit, got=$rc)"
    FAIL=1
  fi
}

if [ -f models/en_US-amy-low.onnx ]; then
  M=models/en_US-amy-low.onnx
  run_case "implicit RAII runs clean" 0 "    let m = load_model(\"$M\");
    println(\"loaded\");"
  run_case "explicit model_free runs clean" 0 "    let m = load_model(\"$M\");
    model_free(m);
    println(\"freed\");"
else
  echo "skip  runtime cases (models/en_US-amy-low.onnx not present)"
fi

echo
echo "=== return value preserved across the destructor ==="
# The destructor passes the model handle in X0 and ionic_model_free clobbers
# every caller-saved register, so a `return expr` in a function that owns a
# model must spill the value before the free and reload it after. Before that
# fix these printed heap addresses instead of 42 / 7.
if [ -f models/en_US-amy-low.onnx ]; then
  M=models/en_US-amy-low.onnx
  cat > "$WORK/v.ionic" <<EOF
fn early(string path, int64 n) -> int64 {
    let m = load_model(path);
    if (n > 0) { return 42; }
    return 7;
}

fn fsum(string path, int64 n) -> float64 {
    let m = load_model(path);
    if (n > 0) { return 1.5; }
    return 2.5;
}

fn two(string path, int64 n) -> int64 {
    let a = load_model(path);
    let b = load_model(path);
    if (n > 0) { return 11; }
    return 22;
}

fn main() -> int64 {
    println(early("$M", 1));
    println(early("$M", 0));
    println(fsum("$M", 1));
    println(fsum("$M", 0));
    println(two("$M", 1));
    println(two("$M", 0));
    return 0;
}
EOF
  if ! "$COMPILER" "$WORK/v.ionic" -o "$WORK/v" >"$WORK/v.err" 2>&1; then
    echo "COMPILE-FAIL  return values"
    FAIL=1
  else
    "$WORK/v" 2>/dev/null | grep -v '^\[ionic' > "$WORK/v.out"
    want="42
7
1.5
2.5
11
22"
    if [ "$(cat "$WORK/v.out")" = "$want" ]; then
      echo "ok    return values survive the destructor"
    else
      echo "FAIL  return values survive the destructor"
      echo "--- expected ---"; printf '%s\n' "$want"
      echo "--- actual ---"; cat "$WORK/v.out"
      FAIL=1
    fi
  fi
else
  echo "skip  return-value case (models/en_US-amy-low.onnx not present)"
fi

echo
if [ "$FAIL" = 0 ]; then
  echo "RESULT: PASS"
else
  echo "RESULT: FAIL"
fi
exit "$FAIL"
