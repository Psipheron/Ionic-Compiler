#!/bin/sh
# Array reference-counting probe gate (memory-model §11.4, stage 2).
#
# Stage 1 of array RC routed the header through `ionic_rc_alloc` and emitted no
# retain/release. Stage 2 emits both. That is where the risk lives: a missing
# retain hands the caller a pointer into freed memory, and a missing release
# leaks — but a *misplaced* release frees memory that is still live, and the
# symptom is a heap-use-after-free that a plain build reports as "mostly
# worked".
#
# Every case here is compiled with the real compiler and linked against the real
# runtime under AddressSanitizer and UndefinedBehaviorSanitizer, so a failure
# is a real memory error in the emitted code, not a model of one.
#
# The cases and what each one pins down:
#
#   alias_return        let b = a; return b — the copy must retain, or the
#                       exit block frees `a` under the caller's `b`.
#   global_alias        let a = g — a global's array is a borrow; the local
#                       must not release it.
#   borrowed_reassign   a slot that held a callee's borrow is reassigned over;
#                       the borrowed value must not be released.
#   escape_loop         let row = ...; outer.push(row) inside a loop — the
#                       re-entry release must not free the value `outer` holds.
#   escape_alias        a retained alias escapes into a container; the original
#                       still holds a count.
#   escape_conditional_reassign
#                       a value escapes into a container and is then
#                       conditionally reassigned — the assignment path must not
#                       free the container's value on the path where the
#                       assignment did not run. This is the case the escape bug
#                       was found in.
#   global_store_array  a local array stored into a global; the global outlives
#                       the frame, so it needs a count of its own.
#   borrow_slot_copy    a borrowed array slot, conditionally reassigned.
#   string_escape       the string half of the escape rule.
#   string_escape_conditional
#                       the string half, with the escape and the reassignment on
#                       opposite branches.
#   loop_reentry        a `let` inside a loop that never escapes — the
#                       re-entry release must reclaim the previous iteration.
#   fn_push_string      the *function* form `push(a, v)`, which reaches codegen
#                       through ncg_call rather than the ND_METHOD_CALL path the
#                       `.push()` method uses.
#   array_lit_string    a string array-literal element taken from an owning
#                       local. An array literal emits a raw `bl`, never through
#                       ncg_call, so it needs its own retain.
#   struct_field_string a struct-literal field fed by an owning local.
#   struct_assign_string
#                       `b.val = s`, a *different* emit site from the struct
#                       literal (ND_ASSIGN vs ND_STRUCT_LIT).
#   cyclic_array        an array that contains itself. The leak is accepted
#                       (docs/memory-model.md §6.1), so this asserts *safety*
#                       rather than reclamation: the cyclic graph must stay
#                       readable, and nothing may be freed while it is
#                       reachable.
#
# Every escape probe carries a *conditional reassignment after the escape*, and
# that is load-bearing rather than incidental. Dropping the claim without
# retaining only leaks — the block is never freed, so nothing that reads it
# crashes, and a probe without the reassignment passes against the buggy
# compiler. The use-after-free needs an assignment that re-raises a claim over a
# slot still holding the escaped value on the path where the branch did not run.
# All four escape probes here were verified to fail under ASan when
# ncg_escape_retain is reverted to the old drop-the-claim body.
#
# Leak detection is off: the accepted leak for a container escape is a
# *decision*, and a leak report would flag it as a bug. That is also why the
# probes above must catch the use-after-free rather than the leak — the harness
# cannot see the leak at all.
#
# Usage:
#   scripts/arr_rc_check.sh
set -u
case "$0" in
  */*) cd "$(dirname "$0")/.." || exit 1 ;;
  *)   cd . || exit 1 ;;
esac

CC="${CC:-clang}"
IONIC="${IONIC:-./ionic_new}"
RUNTIME=src/ionic_model_runtime.c
PROBE_DIR=scripts/probes
WORK="${TMPDIR:-/tmp}/ionic_arr_rc_probe"

if [ ! -x "$IONIC" ]; then
    echo "arr_rc_check: no compiler at '$IONIC'; set IONIC= to override" >&2
    exit 2
fi
if ! command -v "$CC" >/dev/null 2>&1; then
    echo "arr_rc_check: no C compiler ('$CC'); set CC= to override" >&2
    exit 2
fi

rm -rf "$WORK"
mkdir -p "$WORK" || exit 2

echo "==> Array RC probes (ASan + UBSan)"

fail=0
for src in "$PROBE_DIR"/*.ionic; do
    name=$(basename "$src" .ionic)
    out="$WORK/$name"

    # Compile with the real compiler. The compiler links the runtime itself, so
    # the object is the compiler's own output and the runtime is the shipped one.
    if ! "$IONIC" "$src" -o "$out" > "$WORK/$name.compile.log" 2>&1; then
        echo "    ✗ FAIL: $name did not compile"
        sed 's/^/        /' "$WORK/$name.compile.log"
        fail=1
        continue
    fi

    # Relink against the runtime under the sanitizers. The compiler's own link
    # has no sanitizer flags, so the object is reused and only the link differs.
    if ! "$CC" -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer \
         "$out.o" "$RUNTIME" -o "$out.asan" -lm 2> "$WORK/$name.link.log"; then
        echo "    ✗ FAIL: $name did not link under the sanitizers"
        sed 's/^/        /' "$WORK/$name.link.log"
        fail=1
        continue
    fi

    if ! ASAN_OPTIONS=detect_leaks=0 UBSAN_OPTIONS=halt_on_error=1 \
         "$out.asan" > "$WORK/$name.run.log" 2>&1; then
        echo "    ✗ FAIL: $name reported a memory error"
        sed 's/^/        /' "$WORK/$name.run.log" | head -30
        fail=1
        continue
    fi

    echo "    ✓ $name: $(tr '\n' ' ' < "$WORK/$name.run.log")"
done

if [ "$fail" -ne 0 ]; then
    echo "    ✗ FAIL: array RC is not memory-safe (see above)."
    exit 1
fi

echo "    ✓ PASS: every array aliasing case is memory-safe."
exit 0
