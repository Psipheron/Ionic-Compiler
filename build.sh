#!/bin/sh
# Build the Ionic compiler from the split source tree.
#
# Usage:
#   ./build.sh                       — build ionic_new using ionic_self
#                                       (the committed bootstrap binary that
#                                        ships with the repo). Works on any
#                                        OS without any toolchain installed.
#   ./build.sh --verify-self-hosting — build ionic_new, then confirm that
#                                       ./ionic_self and the freshly-built
#                                       self-hosted compiler emit an identical
#                                       object file for the same sources.
#                                       Compares .o (not the linked binary,
#                                       which the linker always stamps with a
#                                       fresh UUID/signature). Exits 0 on
#                                       match, 1 otherwise.
#   IONIC=./some_binary ./build.sh   — use a specific Ionic binary instead of
#                                       the default ./ionic_self
#
# End users install the released tarball (./ionic + lib/std). To rebuild from
# source they just need ./ionic_self from the same tarball; no toolchain
# required.
#
# The Rust source tree (src/*.rs + Cargo.toml) is for the dev-only Rust
# compiler. It's used by CI as a cross-OS bootstrap — `cargo build --release`
# produces target/release/ionic which can compile user programs on any OS
# without needing the committed ionic_self. But target/release/ionic is
# missing bitwise operator support (it predates that addition), so it
# CANNOT compile the split .ionic sources themselves — use ionic_self for
# that. End users don't need Rust or cargo; releases ship self-hosted.

set -e

IONIC="${IONIC:-./ionic_self}"
OUT="${OUT:-ionic_new}"

# ── Optional ML backends ────────────────────────────────────────────────────
# The compiler binary itself never links ML libraries — those are linked into
# the *programs* it produces, and only when a program's source references them
# (see the usage gate in src/main.ionic). So there is nothing to configure here
# for a normal build.
#
# What you may want on a machine with a half-installed mlx-c is to force the
# backend off when compiling a program:
#
#   IONIC_NO_MLX=1 ./ionic_new model.ionic -o model
#
# or point at a non-Homebrew prefix:
#
#   IONIC_MLX_INCLUDE=/opt/mlx/include IONIC_MLX_LIB=/opt/mlx/lib \
#       ./ionic_new model.ionic -o model
#
# These overrides are read by the self-hosted compiler (src/main.ionic,
# probe_dir/probe_dir_any). The dev-only Rust bootstrap in src/main.rs has its
# own hardcoded candidate list and does NOT consult them — it only understands
# IONIC_NO_MLX=1.

# Single entry point. Its `import "..."` statements pull in the rest of the
# tree (resolved relative to src/) and splice the modules in dependency order,
# so the source order lives in the code instead of here in the build script.
SOURCES="src/main.ionic"

# ── Mode: build then verify self-hosting ────────────────────────────────────
# The compiler writes its object next to the output as "<out>.o", then shells
# out to `clang` to link it. Comparing the *linked binaries* can never match:
# the linker stamps each output with a fresh LC_UUID and an ad-hoc
# LC_CODE_SIGNATURE, so byte-identity is impossible even for identical input.
# We therefore compare the compiler-generated .o files, which ARE deterministic
# and are exactly what this project owns.
#
#   1. $IONIC compiles the sources → capture its .o as A
#   2. $OUT (freshly built) compiles the same sources → capture its .o as B
#   3. A must equal B: the bootstrap and the self-hosted compiler agree.
if [ "$1" = "--verify-self-hosting" ]; then
    echo "==> Building $OUT from split source using $IONIC..."
    $IONIC $SOURCES -o "$OUT"
    cp "$OUT.o" /tmp/_ionic_verify_a
    echo "==> Verifying self-hosting: $OUT emits an identical object file..."
    OUT2="/tmp/_ionic_verify_b.bin"
    ./"$OUT" $SOURCES -o "$OUT2"
    cp "$OUT2.o" /tmp/_ionic_verify_b
    if cmp -s /tmp/_ionic_verify_a /tmp/_ionic_verify_b; then
        echo "    ✓ object files byte-identical ($(wc -c < /tmp/_ionic_verify_a | tr -d ' ') bytes)"
    else
        echo "    ✗ DIFFER — self-hosting is BROKEN"
        echo "    Run: diff <(xxd /tmp/_ionic_verify_a) <(xxd /tmp/_ionic_verify_b) | head"
        exit 1
    fi
    echo "==> Done: $OUT"
    exit 0
fi

# ── Default: build via existing ionic (self-hosted loop) ────────────────────
echo "==> Compiling $OUT from split source using $IONIC..."
$IONIC $SOURCES -o "$OUT"
echo "==> Done: $OUT"
