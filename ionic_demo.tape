# Ionic — cinematic demo
#
# A four-act recording that shows the two things Ionic actually claims:
# it is fast, and its type system catches things other languages find at
# run time (placement, moves, tensor shapes).
#
# Every number below was measured on the machine that recorded it, not
# invented. To re-measure after a change:
#
#   ./build.sh                          # act 1
#   ./ionic_new examples/hello.ionic -o /tmp/h
#   ./build.sh --verify-self-hosting    # act 2
#   ./ionic_new examples/mlx_matmul.ionic -o /tmp/m && /tmp/m
#   ./ionic_new examples/mastery/fail/mixed_pool.ionic -o /tmp/x
#
# Requires: vhs, bash, and a built ./ionic_new (act 1 builds it).
# Render with:  vhs ionic_demo.tape
#
# Timing is deliberately uneven. A demo where every beat takes the same
# time reads as a slideshow; the pauses are where the reader is meant to
# look at the screen.

Output assets/ionic_demo.gif

Require bash

# ── Presentation ────────────────────────────────────────────────────────────
# Wide and short, so command output sits on one screen without scrolling
# mid-beat. 1400x760 is the smallest size where the compiler's diagnostic
# (file:line:col + source line + caret + hint) fits without wrapping, and
# that diagnostic is the visual punchline of act 3.
Set Shell "bash"
Set FontSize 26
Set FontFamily "JetBrains Mono"
Set Width 1400
Set Height 760
Set Padding 24
Set MarginFill "#0b0e14"
Set Margin 40
Set BorderRadius 10
Set WindowBar Colorful
Set WindowBarSize 34
Set Theme "Catppuccin Mocha"
Set TypingSpeed 45ms

# 15fps, not the 25 default. A terminal demo is mostly-static text: the only
# motion is the caret and the typed characters, and 15fps renders both
# smoothly while cutting the frame count (and so the GIF) by ~40%. At 25fps
# this tape produced a 2.4 MB file, which is a lot to load in a README.
Set Framerate 15

# A slow, deliberate cursor. The default blinks fast enough to look nervous
# at 26pt.
Set CursorBlink true

# ── Cold open ───────────────────────────────────────────────────────────────
Hide
Type "clear"
Enter
Show
Sleep 600ms

Type "# Ionic — a self-hosting compiler for native ARM64."
Enter
Sleep 400ms
Type "# No LLVM at runtime. No garbage collector. No interpreter."
Enter
Sleep 900ms

# ── Act 1 — speed ───────────────────────────────────────────────────────────
# The compiler is written in the language it compiles, so this single
# command is the whole bootstrap: 4122 lines of runtime C aside, every
# stage — lex, parse, check, optimise, emit ARM64 — is Ionic.
Type "#"
Space
Type "Act 1 — it builds itself"
Enter
Sleep 700ms

Type "time ./build.sh"
Enter
Sleep 900ms

# ── Act 2 — the self-hosting proof ──────────────────────────────────────────
Sleep 1200ms
Type "#"
Space
Type "Act 2 — and the result is bit-identical"
Enter
Sleep 600ms
Type "./build.sh --verify-self-hosting"
Enter
Sleep 1000ms

# This is the claim that matters: the committed bootstrap and the compiler
# it just produced emit the SAME object file from the same sources. Not
# "similar" — byte-identical, 745772 of them.

# ── Act 3 — the type system does the work ───────────────────────────────────
Sleep 1600ms
Type "#"
Space
Type "Act 3 — placement is a type, not a convention"
Enter
Sleep 600ms

Type "./ionic_new examples/mlx_matmul.ionic -o /tmp/mm && /tmp/mm"
Enter
Sleep 1200ms

Sleep 1400ms
Type "#"
Space
Type "A tensor in the wrong pool does not compile:"
Enter
Sleep 500ms

Type "./ionic_new examples/mastery/fail/mixed_pool.ionic -o /tmp/x"
Enter
Sleep 1400ms

# ── Act 4 — the errors are the feature ──────────────────────────────────────
Sleep 1800ms
Type "#"
Space
Type "Neither does a shape that cannot multiply, or a value used after a move:"
Enter
Sleep 500ms

Type "./ionic_new examples/mastery/fail/matmul_shape.ionic -o /tmp/y"
Enter
Sleep 1600ms
Type "./ionic_new examples/mastery/fail/use_after_move.ionic -o /tmp/z"
Enter
Sleep 1800ms

# ── Close ───────────────────────────────────────────────────────────────────
Type "#"
Space
Type "Four diagnostics that a Python or C++ program would meet at 2am in production."
Enter
Sleep 1200ms
Type "#"
Space
Type "Ionic meets them at compile time."
Enter
Sleep 1600ms
