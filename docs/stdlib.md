# Ionic Standard Library

Import any module with a wildcard (`.*`) for selective compilation, or name specific symbols with `{...}`.

```ionic
import std.math.*;
import std.str.{contains, trim};
import std.array.*;
import std.io.*;
```

Only the symbols you actually use are compiled into the output binary.

---

## `std.math`

Mathematical constants and utility functions.

### Constants

| Name | Value | Description |
|------|-------|-------------|
| `PI` | `3.14159265358979...` | π |
| `TAU` | `6.28318530717958...` | 2π — full turn in radians |
| `E` | `2.71828182845904...` | Euler's number |

### Functions

#### `clamp(float64 x, float64 lo, float64 hi) -> float64`
Clamps `x` to the closed interval `[lo, hi]`.

```ionic
import std.math.*;
let v = clamp(1.5, 0.0, 1.0);  // 1.0
```

#### `lerp(float64 a, float64 b, float64 t) -> float64`
Linear interpolation: `a + t*(b-a)`. `t=0` returns `a`, `t=1` returns `b`.

#### `sign(float64 x) -> float64`
Returns `-1.0`, `0.0`, or `1.0`.

#### `min_f(float64 a, float64 b) -> float64`
Returns the smaller value.

#### `max_f(float64 a, float64 b) -> float64`
Returns the larger value.

#### `degrees(float64 r) -> float64`
Converts radians to degrees.

#### `radians(float64 d) -> float64`
Converts degrees to radians.

#### `pow_i(float64 base, int64 exp) -> float64`
Raises `base` to an integer power `exp`.

---

## `std.str`

String utilities. All functions are pure (return new strings, never modify in place).

#### `contains(string s, string sub) -> bool`
Returns `true` if `s` contains `sub` as a substring.

```ionic
import std.str.*;
let found = contains("hello world", "world");  // true
```

#### `starts_with(string s, string prefix) -> bool`
Returns `true` if `s` begins with `prefix`.

#### `ends_with(string s, string suffix) -> bool`
Returns `true` if `s` ends with `suffix`.

#### `repeat(string s, int64 n) -> string`
Returns `s` repeated `n` times.

```ionic
let sep = repeat("-", 40);  // "----------------------------------------"
```

#### `count_char(string s, int64 ch) -> int64`
Counts occurrences of ASCII character code `ch` in `s`.

```ionic
let dots = count_char("1.2.3", 46);  // 46 = '.', result = 2
```

#### `pad_left(string s, int64 width, string pad_ch) -> string`
Left-pads `s` with `pad_ch` until at least `width` characters wide.

```ionic
let padded = pad_left("42", 6, " ");  // "    42"
```

#### `pad_right(string s, int64 width, string pad_ch) -> string`
Right-pads `s` with `pad_ch`.

#### `trim_left(string s) -> string`
Removes leading ASCII spaces.

#### `trim_right(string s) -> string`
Removes trailing ASCII spaces.

#### `trim(string s) -> string`
Removes leading and trailing ASCII spaces.

#### `str_reverse(string s) -> string`
Returns `s` with characters in reversed order.

---

## `std.array`

Array utilities for `[int64]` and `[float64]` arrays.

#### `arr_sum([int64] a) -> int64`
Sum of all elements.

#### `arr_sum_f([float64] a) -> float64`
Sum of all float64 elements.

#### `arr_mean([float64] a) -> float64`
Arithmetic mean. Returns `0.0` for empty arrays.

#### `arr_max([int64] a) -> int64`
Maximum element. Assumes non-empty array.

#### `arr_min([int64] a) -> int64`
Minimum element. Assumes non-empty array.

#### `arr_contains([int64] a, int64 x) -> bool`
Returns `true` if `x` appears anywhere in `a`.

#### `arr_fill([int64] a, int64 val) -> int64`
Sets every element to `val` in-place.

#### `arr_reverse([int64] a) -> [int64]`
Returns a new array with elements in reversed order.

#### `arr_index_of([int64] a, int64 x) -> int64`
Returns the first index of `x`, or `-1` if not found.

#### `arr_copy([int64] a) -> [int64]`
Returns a shallow copy.

---

## `std.io`

I/O utilities beyond `print` and `file_write`.

#### `readline() -> string`
Reads one line from stdin (up to 4096 bytes). Strips the trailing newline.

```ionic
import std.io.*;
let line = readline();
print(str_concat("You typed: ", line));
```

#### `readline_n(int64 max_bytes) -> string`
Like `readline` but with a custom buffer size.

#### `eprint(string s) -> int64`
Prints `s` to stderr without a newline.

#### `eprintln(string s) -> int64`
Prints `s` to stderr followed by a newline.

#### `print_hr(int64 width) -> int64`
Prints a horizontal rule of `width` dashes to stdout.

---

## Terminal TUI framework (`lepton`)

The alternate-screen TUI framework is no longer part of the stdlib. It lives in
the sibling **`lepton`** project and is consumed by path import:

```ionic
import "../lepton/src/screen.ionic";
```

The `term_*` runtime builtins it depends on (raw mode, alternate screen, key
decoding) remain part of the compiler's C runtime.

This is only possible if you have the lepton repository installed on your computer, next to the project.

---

## Selective compilation — how it works

When you write:

```ionic
import std.math.*;
```

and only call `clamp` in your program, the compiler:

1. Parses all of `lib/std/math.ionic` into an in-memory pool.
2. Walks your program's AST and finds that `clamp` is referenced.
3. Walks `clamp`'s body — it references nothing from the library, so BFS stops.
4. Only `clamp` (plus the constant `PI`, `TAU`, `E` if referenced) is emitted to IR.
5. `lerp`, `sign`, `min_f`, `max_f`, `degrees`, `radians`, `pow_i` are **not compiled**.

The result is zero dead-code overhead from large import wildcards.
