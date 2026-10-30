# Ionic Language Reference

Ionic is a statically-typed, curly-brace language for data science and AI workloads. It compiles to native ARM64 machine code (direct Mach-O emission; an LLVM path exists as a legacy/debug backend) and supports explicit hardware placement (`@cpu` / `@gpu`).

---

## Types

| Type | Description |
|------|-------------|
| `int64` | 64-bit signed integer |
| `float64` | 64-bit IEEE 754 double |
| `bool` | Boolean (`true` / `false`) |
| `string` | UTF-8 string (immutable value) |
| `tensor@cpu` | CPU-resident tensor |
| `tensor@gpu` | GPU-resident tensor |
| `model` | Opaque ML model handle |
| `[T]` | Dynamic array of element type T |
| `void` | No return value |
| `StructName` | User-defined struct |

---

## Variables

```ionic
let x = 42;               // immutable int64 (type inferred)
mut y = 3.14;             // mutable float64
let name: string = "hi";  // explicit type annotation
```

`let` variables cannot be reassigned. `mut` variables can.

---

## Functions

```ionic
fn add(int64 a, int64 b) -> int64 {
    return a + b;
}

fn greet(string name) -> string {
    return str_concat("Hello, ", name);
}
```

Parameters are typed before their name. Return type follows `->`. Recursion is fully supported.

---

## Control flow

```ionic
if (x > 0) {
    print("positive");
} else {
    print("non-positive");
}

while (x > 0) {
    x = x - 1;
}

for (i in 0..10) {        // exclusive upper bound
    print(int64_to_str(i));
}

break;     // exits the nearest loop
continue;  // skips to the next iteration
```

---

## Structs

```ionic
struct Point { x: float64, y: float64 }

let p = Point { x: 1.0, y: 2.0 };
print(float64_to_str(p.x));
p.y = 3.0;               // field assignment
```

Structs are heap-allocated. All fields are mutable regardless of whether the binding is `let` or `mut`.

---

## Arrays

```ionic
mut nums = [1, 2, 3];     // [int64]
nums.push(4);
let n = nums.len;          // int64
let first = nums[0];       // element access
nums[0] = 99;              // element assignment
```

Arrays grow dynamically. Element type is inferred from the first literal.

---

## Hardware placement

```ionic
@gpu fn train_step(...) -> ... { ... }   // run on GPU
@cpu fn load_data(...) -> ...  { ... }   // run on CPU

@gpu(0.5) fn partial(...) -> ... { ... } // use 50% of GPU

let weights: tensor@gpu = ...;
gpu {
    // Only tensor@gpu variables and compute are allowed here.
    // CPU I/O (print, file_read, file_write) is forbidden.
}
```

An `@gpu fn` body is itself a GPU context: it is checked with the same rules
as a `gpu { ... }` block. Inside it,

- `tensor@cpu` values are rejected (use `.toGpu()` first),
- CPU-only I/O (`print`, `file_read`, `file_write`, …) is rejected,
- calling an `@cpu`-annotated function is rejected.

A parameter annotated with a placement that contradicts its function
(`@gpu fn f(@cpu tensor@cpu x)`) is also rejected — no call site could ever
satisfy it. `@cpu fn` bodies and unannotated functions run in CPU context.

`.toGpu()` and `.toCpu()` convert tensors between hardware domains.

---

## Imports

```ionic
import std.math.*;                    // glob — all symbols, only used ones compiled
import std.str.{contains, trim};      // named — exactly these two symbols
import std.io;                        // module — all symbols (same as glob for now)
```

Dotted paths resolve against project-local packages (`.ionic/packages/`),
then user-central (`$HOME/.ionic/packages/`), then the bundled stdlib
(`lib/`, only for the `std.` prefix).

String-literal (path) imports resolve against the directory of the importing
file — the mechanism the compiler uses to assemble its own source tree from a
single entry point:

```ionic
import "imports.ionic";            // same directory as this file
import "lexer/tokens.ionic";       // subdirectory
import "../diagnostics.ionic";     // parent directory
```

`.` and `..` are collapsed during resolution so a module reached by two
different paths is loaded exactly once.

**Selective compilation:** when using wildcards, the compiler walks the AST and determines which library symbols are actually referenced — including transitive dependencies — and only emits IR for those. Unused library functions add zero overhead.

---

## Doc comments

```ionic
/// Returns the absolute value of x.
fn abs(float64 x) -> float64 {
    if (x < 0.0) { return -x; }
    return x;
}
```

`///` lines immediately before a `fn` or `struct` definition are attached to that definition as its documentation, so they travel with it in tooling that reads the AST.

---

## Built-in functions

### Printing
| Function | Description |
|----------|-------------|
| `print(string s)` | Print to stdout with newline |

### String operations
| Function | Description |
|----------|-------------|
| `str_concat(string a, string b) -> string` | Concatenate two strings |
| `str_len(string s) -> int64` | Length in bytes |
| `str_eq(string a, string b) -> bool` | Equality test |
| `str_index(string s, int64 i) -> int64` | ASCII code of character at index i |
| `int64_to_str(int64 n) -> string` | Format integer as string |
| `float64_to_str(float64 x) -> string` | Format float as string |
| `char_to_str(int64 code) -> string` | One-character string from a Unicode codepoint (UTF-8 encoded) |

String literals support the escapes `\n`, `\t`, `\r`, `\"`, `\\`, `\e` (ESC,
0x1B), and `\xHH` (a raw byte from two hex digits):

```ionic
print("\e[31mred\e[0m");    // ANSI color
print("\x41\x42");           // "AB"
```

`char_to_str` takes a Unicode codepoint and returns its UTF-8 encoding, so
non-ASCII glyphs work: `char_to_str(0x2500)` is a horizontal bar, and
`char_to_str(0x250C)` is a top-left corner. The `lepton` project (a sibling of this repo) builds a
terminal UI framework on these.

### Conversion
| Function | Description |
|----------|-------------|
| `int64_to_float64(int64 n) -> float64` | Widen integer to float |
| `float64_to_int64(float64 x) -> int64` | Truncate float to integer |

### Math
| Function | Description |
|----------|-------------|
| `sqrt(float64 x) -> float64` | Square root |
| `abs(float64 x) -> float64` | Absolute value |

### File I/O
| Function | Description |
|----------|-------------|
| `file_read(string path) -> string` | Read entire file |
| `file_write(string path, string content)` | Write/overwrite file |
| `file_exists(string path) -> bool` | Check file existence |

### System
| Function | Description |
|----------|-------------|
| `get_arg(int64 n) -> string` | Command-line argument (0-indexed after program name) |
| `cpu_core_count() -> int64` | Number of logical CPU cores |

### ML model
| Function | Description |
|----------|-------------|
| `load_model(string path) -> model` | Load ONNX or GGUF model |
| `model_free(model m)` | Release model resources |
| `gguf_generate(model m, string prompt, int64 max_tokens) -> string` | LLM text generation |
| `gguf_set_temp(model m, float64 temp)` | Set sampling temperature |
| `gguf_set_top_p(model m, float64 p)` | Set nucleus sampling threshold |
| `piper_forward(model m, int64[] phonemes, float64 ns, float64 ls, float64 nw) -> float64[]` | Piper TTS inference |
| `write_wav(string path, float64[] samples, int64 n, int64 rate)` | Write PCM WAV file |

### MLX tensor backend

Apple-silicon GPU array runtime. Handles are typed `tensor@cpu` — they are
host-side handles to a lazily-built device graph, so `.toGpu()` is a no-op
transfer and is only needed to enter a `gpu { }` context. Unlike `load_model`,
**handles are not RAII-freed**: call `mlx_free()` explicitly.

Backend selection is automatic and per-program: the compiler links `mlx-c` only
when the source mentions `mlx_*`. On Intel Macs and Linux every function below
still exists, so a program links — it reports "MLX backend not compiled in" at
runtime. GGUF/llama.cpp remains the portable path.

The native ARM64 codegen passes `float64` as an IEEE-754 bit pattern in a
general-purpose register, not in `D0`-`D7`. Every `ionic_mlx_*` entry point is
therefore declared with `int64_t` in place of `double` for float parameters and
converts with `i642d()`/`d2i64()`. This is invisible from Ionic — you still pass
and receive ordinary `float64` values — but it matters if you ever add a new
entry point: a `double` parameter would read the wrong register bank and receive
garbage.

Construction (shapes are `[int64]`; dtype is `0`=f32, `1`=f16, `2`=bf16,
`3`=f64, `4`=i32, `5`=i64, `6`=bool, `7`=u8). Note that **MLX's GPU does not
support `f64`** — a `3` dtype fails with "float64 is not supported on the GPU"
and yields a NULL handle. Use `0` (f32) for GPU math; `f64` only works if the
tensor is never evaluated on Metal.

| Function | Description |
|----------|-------------|
| `mlx_zeros(int64[] shape, int64 dtype) -> tensor` | Zero tensor |
| `mlx_ones(int64[] shape, int64 dtype) -> tensor` | Ones tensor |
| `mlx_full(int64[] shape, float64 val, int64 dtype) -> tensor` | Filled tensor |
| `mlx_arange(float64 start, float64 stop, float64 step, int64 dtype) -> tensor` | Range vector |
| `mlx_eye(int64 n, int64 m, int64 dtype) -> tensor` | Identity matrix |
| `mlx_random_normal(int64[] shape, float64 loc, float64 scale, int64 dtype) -> tensor` | Normal init |
| `mlx_random_uniform(int64[] shape, float64 lo, float64 hi, int64 dtype) -> tensor` | Uniform init |
| `mlx_from_array(float64[] flat, int64[] shape, int64 dtype) -> tensor` | Tensor from an Ionic array |

Lifetime and inspection:

| Function | Description |
|----------|-------------|
| `mlx_eval(tensor t) -> int64` | Force the lazy graph to run (0 ok, -1 error) |
| `mlx_free(tensor t)` | Release the handle |
| `mlx_to_array(tensor t) -> float64[]` | Copy out as a flat row-major array (evaluates) |
| `mlx_item(tensor t) -> float64` | Scalar value of a 1-element tensor (any dtype, converted) |
| `mlx_ndim(tensor t) -> int64` | Rank |
| `mlx_size(tensor t) -> int64` | Element count |
| `mlx_dim(tensor t, int64 i) -> int64` | Size of dimension `i` |
| `mlx_shape(tensor t) -> int64[]` | Shape as an `[int64]` (feeds back into `reshape`) |
| `mlx_dtype(tensor t) -> int64` | Element dtype code |

Elementwise and linear algebra:

| Function | Description |
|----------|-------------|
| `mlx_add` / `mlx_subtract` / `mlx_multiply` / `mlx_divide` | `(tensor, tensor) -> tensor` |
| `mlx_power` / `mlx_maximum` / `mlx_minimum` | `(tensor, tensor) -> tensor` |
| `mlx_matmul(tensor a, tensor b) -> tensor` | Matrix multiply |
| `mlx_negative` / `mlx_square` / `mlx_sqrt` / `mlx_exp` / `mlx_log` | `(tensor) -> tensor` |
| `mlx_abs` / `mlx_tanh` / `mlx_erf` | `(tensor) -> tensor` |
| `mlx_contiguous(tensor t) -> tensor` | Materialised contiguous copy |
| `mlx_concatenate(tensor a, tensor b, int64 axis) -> tensor` | Join along an axis |

Reductions (the `keepdims` argument is `0` or `1`):

| Function | Description |
|----------|-------------|
| `mlx_sum` / `mlx_mean` / `mlx_max` / `mlx_min` `(tensor, int64 keepdims)` | Reduce everything |
| `mlx_sum_axis` / `mlx_mean_axis` / `mlx_argmax` `(tensor, int64 axis, int64 keepdims)` | Reduce one axis (negative = from end) |
| `mlx_softmax(tensor t, int64 axis) -> tensor` | Softmax |
| `mlx_log_softmax(tensor t, int64 axis) -> tensor` | `log(softmax(t))` |

Shape manipulation:

| Function | Description |
|----------|-------------|
| `mlx_reshape(tensor t, int64[] shape) -> tensor` | Reshape |
| `mlx_transpose(tensor t, int64[] axes) -> tensor` | Permute dims (`[]` = reverse) |
| `mlx_astype(tensor t, int64 dtype) -> tensor` | Cast element type |
| `mlx_slice(tensor t, int64 start, int64 stop, int64 axis) -> tensor` | Half-open slice |
| `mlx_take(tensor t, int64[] indices, int64 axis) -> tensor` | Gather by index |

NN primitives (fused Metal kernels — building blocks, not a layer stack):

| Function | Description |
|----------|-------------|
| `mlx_sdpa(tensor q, tensor k, tensor v, float64 scale, int64 causal) -> tensor` | Scaled dot-product attention |
| `mlx_rms_norm(tensor x, tensor weight, float64 eps) -> tensor` | RMSNorm |
| `mlx_layer_norm(tensor x, tensor weight, tensor bias, float64 eps) -> tensor` | LayerNorm |
| `mlx_rope(tensor x, int64 dims, float64 base, int64 traditional, int64 offset) -> tensor` | Rotary embedding |

Diagnostics and memory:

| Function | Description |
|----------|-------------|
| `mlx_available() -> int64` | 1 if the backend is compiled in |
| `mlx_metal_available() -> int64` | 1 if a Metal device is present |
| `mlx_last_error() -> string` | Message from the last failed op |
| `mlx_clear_error()` | Reset the stored error |
| `mlx_active_memory() -> int64` / `mlx_peak_memory() -> int64` | Live / high-water bytes |
| `mlx_reset_peak_memory()` / `mlx_clear_cache()` | Reset stats / release cached buffers |
| `mlx_set_memory_limit(int64 bytes) -> int64` | Cap MLX allocation (0 = no limit) |

Safetensors weights (handles are plain `int64`, not tensors):

| Function | Description |
|----------|-------------|
| `mlx_load_safetensors(string path) -> int64` | Load a `.safetensors` file |
| `mlx_weights_count(int64 w) -> int64` | Number of tensors |
| `mlx_weights_at(int64 w, int64 i) -> tensor` | Tensor by index (iteration order is MLX's) |
| `mlx_weights_key_at(int64 w, int64 i) -> string` | Key by index |
| `mlx_weights_get(int64 w, string name) -> tensor` | Tensor by name |
| `mlx_save_safetensors(tensor t, string path, string key) -> int64` | Save one tensor |
| `mlx_weights_free(int64 w)` | Release the weight handle |

> **No training algorithms.** There is no optimizer, loss function, or backward
> pass in this API — deliberately. What ships is the set of primitives a
> training loop is assembled *from*, so the algorithm belongs in `.ionic`
> source rather than baked into the runtime. Gradient computation would need
> MLX's C closures (`mlx_value_and_grad`), which require first-class function
> values; Ionic has none yet, so autodiff is a language-surface change rather
> than a backend gap.

---

## CLI flags

```
ionic <file.ionic> [more.ionic ...] -o <output>

  -o <name>       Output binary name (default: a.out)
```

`ionic_new` compiles the source file(s) to a native binary: it emits a Mach-O
object as `<output>.o` (next to the executable) and then links it with the
runtime. Passing several source files concatenates them in order, but the usual
mechanism for multi-file programs is `import` rather than listing files.

The self-hosted compiler has one extra debug flag, `--llvm`, which writes LLVM
IR to `<output>.ll` instead of the native object (legacy path; not linked).
