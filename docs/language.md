# Ionic Language Reference

Ionic is a statically-typed, curly-brace language for data science and AI workloads. It compiles to native ARM64 machine code for macOS (Mach-O) and Linux (ELF) — emitting object files directly, with no LLVM at runtime (an LLVM path exists as a legacy/debug backend) — and supports explicit hardware placement (`@cpu` / `@gpu`).

---

## Types

| Type | Description |
|------|-------------|
| `int64` | 64-bit signed integer |
| `float64` | 64-bit IEEE 754 double |
| `bool` | Boolean (`true` / `false`) |
| `string` | UTF-8 string (immutable value) |
| `tensor@cpu` | Tensor in the host memory pool (SRAM) |
| `tensor@gpu` | Tensor in the accelerator memory pool (VRAM) |
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

`@cpu` and `@gpu` name **memory pools**, not devices:

| Placement | Pool | Capacity |
|---|---|---|
| `tensor@cpu` | SRAM — host memory | the machine's RAM |
| `tensor@gpu` | VRAM — accelerator memory | the card's memory |

The two pools have **different capacities**, and VRAM is usually the smaller
of the two. A `tensor@gpu` value means "its bytes live in the accelerator's
memory pool" — a statement about *where the bytes are*, not about which chip
will eventually touch them.

```ionic
let vram = 25769803776;                 // bytes of accelerator memory

@gpu fn train_step(...) -> ... { ... }   // may use the whole pool
@cpu fn load_data(...) -> ...  { ... }  // host memory

@gpu(0.25) fn partial(...) -> ... { ... } // at most a quarter of the pool

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

### Transfers move

`.toGpu()` and `.toCpu()` move a value between pools. The source is
**consumed**: after `let g = x.toGpu();`, `x` is dead and using it is an
error. This is what makes a pool's occupancy meaningful — a value occupies
exactly one pool at a time, so the compiler can check that the GPU pool is
not oversubscribed. A copy is available as a separate, explicitly-named
operation for when the source must stay usable.

### Capacity is checked at compile time

The compiler sums the sizes of the `tensor@gpu` values live in a scope and
rejects the program if the total exceeds that scope's share of the pool. A
tensor whose size cannot be determined statically is rejected in a GPU scope
rather than silently allowed: an unbounded GPU allocation would defeat the
budget.

### Unified memory

On Apple silicon the CPU and GPU address the same physical memory, so a
transfer there is free — the bytes do not move. The type still changes,
because the value is now accounted against the accelerator pool, and that
accounting is what the budget checks. Everywhere else the pools are separate
and a transfer really does copy.

### Backends

The pools are a language-level concept; the implementation is a backend.
MLX is the backend that ships today, and on Apple silicon it provides the
unified-memory case. A portable backend (threaded host execution) or a
hardware one (CUDA, ROCm) implements the same interface.

A GPU scope on a machine with **no** accelerator backend refuses to run
rather than silently executing on the host. Everything the compiler can check
— pool separation, budget, move semantics — is checked statically, so the
refusal is only ever reached for what genuinely cannot be known at build time.

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
fn abs64(int64 x) -> int64 {
    if (x < 0) { return 0 - x; }
    return x;
}
```

`///` lines immediately before a `fn` or `struct` definition are attached to that definition as its documentation, so they travel with it in tooling that reads the AST.

---

## Built-in functions

### Printing
| Function | Description |
|----------|-------------|
| `print(s)` | Write `s` to stdout with **no** trailing newline |
| `println(s)` | Write `s` to stdout followed by a newline |

Both accept a `string` or an `int64`; the integer form is formatted for you.

```ionic
print("A");
print("B");
println("");      // "AB" then a newline
println(42);      // "42"
```

### String operations
| Function | Description |
|----------|-------------|
| `str_concat(string a, string b) -> string` | Concatenate two strings |
| `str_len(string s) -> int64` | Length in bytes |
| `str_eq(string a, string b) -> bool` | Equality test |
| `str_index(string s, int64 i) -> int64` | ASCII code of character at index i |
| `int64_to_str(int64 n) -> string` | Format integer as string |
| `float64_to_str(float64 x) -> string` | Format float as string |
| `char_to_str(int64 code) -> string` | One-**byte** string from a code in `0`-`255` |

String literals support the escapes `\n`, `\t`, `\r`, `\"`, `\\`, `\e` (ESC,
0x1B), and `\xHH` (a raw byte from two hex digits):

```ionic
print("\e[31mred\e[0m");    // ANSI color
print("\x41\x42");           // "AB"
```

`char_to_str` emits exactly one **raw byte** from the low 8 bits of the code, so
it is byte-identity for `0`-`255` and cannot represent a codepoint above that —
`char_to_str(0x2500)` returns an empty string, not `─`. It exists so the lexer
and JSON reader can rebuild string literals byte by byte without corrupting
non-ASCII source. UTF-8 *encoding* lives in Ionic (`lib/std/utf8.ionic`), which
builds the multi-byte sequence from `char_to_str` calls.

### Conversion
| Function | Description |
|----------|-------------|
| `int64_to_float64(int64 n) -> float64` | Widen integer to float |
| `float64_to_int64(float64 x) -> int64` | Truncate float to integer |

### Math
| Function | Description |
|----------|-------------|
| `sqrt(float64 x) -> float64` | Square root |
| `fabs(float64 x) -> float64` | Absolute value of a **float** |
| `abs(int64 n) -> int64` | Absolute value of an **integer** |
| `pow(float64 a, float64 b) -> float64` | `a` raised to `b` |
| `floor(float64 x) -> float64` | Round down |
| `ceil(float64 x) -> float64` | Round up |
| `min(int64 a, int64 b) -> int64` | Smaller of two integers |
| `max(int64 a, int64 b) -> int64` | Larger of two integers |

`abs` and `fabs` are separate functions with different parameter types — this is
not an overload. `abs(-5.0)` is a **type error** ("expected int64, got
float64"); write `fabs(-5.0)`. Likewise `min`/`max` are `int64`-only; there is
no float or array overload.

### File I/O
| Function | Description |
|----------|-------------|
| `file_read(string path) -> string` | Read entire file |
| `file_write(string path, string content)` | Write/overwrite file |
| `file_write_binary(int64[] bytes, string path)` | Write each **element** as one byte (`0`-`255`) |
| `file_size(string path) -> int64` | Byte length, or `-1` if missing / not a regular file |

> `file_exists` is **not** available on the native path — it is registered only
> in the legacy LLVM backends, so calling it fails to compile with "unknown
> function". Use `file_size(path) >= 0` instead, which is exactly equivalent
> (`ionic_file_size` returns `-1` for a missing path or a non-regular file, and
> `>= 0` for a real one, including an empty file).

### System
| Function | Description |
|----------|-------------|
| `get_arg(int64 n) -> string` | Command-line argument (0-indexed after program name) |
| `cpu_core_count() -> int64` | Number of logical CPU cores |
| `target_is_linux() -> int64` | 1 when the program was compiled for Linux |
| `getenv(string name) -> string` | Environment variable, or `""` when unset |
| `system(string cmd) -> int64` | Run a shell command; returns the raw `system(3)` wait status |
| `exit(int64 code)` | Terminate the process immediately |
| `read_line() -> string` | Read one line from stdin, newline stripped |

### Arrays
| Function | Description |
|----------|-------------|
| `len([T] a) -> int64` | Element count (same as `a.len`) |
| `push([T] a, T v)` | Append one element |
| `arr_reset([T] a)` | Set length to 0, keeping the capacity |
| `arr_push_i64([int64] a, int64 v)` | Append to an `[int64]` (runtime entry point) |

### Strings (additional)
| Function | Description |
|----------|-------------|
| `format(string fmt, ...) -> string` | Replace `{}` placeholders in order |
| `str_contains(string s, string sub) -> bool` | Substring test |
| `str_starts_with(string s, string prefix) -> bool` | Prefix test |
| `str_ends_with(string s, string suffix) -> bool` | Suffix test |
| `str_replace(string s, string from, string to) -> string` | Replace all occurrences |
| `str_slice(string s, int64 start, int64 end) -> string` | Half-open substring `[start, end)` |
| `str_hash(string s) -> int64` | djb2-xor hash |
| `str_to_int64(string s) -> int64` | Parse decimal int, `0` on failure |

```ionic
format("{} has {} items", name, int64_to_str(n));
str_slice("abcdef", 1, 4);        // "bcd"
```

### ML model
| Function | Description |
|----------|-------------|
| `load_model(string path) -> model` | Load an ONNX model (`model@cpu`) |
| `model_free(model m)` | Release model resources early (optional — RAII does it) |
| `model_forward(model m, tensor t) -> tensor` | Stub: warns and returns `t` unchanged |
| `piper_forward(model m, int64[] phonemes, float64 ns, float64 ls, float64 nw) -> float64[]` | Piper TTS inference |
| `write_wav(string path, float64[] samples, int64 n, int64 rate)` | Write PCM WAV file |

ONNX Runtime is the only model backend. `.gguf` files are still *recognised*
(`detect_format` reads the magic bytes) but there is no runner: GGUF support was
removed along with llama.cpp, because llama.cpp's only placement control is
`n_gpu_layers` — one load-time integer for the whole model. That cannot express
`tensor@cpu` / `tensor@gpu` per value, so it could not back the language's
placement types honestly. Export to ONNX, or wait for a low-level ggml path.

`piper_forward` is the only wired inference entry point today; it is used by
`src/piper_demo.ionic`. `model_forward` is a **stub** — it prints
"model handles are backend-specific; use ionic_piper_forward for ONNX" and
returns its input tensor without running anything.

`load_model` is an RAII resource: the compiler frees it automatically at scope
exit, so you normally **do not** call `model_free` yourself.

```ionic
let mdl = load_model("model.onnx");
let samples = piper_forward(mdl, phonemes, 0.667, 1.0, 0.8);
write_wav("/tmp/out.wav", samples, 0, 16000);
// no model_free — the compiler emits it at scope exit
```

Calling `model_free(m)` yourself is also correct: the explicit call consumes the
local's ownership, so the implicit destructor is suppressed and exactly one
`ionic_model_free` runs. The two forms are mutually exclusive, not additive.

```ionic
let mdl = load_model("model.onnx");
let samples = piper_forward(mdl, phonemes, 0.667, 1.0, 0.8);
write_wav("/tmp/out.wav", samples, 0, 16000);
model_free(mdl);   // released here instead of at scope exit
```

`scripts/model_raii_check.sh` asserts the emitted reference count for each
shape: one for the implicit form, one for the explicit form, and one per model
when several are live.

### MLX tensor backend

Apple-silicon array runtime. A handle's type records its **pool**, not a
device: `tensor@cpu` is SRAM-resident and `tensor@gpu` is
accelerator-resident, and the runtime handle carries the same flag, so the
distinction holds at run time and not only in the checker. Builtins are
pool-polymorphic — `mlx_add(cpu_a, cpu_b)` yields `tensor@cpu` and
`mlx_add(gpu_a, gpu_b)` yields `tensor@gpu` — while user functions stay strictly
placement-typed.

A `.toGpu()` / `.toCpu()` transfer **moves**: the source binding is consumed (a
later use is a use-after-move error) and the destination pool is charged for
the bytes. On Apple silicon the CPU and GPU address the same physical memory, so
no data is copied and only the accounting moves; on a discrete backend the value
is copied onto the destination stream and the source's storage is released. That
is why MLX is the *unified memory* backend — it is one implementation of the pool
model, not the model itself.

`mlx_pool(t)` reports which pool a handle lives in, `mlx_pool_bytes(p)` how many
bytes that pool currently holds, `mlx_pool_count()` how many pools there are, and
`mlx_pools_unified()` whether a transfer between them is free on this backend.
They exist so the pool model is observable, and therefore testable, from Ionic.

Unlike `load_model`, **handles are not RAII-freed**: call `mlx_free()`
explicitly.

Backend selection is automatic and per-program: the compiler links `mlx-c` only
when the source mentions `mlx_*`. On Intel Macs and Linux every function below
still exists, so a program links — it reports "MLX backend not compiled in" at
runtime.

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
ionic <file.ionic> [more.ionic ...] -o <output> [--target <triple>] [--print-link]

  -o <name>         Output binary name (default: a.out)
  --target <triple> Object format to emit (default: the host's)
  --print-link      Print the link command and exit, without linking
```

`ionic_new` compiles the source file(s) to a native binary: it emits a Mach-O
object as `<output>.o` (next to the executable) and then links it with the
runtime. Passing several source files concatenates them in order, but the usual
mechanism for multi-file programs is `import` rather than listing files.

### Cross-compiling to Linux ARM64

By default the object format follows the machine you are running on. `--target`
overrides that, so a macOS host can emit a Linux object:

```sh
ionic app.ionic -o app --target aarch64-linux
```

Recognised triples (aliases in parentheses):

- `aarch64-linux` (`aarch64-unknown-linux-gnu`, `aarch64-linux-gnu`, `linux`)
- `aarch64-macos` (`aarch64-apple-darwin`, `macos`)

An unrecognised triple is an error, not a silent fallback — emitting the wrong
object format produces a file that fails to link with a misleading error.

When the target differs from the host, the compiler stops after writing
`<output>.o` and prints the link commands to run on the target machine. It does
**not** invoke the host linker: a Mach-O `ld` cannot consume an ELF object, and
failing there would look like a codegen bug rather than an unsupported action.
Linking needs a C compiler for the target plus that target's libc, and any
backend libraries must be present on the machine that links — which is also
where backend discovery runs.

```sh
# on the build machine
ionic app.ionic -o app --target aarch64-linux
scp app.o src/ionic_model_runtime.c target:

# on the Linux ARM64 target
clang app.o ionic_model_runtime.c -o app -lm
```

`--print-link` prints the exact `clang` command line (including any detected
`-DIONIC_HAVE_*`, `-I`, `-L` and `-l` backend flags) and exits without linking.
It is how CI asserts that a plain program links no ML dependencies.
