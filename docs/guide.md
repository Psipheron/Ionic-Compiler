# Ionic — Programmer's Guide

A practical walkthrough of the language from first program to ML inference.

---

## 1. Hello world

```ionic
fn main() -> int64 {
    print("Hello, Ionic!");
    return 0;
}
```

Compile and run:

```sh
ionic hello.ionic --run
```

Every program with a `fn main() -> int64` uses it as the entry point. The return value becomes the process exit code. Programs without `fn main` run their top-level statements directly — useful for quick scripts.

---

## 2. Variables

```ionic
let x = 42;            // immutable int64 (inferred)
mut count = 0;         // mutable int64
let pi: float64 = 3.14159265358979;
let name: string = "Ionic";
```

`let` bindings cannot be reassigned. `mut` bindings can. The type is always inferred from the right-hand side unless you annotate it explicitly.

**Types at a glance:**

| Literal | Type |
|---------|------|
| `42` | `int64` |
| `3.14` | `float64` |
| `"hello"` | `string` |
| `true` / `false` | `bool` |
| `[1, 2, 3]` | `[int64]` |
| `[1.0, 2.0]` | `[float64]` |

---

## 3. Functions

```ionic
fn add(int64 a, int64 b) -> int64 {
    return a + b;
}

fn greet(string name) -> string {
    return str_concat("Hello, ", name);
}
```

Parameters are written `type name`. Return type follows `->`. Recursion works:

```ionic
fn factorial(int64 n) -> int64 {
    if (n <= 1) { return 1; }
    return n * factorial(n - 1);
}
```

Functions can be defined after they're called — the compiler does a full-program pass before codegen.

---

## 4. Control flow

### If / else

```ionic
if (x > 0) {
    print("positive");
} else if (x == 0) {
    print("zero");
} else {
    print("negative");
}
```

### While

```ionic
mut i = 0;
while (i < 10) {
    i = i + 1;
}
```

### For (range)

```ionic
for (i in 0..10) {        // 0, 1, ..., 9  (exclusive upper bound)
    print(int64_to_str(i));
}
```

### Break and continue

```ionic
for (i in 0..100) {
    if (i == 50) { break; }
    if (i - (i / 2) * 2 == 0) { continue; }   // skip evens
    print(int64_to_str(i));
}
```

---

## 5. Strings

Strings are immutable UTF-8 values. Use built-ins to work with them:

```ionic
let s = "hello world";

let n     = str_len(s);                       // 11
let upper = str_concat("prefix: ", s);        // "prefix: hello world"
let eq    = str_eq(s, "hello world");         // true
let ch    = str_index(s, 0);                  // 104 (ASCII 'h')
let cs    = char_to_str(65);                  // "A"
let ns    = int64_to_str(42);                 // "42"
let fs    = float64_to_str(3.14);             // "3.140000"
```

The standard library adds more — `contains`, `trim`, `pad_left`, `starts_with`, etc.:

```ionic
import std.str.*;

let found   = contains("hello world", "world");   // true
let trimmed = trim("   hello   ");                 // "hello"
let padded  = pad_left("7", 4, "0");              // "0007"
```

---

## 6. Arrays

Arrays are dynamic and grow with `.push`. The element type is inferred from the first literal.

```ionic
mut nums = [10, 20, 30];
nums.push(40);

let n     = nums.len;        // 4  (property access, no parens)
let first = nums[0];         // 10
nums[0]   = 99;              // element assignment
```

Arrays of floats:

```ionic
mut scores = [0.9, 0.7, 0.85];
scores.push(1.0);
```

Standard library helpers:

```ionic
import std.array.*;

let total   = arr_sum([1, 2, 3, 4, 5]);      // 15
let biggest = arr_max([3, 1, 4, 1, 5]);      // 5
let idx     = arr_index_of([10, 20, 30], 20); // 1
```

---

## 7. Structs

```ionic
struct Point {
    x: float64,
    y: float64
}

fn distance(Point a, Point b) -> float64 {
    let dx = a.x - b.x;
    let dy = a.y - b.y;
    return sqrt(dx * dx + dy * dy);
}

fn main() -> int64 {
    let p1 = Point { x: 0.0, y: 0.0 };
    let p2 = Point { x: 3.0, y: 4.0 };
    let d = distance(p1, p2);
    print(float64_to_str(d));    // 5.000000
    return 0;
}
```

All struct fields are mutable regardless of whether the binding is `let` or `mut`.

---

## 8. Doc comments

Triple-slash comments immediately before a `fn` or `struct` are attached as documentation. They appear in `--dump-ast` output.

```ionic
/// Returns the area of a circle with the given radius.
fn circle_area(float64 r) -> float64 {
    return 3.14159265358979 * r * r;
}
```

---

## 9. Imports and the standard library

```ionic
import std.math.*;            // wildcard — all symbols available, only used ones compiled
import std.str.{contains, trim};  // named — exactly these two symbols
import std.io.*;
```

Wildcard imports are zero-overhead: the compiler walks your program's AST, finds which symbols you reference (including transitive dependencies), and only emits IR for those. A `math.*` import when you only call `clamp` compiles identically to `import std.math.{clamp}`.

**Available modules:**

| Module | Import | Contents |
|--------|--------|----------|
| `std.math` | `import std.math.*;` | `clamp`, `lerp`, `sign`, `min_f`, `max_f`, `degrees`, `radians`, `pow_i`, `PI`, `TAU`, `E` |
| `std.str` | `import std.str.*;` | `contains`, `starts_with`, `ends_with`, `repeat`, `count_char`, `pad_left`, `pad_right`, `trim_left`, `trim_right`, `trim`, `str_reverse` |
| `std.array` | `import std.array.*;` | `arr_sum`, `arr_sum_f`, `arr_mean`, `arr_max`, `arr_min`, `arr_contains`, `arr_fill`, `arr_reverse`, `arr_index_of`, `arr_copy` |
| `std.io` | `import std.io.*;` | `readline`, `readline_n`, `eprint`, `eprintln`, `print_hr` |

Full API: [stdlib.md](stdlib.md)

---

## 10. Hardware placement

Ionic tracks which device a tensor lives on at compile time.

```ionic
let weights: tensor@gpu = load_model("model.onnx").forward(inputs);
let inputs:  tensor@cpu = ...;

// Move data across devices explicitly
let inputs_gpu = inputs.toGpu();

// gpu block — compiler enforces:
//   - only tensor@gpu variables and compute allowed
//   - file I/O (print, file_read, file_write) is forbidden here
gpu {
    let prediction = forward(inputs_gpu, weights);
}
```

Annotate functions with their target device:

```ionic
@gpu fn train_step(tensor@gpu x, tensor@gpu w) -> tensor@gpu {
    return forward(x, w);
}

@cpu fn load_batch(string path) -> tensor@cpu {
    return ...;
}

@gpu(0.5) fn half_gpu(...) -> ... { ... }   // use 50% of GPU
```

---

## 11. ML model inference

### GGUF (LLM text generation)

```ionic
fn main() -> int64 {
    let mdl = load_model("/path/to/model.gguf");
    gguf_set_temp(mdl, 0.7);
    gguf_set_top_p(mdl, 0.9);
    let reply = gguf_generate(mdl, "Explain recursion briefly.", 256);
    print(reply);
    model_free(mdl);
    return 0;
}
```

Supports any GGUF model — including Ollama's extension-less blob files. The runtime detects the format from the file's magic bytes and routes to llama.cpp with full Metal GPU acceleration.

### ONNX (general inference + Piper TTS)

```ionic
fn main() -> int64 {
    let mdl = load_model("/path/to/model.onnx");
    let phonemes = [1, 20, 61, 24, 27, 100, 3, 35, 62, 122, 24, 17, 2];
    let samples = piper_forward(mdl, phonemes, 0.667, 1.0, 0.8);
    write_wav("/tmp/out.wav", samples, 0, 16000);
    model_free(mdl);
    return 0;
}
```

ONNX models run via ONNX Runtime with CoreML execution provider on Apple Silicon.

### MLX (Apple-silicon tensor runtime)

For model math you write yourself — rather than a model file you load — use the
MLX backend. It gives you n-dimensional arrays on the GPU with lazy evaluation:

```ionic
fn main() -> int64 {
    if (mlx_available() == 0) {
        println("MLX backend not compiled in on this platform");
        return 1;
    }

    // A = [[1,2],[3,4]]
    let a = mlx_from_array([1.0, 2.0, 3.0, 4.0], [2, 2], 0);

    let c  = mlx_matmul(a, a);           // [[7,10],[15,22]]
    let v  = mlx_to_array(c);            // evaluates and copies out
    println(float64_to_str(v[0]));       // 7

    let sm = mlx_softmax(a, -1);         // softmax over the last axis
    println(float64_to_str(mlx_item(mlx_sum(sm, 0))));   // ~2 (rows sum to 1)

    mlx_free(sm); mlx_free(c); mlx_free(a);
    return 0;
}
```

Key points:

- **Handles are `tensor@cpu`**, not `tensor@gpu`. They are host-side handles to a
  device graph, so they are usable anywhere; `.toGpu()` is a no-op transfer that
  only exists to enter a `gpu { }` context.
- **Nothing computes until you ask.** Ops build a lazy graph; `mlx_eval` runs it,
  and `mlx_to_array` / `mlx_item` evaluate implicitly. Chain many ops and
  evaluate once.
- **No RAII.** Unlike `load_model`, handles are not freed at scope exit — call
  `mlx_free()` (and `mlx_weights_free()` for a safetensors handle).
- **Errors return NULL, they don't unwind.** A failed op yields a NULL handle and
  stores a message readable via `mlx_last_error()`. Check for NULL before use.
- **Use `f32` (dtype `0`) for GPU math.** MLX's Metal backend rejects `f64`
  ("float64 is not supported on the GPU") and returns NULL.

Loading pretrained weights:

```ionic
let w = mlx_load_safetensors("/path/to/model.safetensors");
println(str_concat("tensors: ", int64_to_str(mlx_weights_count(w))));

let embed = mlx_weights_get(w, "model.embed_tokens.weight");
println(str_concat("embed shape: ", int64_to_str(mlx_dim(embed, 0))));

mlx_free(embed);
mlx_weights_free(w);
```

The fused kernels a transformer needs are exposed as primitives —
`mlx_rms_norm`, `mlx_layer_norm`, `mlx_sdpa` (attention), `mlx_rope`. There is
**no layer stack and no training loop** in the API: those are meant to be
written in Ionic using these building blocks.

### Which backend gets linked

Backends are linked per-program, only when the source references them — a plain
`ionic hello.ionic` links none of them:

- `mlx_*` → `mlx-c` (Apple silicon only)
- `gguf_*` / `load_model` → llama.cpp
- `piper_forward` / `model_forward` / `load_model` → ONNX Runtime

On a platform without the library, the program still **compiles and links**: the
runtime exports every symbol and reports "not compiled in" at runtime. That
keeps one source tree portable — MLX on Apple silicon, GGUF/llama.cpp elsewhere.

Overrides:

```sh
IONIC_NO_MLX=1 ionic model.ionic -o model          # force MLX off
IONIC_MLX_INCLUDE=/opt/mlx/include \
IONIC_MLX_LIB=/opt/mlx/lib ionic model.ionic -o model   # non-Homebrew prefix
```

---

## 12. File I/O

```ionic
// Read / write
let content = file_read("data.txt");
file_write("out.txt", content);

// Check existence
if (file_exists("config.json")) {
    let cfg = file_read("config.json");
}
```

---

## 13. Interactive input

```ionic
import std.io.*;

fn main() -> int64 {
    print("Enter your name:");
    let name = readline();
    print(str_concat("Hello, ", name));
    return 0;
}
```

---

## 14. Command-line arguments

```ionic
fn main() -> int64 {
    let path = get_arg(0);    // first argument after the binary name
    if (file_exists(path)) {
        let data = file_read(path);
        print(data);
    } else {
        print("File not found.");
    }
    return 0;
}
```

---

## 15. Complete example — word frequency counter

```ionic
import std.str.*;
import std.io.*;

fn count_word(string text, string word) -> int64 {
    let tlen = str_len(text);
    let wlen = str_len(word);
    mut count = 0;
    mut i = 0;
    while (i <= tlen - wlen) {
        let sub = str_index(text, i);
        if (contains(text, word)) {
            count = count + 1;
        }
        i = i + 1;
    }
    return count;
}

fn main() -> int64 {
    print("Enter a line of text:");
    let line = readline();
    let trimmed = trim(line);
    print(str_concat("You typed: ", trimmed));
    print(str_concat("Length: ", int64_to_str(str_len(trimmed))));
    return 0;
}
```

---

## Common pitfalls

**Mutability** — forgetting `mut` when you need to reassign:
```ionic
let x = 0;
x = 1;       // error: cannot assign to immutable binding
mut y = 0;
y = 1;       // ok
```

**Array type syntax** — element type goes inside brackets:
```ionic
fn sum([int64] a) -> int64 { ... }    // correct
fn sum(int64[] a) -> int64 { ... }    // wrong — won't parse
```

**String comparison** — use `str_eq`, not `==`:
```ionic
if (str_eq(name, "ionic")) { ... }   // correct
if (name == "ionic") { ... }          // type error
```

**`model` is a keyword** — use a different binding name:
```ionic
let mdl = load_model("model.gguf");   // correct
let model = load_model("...");         // parse error — `model` is reserved
```

**For-range is exclusive** — `0..n` iterates `0` through `n-1`:
```ionic
for (i in 0..3) { ... }    // i = 0, 1, 2
```
