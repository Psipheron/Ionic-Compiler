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

Ionic tracks which **memory pool** a tensor lives in, at compile time.
`@cpu` is host memory (SRAM), `@gpu` is accelerator memory (VRAM), and the two
have different capacities.

```ionic
let weights: tensor@gpu = load_model("model.onnx").forward(inputs);
let inputs:  tensor@cpu = ...;

// Move data between pools explicitly. This consumes `inputs` — using it
// afterwards is an error, because a value occupies exactly one pool.
let inputs_gpu = inputs.toGpu();

// gpu block — compiler enforces:
//   - only tensor@gpu variables and compute allowed
//   - file I/O (print, file_read, file_write) is forbidden here
//   - the sized tensor@gpu values created here must fit the pool's budget
gpu {
    let prediction = forward(inputs_gpu, weights);
}
```

Annotate functions with the pool they run in:

```ionic
@gpu fn train_step(tensor@gpu x, tensor@gpu w) -> tensor@gpu {
    return forward(x, w);
}

@cpu fn load_batch(string path) -> tensor@cpu {
    return ...;
}

// At most a fraction of the accelerator pool. The pool size itself is a
// deployment fact, declared once:
let vram = 25769803776;               // bytes of accelerator memory

@gpu(0.5) fn half_gpu(...) -> ... { ... }   // at most half of the pool
```

The pool declaration and the fraction are checked: a fraction must be in
`(0, 1]`, and no fraction is allowed unless a `vram` pool is declared.

The budget is enforced for what the checker can size statically. A tensor's
size is `prod(shape) × dtype_bytes`, derived from the constructor (`mlx_zeros`,
`mlx_ones`, `mlx_full`, `mlx_eye`, …) or inherited from a shape-preserving op
(`mlx_add`, `mlx_matmul`, the elementwise family). Each sized `tensor@gpu` that
a `@gpu fn` body or `gpu { }` block *creates* is added up, and the declaration
that pushes the total past the scope's share of the pool is a compile error
naming both byte counts.

It is a necessary condition, not a proof: a value whose size is not statically
knowable — a tensor *parameter*, or the result of an op the size pass does not
model, such as `mlx_reshape` — is exempt rather than rejected. A parameter was
already allocated and charged in the caller's scope, and rejecting it would make
every function that takes a tensor uncompilable. So the check under-approximates
(it can miss an over-budget program) but never falsely rejects one.

An exemption is **reported**, not silent. When a budgeted scope closes with
values it could not size, the compiler prints a `[warn]` naming the first one,
how many there were, and what it could actually size:

```
[warn] 5:40: 1 GPU value in this scope could not be sized statically, so the budget does not account for it (nothing in this scope could be sized, against a 1073741824 byte budget) — give it a literal shape, or check mlx_pool_bytes() at run time
```

A warning does not fail the build — an unbounded value is not provably wrong —
but it makes "this fits" and "this was never measured" different answers, which
a silent exemption did not.

Live-range analysis is also not done yet: the sum covers every sized binding in
the scope, so a value that dies before the next allocation still counts.

What the compiler cannot prove is caught at run time instead. The declared pool
size is published to the runtime at startup, and every allocation is charged
against it; one that would not fit aborts with a message naming the request, the
pool, the current occupancy, and the declared capacity. That is `exit(1)` rather
than a NULL handle, because Ionic has no way to observe NULL from an `mlx_*` call
— returning one would turn an over-budget program into a segfault at an unrelated
line. So the two halves compose: the checker rejects what it can *prove*, and the
allocator rejects the rest where the bytes are actually taken.

`mlx_pool_capacity(pool)` reports what the program declared (0 for an unbounded
pool, -1 for a pool id that does not exist), so a program can compare its
declaration against the device and warn before allocating rather than after:

```ionic
let vram = 25769803776;
if (mlx_pool_capacity(1) < needed) {
    println("declared pool is too small for this model");
    return 1;
}
```

A transfer is a **move**. After `let g = x.toGpu();` the binding `x` no longer
owns a value, and reading it is a compile error that names the transfer and the
line it happened on:

```ionic
let a = mlx_ones([2, 2], 0);
let g = a.toGpu();          // moves `a`
gpu { let s = mlx_add(g, g); }   // fine: builtins borrow, so `g` is read twice
let back = g.toCpu();       // moves `g`
// mlx_free(a);             // error: `a` was moved on line 2
```

Only transfers move; everything else borrows, which is why `mlx_add(g, g)` stays
legal. Rebinding a name clears the mark, and transferring a temporary
(`mlx_ones(...).toGpu()`) consumes no binding at all.

On Apple silicon the two pools are physically the same memory, so a transfer
does not copy — only the accounting changes. Everywhere else they are separate
and a transfer really does move bytes.

---

## 11. ML model inference

### ONNX (the only model backend)

```ionic
fn main() -> int64 {
    let mdl = load_model("/path/to/model.onnx");
    let phonemes = [1, 20, 61, 24, 27, 100, 3, 35, 62, 122, 24, 17, 2];
    let samples = piper_forward(mdl, phonemes, 0.667, 1.0, 0.8);
    write_wav("/tmp/out.wav", samples, 0, 16000);
    // no model_free: load_model is RAII and is freed at scope exit. An
    // explicit model_free(mdl) would also be correct — it suppresses the
    // implicit destructor, so exactly one free runs either way.
    return 0;
}
```

ONNX models run via ONNX Runtime, with the CoreML execution provider enabled on
Apple Silicon.

### Why not GGUF

GGUF support shipped briefly on top of llama.cpp and was removed. llama.cpp's
only placement control is `n_gpu_layers` — one integer, set at load time, that
says how many transformer layers live on the GPU. The whole model moves as a
unit, and nothing above the llama.cpp API can ask where an individual tensor
lives. That cannot back `tensor@cpu` / `tensor@gpu`, which the type checker
enforces per value. Shipping GGUF behind those types would have meant the
placement guarantee held for ONNX and MLX but was fiction for the format most
people actually run.

`.gguf` is still recognised so the failure is explicit rather than "unknown
format":

```
[ionic] load_model: GGUF inference was removed with llama.cpp;
        export the model to ONNX instead
```

Reinstating it means going below llama.cpp to ggml, where a tensor's buffer type
is reachable per tensor (`ggml_tensor->buffer`, `tensor_buft_overrides`). That is
the same shape of control ORT's `OrtIoBinding` and MLX's per-stream arrays give,
which is why those two are the reference implementations for placement.

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
  device graph, so they are usable anywhere. On Apple silicon the CPU and GPU
  share physical memory, so `.toGpu()` does not copy — the bytes stay put and
  only the accounting pool changes. That is the *unified memory* case: one
  implementation of the pool model, not the model itself.
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
- `piper_forward` / `model_forward` / `load_model` → ONNX Runtime

On a platform without the library, the program still **compiles and links**: the
runtime exports every symbol and reports "not compiled in" at runtime. That
keeps one source tree portable — MLX on Apple silicon, ONNX Runtime everywhere.

Overrides:

```sh
IONIC_NO_MLX=1 ionic model.ionic -o model          # force MLX off
IONIC_MLX_INCLUDE=/opt/mlx/include \
IONIC_MLX_LIB=/opt/mlx/lib ionic model.ionic -o model   # non-Homebrew prefix
```

---

## 12. Local inference server

The runtime also ships the two pieces needed to put a model behind a socket:
a BPE tokenizer that reads a HuggingFace `tokenizer.json`, and a minimal
blocking HTTP/1.1 server. Together they turn token ids into text and back.

### BPE tokenizer

```ionic
let n = bpe_load(str_concat(model_dir, "/tokenizer.json"));   // vocab size, or -1
print(int64_to_str(n));                                        // 32000

let ids = [0];
arr_reset(ids);                       // empty the array
bpe_encode("The capital of France is", ids, 1);   // 1 = prepend BOS
print(bpe_decode(ids));               // " The capital of France is"
```

- `bpe_load(path)` — parse `tokenizer.json` and build the vocab hash table.
  Returns the vocabulary size, or `-1` on failure. Idempotent: loading a second
  file releases the first.
- `bpe_encode(text, ids, add_bos)` — append ids to an existing `[int64]` array
  (it is cleared first). Returns the token count.
- `bpe_decode(ids)` — the inverse, as a fresh string.
- `bpe_bos()` / `bpe_eos()` — the special-token ids, resolved from the vocab.

Encoding follows the HF `tokenizers` reference for this model family:
normalize (`' '` → `▁`), greedy longest-match seeding, lowest-rank pair merging,
then `<0xNN>` byte fallback for anything still unknown. Ids match HF exactly on
TinyLlama-1.1B; `testdata/tokenizer_mini.json` is a small hand-built fixture
that CI uses to exercise the same code path without a model download.

### HTTP server

```ionic
fn main() -> int64 {
    let fd = http_serve(8080);          // listener, or -1
    if (fd < 0) { return 1; }

    while (true) {
        let req = http_accept(fd);      // *request* handle, or -1
        if (req >= 0) {
            let method = http_method(req);        // "GET", "POST", ...
            let path   = http_path(req);          // query string already stripped
            let body   = http_req_body(req);      // "" when there is none

            if (str_eq(path, "/health")) {
                http_respond(req, 200, "text/plain", "ok");
            } else {
                http_respond(req, 404, "application/json", "{\"error\":\"not found\"}");
            }
            http_req_done(req);         // close the connection, free the request
        }
    }
    http_close(fd);
    return 0;
}
```

One handle owns the whole exchange: `http_accept` returns a request that holds
its own connection fd, and `http_respond` / `http_req_done` take that same
handle back. There is no way to mix up a socket fd with a request pointer.

Every response carries `Connection: close`; `http_req_done` closes the socket.
The loop is single-request-at-a-time by design — this is enough to serve a
model behind `/v1/chat/completions` for a proof of concept, not a production
web server. A request that never arrives (client connects and hangs up) makes
`http_accept` return `-1` without disturbing later accepts.

---

## 13. File I/O

```ionic
// Read / write
let content = file_read("data.txt");
file_write("out.txt", content);

// Check existence
if (file_size("config.json") >= 0) {
    let cfg = file_read("config.json");
}
```

`file_size` returns `-1` for a missing path or a non-regular file and the byte
count otherwise, so `>= 0` is the existence test. It costs one `stat` and
allocates nothing; `file_read` mallocs the whole file and Ionic strings are
never freed, which is why existence checks should not use it.

---

## 14. Interactive input

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

## 15. Command-line arguments

```ionic
fn main() -> int64 {
    let path = get_arg(0);    // first argument after the binary name
    if (file_size(path) >= 0) {
        let data = file_read(path);
        print(data);
    } else {
        print("File not found.");
    }
    return 0;
}
```

---

## 16. Complete example — word frequency counter

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
let mdl = load_model("model.onnx");   // correct
let model = load_model("...");         // parse error — `model` is reserved
```

**For-range is exclusive** — `0..n` iterates `0` through `n-1`:
```ionic
for (i in 0..3) { ... }    // i = 0, 1, 2
```
