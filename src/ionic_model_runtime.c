/*
 * Ionic ML model runtime
 *
 * Supported backends:
 *   .onnx              — ONNX Runtime (CoreML GPU on macOS)
 *   .pt / .pth         — stub (LibTorch not yet linked)
 *   .h5                — stub (Keras/HDF5 not yet linked)
 *   .mlmodel           — stub (CoreML direct load not yet linked)
 *
 * .gguf is recognised by detect_format() but has no runner: GGUF support was
 * removed with llama.cpp (see the note above the MLX section) and will return
 * via a low-level ggml path.
 *
 * Public API:
 *   ionic_load_model(path)                                 -> opaque handle
 *   ionic_model_free(model)                                -> void
 *   ionic_model_forward(model, input)                      -> tensor ptr (ONNX single-input)
 *   ionic_piper_forward(model, phoneme_arr, ns, ls, nw)    -> float64 array
 *   ionic_write_wav(path, arr, n_samples, sample_rate)     -> void
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#ifdef _WIN32
#  include <windows.h>
#else
#  include <unistd.h>
#  include <time.h>
#  include <limits.h>
#  include <stdlib.h>
#  ifdef __APPLE__
#    include <mach-o/dyld.h>
#  endif
#  include <sys/stat.h>
#endif

/* ── ONNX Runtime ─────────────────────────────────────────────────────────── */
#ifdef IONIC_HAVE_ORT
#  include "onnxruntime_c_api.h"
#  ifdef __APPLE__
#    include "coreml_provider_factory.h"
#  endif
#endif

/* ── llama.cpp (GGUF) ─────────────────────────────────────────────────────── */
/* Removed. GGUF text generation ran through llama.cpp's high-level API, whose
 * only placement control is n_gpu_layers — a single load-time integer with no
 * per-tensor residency. That could not back the language's tensor@cpu /
 * tensor@gpu distinction without pretending, so inference is ONNX Runtime only
 * until a low-level ggml path lands. See docs/language.md. */

/* ── MLX (Apple-silicon array/tensor runtime) ───────────────────────────────
 * MLX is an *array* runtime, not a model loader: it gives us n-dim arrays,
 * lazy graph construction, Metal eval, safetensors I/O and (via C closures)
 * autodiff primitives. There is deliberately NO training algorithm here —
 * this is the backend brick, not SGD/Adam. See the "MLX backend" section
 * below for the public ionic_mlx_* surface.
 *
 * Only available on macOS/arm64 (Metal). On Linux, MLX is compiled out
 * entirely and every ionic_mlx_* entry point reports "not compiled in" rather
 * than failing to link.
 * ------------------------------------------------------------------------- */
#if defined(IONIC_HAVE_MLX) && defined(__APPLE__)
#  include "mlx/c/mlx.h"
#  define IONIC_MLX_ACTIVE 1
#else
#  define IONIC_MLX_ACTIVE 0
#endif

/* ── Ionic dynamic-array layout ──────────────────────────────────────────────
 *   struct IonicArray { i64 len; i64 cap; void *data; }
 *   data is an array of i64 slots (8 bytes each)
 * ─────────────────────────────────────────────────────────────────────────── */
typedef struct {
    int64_t len;
    int64_t cap;
    void   *data;
} IonicArray;

static IonicArray *ionic_array_alloc(int64_t cap) {
    IonicArray *a = (IonicArray *)malloc(sizeof(IonicArray));
    a->len  = 0;
    a->cap  = cap > 0 ? cap : 8;
    a->data = malloc((size_t)(a->cap * 8));
    return a;
}

static void ionic_array_push_i64(IonicArray *a, int64_t v) {
    if (a->len >= a->cap) {
        a->cap *= 2;
        a->data = realloc(a->data, (size_t)(a->cap * 8));
    }
    ((int64_t *)a->data)[a->len++] = v;
}

/* ── Memory pools: identity and capacity (Stages C–D) ─────────────────────────
 * `tensor@cpu` / `tensor@gpu` name *pools*, not devices — SRAM (host memory,
 * capacity = system RAM) and VRAM (accelerator memory, capacity = the card).
 *
 * These live outside the MLX conditional because the pool model is a
 * *language* concept: `let vram = N;` is checked at compile time and enforced at
 * run time whether or not a backend is linked in. Keeping the ids and the
 * declared capacity in one place is what lets the checker and the allocator
 * agree on a number instead of each holding its own copy.
 *
 * IONIC_POOL_NONE is for a tensor that has not been assigned to a pool yet;
 * every handle produced by a backend ends up in CPU or GPU.
 */
#define IONIC_POOL_CPU  0
#define IONIC_POOL_GPU  1
#define IONIC_POOL_NONE (-1)
#define IONIC_POOL_COUNT 2

/* Declared capacity of each pool, in bytes; 0 means "unbounded".
 *
 * Set once at startup by ionic_pool_set_capacity(), which codegen emits from
 * the same `let vram = N;` the checker read. SRAM stays unbounded: `vram` names
 * the accelerator pool, and the host pool's capacity is the machine's, not the
 * source's. */
static int64_t g_pool_capacity[IONIC_POOL_COUNT] = {0, 0};

/* Publish the declared size of `pool`. Called from the generated `main` before
 * any user code runs. A negative or zero size is "unbounded" — the checker
 * already rejects a non-positive `vram`, so the only caller that can pass 0 is
 * a program that declares no pool at all. */
void ionic_pool_set_capacity(int64_t pool, int64_t bytes) {
    if (pool < 0 || pool >= IONIC_POOL_COUNT) return;
    g_pool_capacity[pool] = bytes > 0 ? bytes : 0;
}

int64_t ionic_mlx_pool_capacity(int64_t pool) {
    if (pool < 0 || pool >= IONIC_POOL_COUNT) return -1;
    return g_pool_capacity[pool];
}

/* ── System runtime ───────────────────────────────────────────────────────── */

static int    ionic_argc = 0;
static char **ionic_argv = NULL;

char *ionic_fgets_stdin(char *buf, int n) {
    char *r = fgets(buf, n, stdin);
    if (!r) { buf[0] = '\0'; return buf; }
    /* Strip trailing newline */
    size_t len = strlen(buf);
    if (len > 0 && buf[len-1] == '\n') buf[len-1] = '\0';
    return buf;
}

/* Create a new empty array (for array-literal codegen). */
IonicArray *ionic_make_array(void) { return ionic_array_alloc(8); }

/* Write a string to a file (text mode). */
int64_t ionic_file_write(const char *path, const char *content) {
    FILE *f = fopen(path, "w");
    if (!f) return -1;
    fputs(content ? content : "", f);
    fclose(f);
    return 0;
}

/* Write a [int64] byte array as raw binary. Each element is one byte (0-255). */
int64_t ionic_file_write_binary(IonicArray *arr, const char *path) {
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    int64_t *data = (int64_t *)arr->data;
    for (int64_t i = 0; i < arr->len; i++) {
        uint8_t byte = (uint8_t)(data[i] & 0xFF);
        fwrite(&byte, 1, 1, f);
    }
    fclose(f);
    return arr->len;
}

int64_t ionic_arr_reset(IonicArray *arr) {
    arr->len = 0;
    return 0;
}

/* Defined below; forward-declared so ionic_runtime_init can call it. */
const char *ionic_exe_dir(void);

void ionic_runtime_init(int argc, char **argv) {
    ionic_argc = argc;
    ionic_argv = argv;
    /* Publish the executable's directory as $IONIC_ROOT so a self-hosted
     * `ionic` can find its bundled `lib/` stdlib and `src/ionic_model_runtime.c`
     * regardless of the process CWD (e.g. a `brew`-installed binary run from an
     * arbitrary directory). `getenv("IONIC_ROOT")` then reads it back from the
     * Ionic sources without needing a new builtin. */
    const char *dir = ionic_exe_dir();
#if defined(_WIN32)
    char root_buf[PATH_MAX + 16];
    snprintf(root_buf, sizeof(root_buf), "IONIC_ROOT=%s", dir);
    _putenv(root_buf);
#else
    setenv("IONIC_ROOT", dir, 1);
#endif
}

const char *ionic_get_arg(int64_t n) {
    int idx = (int)(n + 1);
    if (idx < 1 || idx >= ionic_argc || !ionic_argv[idx]) return "";
    return ionic_argv[idx];
}

/* Read an environment variable. Returns empty string (not NULL) when unset.
 * Used by the import resolver to locate the per-user package cache
 * (~/.ionic/packages on Unix; falls back to USERPROFILE on Windows). */
const char *ionic_getenv(const char *name) {
    if (!name) return "";
    const char *v = getenv(name);
    return v ? v : "";
}

/* Absolute directory of the running executable, with no trailing slash.
 * Lets a self-hosted `ionic` locate its bundled `lib/` standard library and
 * `src/ionic_model_runtime.c` relative to the binary itself rather than the
 * process CWD, so an installed tarball works from any directory. Returns a
 * pointer to a static buffer (valid until the next call); "." on failure. */
const char *ionic_exe_dir(void) {
    static char dir[PATH_MAX];
#if defined(_WIN32)
    char exe[PATH_MAX];
    DWORD n = GetModuleFileNameA(NULL, exe, (DWORD)sizeof(exe));
    if (n == 0 || n >= sizeof(exe)) { dir[0] = '.'; dir[1] = '\0'; return dir; }
    strncpy(dir, exe, sizeof(dir) - 1); dir[sizeof(dir) - 1] = '\0';
#elif defined(__APPLE__)
    char raw[PATH_MAX];
    uint32_t size = (uint32_t)sizeof(raw);
    if (_NSGetExecutablePath(raw, &size) != 0) { dir[0] = '.'; dir[1] = '\0'; return dir; }
    char resolved[PATH_MAX];
    if (realpath(raw, resolved) == NULL) {
        strncpy(resolved, raw, sizeof(resolved) - 1); resolved[sizeof(resolved) - 1] = '\0';
    }
    strncpy(dir, resolved, sizeof(dir) - 1); dir[sizeof(dir) - 1] = '\0';
#else
    ssize_t n = readlink("/proc/self/exe", dir, sizeof(dir) - 1);
    if (n <= 0) { dir[0] = '.'; dir[1] = '\0'; return dir; }
    dir[n] = '\0';
#endif
    char *slash = strrchr(dir, '/');
#if defined(_WIN32)
    if (!slash) slash = strrchr(dir, '\\');
#endif
    if (slash) { *slash = '\0'; }
    else       { dir[0] = '.'; dir[1] = '\0'; }
    return dir;
}

int64_t ionic_target_is_linux(void) {
#ifdef __linux__
    return 1;
#else
    return 0;
#endif
}

int64_t ionic_cpu_core_count(void) {
#ifdef _WIN32
    SYSTEM_INFO si; GetSystemInfo(&si);
    return (int64_t)si.dwNumberOfProcessors;
#else
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    return (n > 0) ? (int64_t)n : 1;
#endif
}

/* ── File I/O builtins ────────────────────────────────────────────────────── */

/* A freshly allocated empty string.
 *
 * Every string-returning entry point in this runtime must allocate on *every*
 * path. The contract is "the caller owns the result", and Ionic has no way to
 * distinguish an owned empty string from a rodata literal — so returning `""`
 * from a failure path makes free()/rc_release() on the caller's value a free of
 * a string literal. See docs/memory-model.md §11.1.
 *
 * On OOM this returns NULL; that is the one case the contract cannot honour. */
static char *rt_empty_str(void) {
    char *e = (char *)malloc(1);
    if (e) e[0] = '\0';
    return e;
}

/* Copy a string into a fresh allocation. Used by builtins that hand out a
 * pointer to memory they do not own (a struct field, a static buffer, an
 * environment block). The caller-owns-it contract requires the copy: handing
 * back an interior pointer would let a release free the middle of someone
 * else's allocation. NULL and empty both map to an owned "". */
static char *rt_dup_str(const char *s) {
    if (!s) return rt_empty_str();
    size_t n = strlen(s) + 1;
    char *d = (char *)malloc(n);
    if (!d) return rt_empty_str();
    memcpy(d, s, n);
    return d;
}

/* Marked weak: LLVM IR path defines ionic_file_read inline.
 *
 * Every return path allocates (see rt_empty_str above). A failed open used to
 * return a rodata `""`, which made the ownership contract false on exactly the
 * path a program is most likely to hit. See docs/memory-model.md §11.1. */
__attribute__((weak))
char *ionic_file_read(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return rt_empty_str();
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz < 0) sz = 0;                      /* ftell failed: treat as empty */
    char *buf = (char *)malloc((size_t)(sz + 1));
    if (!buf) { fclose(f); return rt_empty_str(); }
    fread(buf, 1, (size_t)sz, f);
    buf[sz] = '\0';
    fclose(f);
    return buf;
}

int64_t ionic_system(const char *cmd) { return (int64_t)system(cmd); }

/* Byte length of a regular file, or -1 when the path is missing or is not a
   regular file. This exists so the compiler's backend discovery can ask "is
   libonnxruntime.so here?" cheaply. ionic_file_read() would do, but it mallocs
   the whole file and Ionic strings are never freed — probing a multi-hundred-
   megabyte shared library that way leaks all of it, once per probe attempt per
   compile. */
int64_t ionic_file_size(const char *path) {
    if (!path || !path[0]) return -1;
    struct stat st;
    if (stat(path, &st) != 0) return -1;
    if (!S_ISREG(st.st_mode)) return -1;
    return (int64_t)st.st_size;
}

/* ── Basic I/O builtins ───────────────────────────────────────────────────── */

int64_t ionic_println(const char *s) { puts(s ? s : ""); return 0; }
int64_t ionic_println_int64(int64_t v) { printf("%lld\n", (long long)v); return 0; }
int64_t ionic_print(const char *s) { fputs(s ? s : "", stdout); return 0; }
int64_t ionic_print_int64(int64_t v) { printf("%lld", (long long)v); return 0; }

/* ── Math builtins ────────────────────────────────────────────────────────── */
/* All float functions use int64 bit-pattern convention: args and return values
   are the IEEE 754 bit representation passed through GP registers (X0/X0).
   This avoids the ARM64 float ABI (D0) mismatch in the native codegen. */
static inline int64_t d2i64(double v) { int64_t b; memcpy(&b, &v, 8); return b; }
static inline double  i642d(int64_t b) { double v; memcpy(&v, &b, 8); return v; }

int64_t ionic_sqrt(int64_t b) { return d2i64(sqrt(i642d(b))); }
int64_t ionic_fabs(int64_t b) { return d2i64(fabs(i642d(b))); }
int64_t ionic_int64_to_float64(int64_t v) { return d2i64((double)v); }
int64_t ionic_float64_to_int64(int64_t b) { return (int64_t)i642d(b); }

/* Compile-time float arithmetic — used by the optimizer to fold floats when
   ionic_new runs. Same int64 bit-pattern ABI as all other float builtins.
   Return type is int64 (the bit pattern) so the optimizer can store results
   directly into nd_iv of freshly-built ND_FLOAT nodes. These are plain-named
   (no ionic_ prefix) so call sites resolve identically through both the old
   bootstrap (ionic_self) and the self-hosted ionic_new. */
int64_t fadd_bits(int64_t a, int64_t b) { return d2i64(i642d(a) + i642d(b)); }
int64_t fsub_bits(int64_t a, int64_t b) { return d2i64(i642d(a) - i642d(b)); }
int64_t fmul_bits(int64_t a, int64_t b) { return d2i64(i642d(a) * i642d(b)); }
int64_t fdiv_bits(int64_t a, int64_t b) { return d2i64(i642d(a) / i642d(b)); }
int64_t sqrt_bits(int64_t b)            { return d2i64(sqrt(i642d(b))); }
int64_t fabs_bits(int64_t b)            { return d2i64(fabs(i642d(b))); }
int64_t floor_bits(int64_t b)           { return d2i64(floor(i642d(b))); }
int64_t ceil_bits(int64_t b)            { return d2i64(ceil(i642d(b))); }
int64_t pow_bits(int64_t a, int64_t b)  { return d2i64(pow(i642d(a), i642d(b))); }
int64_t int_to_fbits(int64_t v)         { return d2i64((double)v); }

/* ── Float literal parsing (used internally by the Ionic parser) ─────────── */
/* No ionic_ prefix — ionic_self generates _str_to_float64_bits directly */
int64_t str_to_float64_bits(const char *s) {
    if (!s || !*s) return 0;
    return d2i64(strtod(s, NULL));
}

/* ── Conversion & hashing ─────────────────────────────────────────────────── */

int64_t ionic_str_to_int64(const char *s) {
    if (!s || !*s) return 0;
    return (int64_t)strtoll(s, NULL, 10);
}

int64_t ionic_str_hash(const char *s) {
    if (!s) return 0;
    int64_t h = 5381;
    unsigned char c;
    while ((c = (unsigned char)*s++)) h = ((h << 5) + h) ^ c;
    return h;
}

int64_t ionic_exit(int64_t code) { exit((int)code); return 0; }

int64_t ionic_min(int64_t a, int64_t b) { return a < b ? a : b; }
int64_t ionic_max(int64_t a, int64_t b) { return a > b ? a : b; }
int64_t ionic_abs(int64_t v)            { return v < 0 ? -v : v; }
int64_t ionic_pow(int64_t ab, int64_t eb) { return d2i64(pow(i642d(ab), i642d(eb))); }
int64_t ionic_floor(int64_t b)          { return d2i64(floor(i642d(b))); }
int64_t ionic_ceil(int64_t b)           { return d2i64(ceil(i642d(b))); }

/* ── String utilities ─────────────────────────────────────────────────────── */

char *ionic_read_line(void) {
    char buf[4096];
    if (!fgets(buf, sizeof(buf), stdin)) { char *r = (char *)malloc(1); r[0] = '\0'; return r; }
    size_t n = strlen(buf);
    if (n > 0 && buf[n-1] == '\n') buf[n-1] = '\0';
    char *r = (char *)malloc(strlen(buf) + 1);
    strcpy(r, buf);
    return r;
}

char *ionic_str_slice(const char *s, int64_t start, int64_t end) {
    if (!s) { char *r = (char *)malloc(1); r[0] = '\0'; return r; }
    int64_t n = (int64_t)strlen(s);
    if (start < 0) start = 0;
    if (end > n) end = n;
    if (start >= end) { char *r = (char *)malloc(1); r[0] = '\0'; return r; }
    int64_t len = end - start;
    char *r = (char *)malloc((size_t)(len + 1));
    memcpy(r, s + start, (size_t)len);
    r[len] = '\0';
    return r;
}

int64_t ionic_str_contains(const char *haystack, const char *needle) {
    if (!haystack || !needle) return 0;
    return strstr(haystack, needle) != NULL ? 1 : 0;
}

int64_t ionic_str_starts_with(const char *s, const char *prefix) {
    if (!s || !prefix) return 0;
    return strncmp(s, prefix, strlen(prefix)) == 0 ? 1 : 0;
}

char *ionic_str_replace(const char *s, const char *from, const char *to) {
    /* Every return path allocates. Returning the input here (the previous
     * behaviour) would mean a caller that owns its result can end up with two
     * names for one buffer, and a release/free of both is a double free.
     * Ionic strings are never NULL, so this path is only reachable from C
     * callers — but the contract has to hold for all of them.
     * See docs/memory-model.md §11.1. */
    if (!s || !from || !to) {
        const char *src = s ? s : "";
        size_t n = strlen(src);
        char *r = (char *)malloc(n + 1);
        memcpy(r, src, n + 1);
        return r;
    }
    size_t ls = strlen(s), lf = strlen(from), lt = strlen(to);
    if (lf == 0) { char *r = (char *)malloc(ls + 1); memcpy(r, s, ls + 1); return r; }
    /* Count occurrences */
    int count = 0;
    const char *p = s;
    while ((p = strstr(p, from))) { count++; p += lf; }
    size_t need = ls + (size_t)count * (lt > lf ? lt - lf : 0) + 1;
    char *r = (char *)malloc(need);
    char *out = r;
    p = s;
    const char *q;
    while ((q = strstr(p, from))) {
        size_t pre = (size_t)(q - p);
        memcpy(out, p, pre); out += pre;
        memcpy(out, to, lt); out += lt;
        p = q + lf;
    }
    size_t tail = strlen(p);
    memcpy(out, p, tail + 1);
    return r;
}

int64_t ionic_str_ends_with(const char *s, const char *suffix) {
    if (!s || !suffix) return 0;
    size_t ls = strlen(s), lp = strlen(suffix);
    if (lp > ls) return 0;
    return strcmp(s + ls - lp, suffix) == 0 ? 1 : 0;
}

/* ── String builtins ──────────────────────────────────────────────────────── */

/* These are also defined inline by the LLVM IR emitter; mark weak so the
   IR's strong definition wins when linking ionic_self via the LLVM path.
   For the native (ARM64) path there are no IR definitions, so these are used. */

__attribute__((weak))
char *ionic_str_concat(const char *a, const char *b) {
    if (!a) a = ""; if (!b) b = "";
    size_t la = strlen(a), lb = strlen(b);
    char *r = (char *)malloc(la + lb + 1);
    memcpy(r, a, la); memcpy(r + la, b, lb); r[la + lb] = '\0';
    return r;
}

int64_t ionic_str_eq(const char *a, const char *b) {
    if (!a) a = ""; if (!b) b = "";
    return strcmp(a, b) == 0 ? 1 : 0;
}

int64_t ionic_str_len(const char *s) {
    return s ? (int64_t)strlen(s) : 0;
}

int64_t ionic_str_index(const char *s, int64_t i) {
    /* O(1): do NOT strlen(s) here. This is called once per character by the
     * lexer and by line/offset scans; computing strlen on every call turned
     * those scans into O(n^2). Callers guard i against the string length
     * before calling (lx_peek does `pos >= slen`; diag_* loop `i < n`), so
     * only `i == len` (the NUL terminator) can ever be read here — which
     * returns 0, exactly as the previous bounds-checked version did. */
    if (!s || i < 0) return 0;
    return (int64_t)(unsigned char)s[i];
}

__attribute__((weak))
char *ionic_int64_to_str(int64_t v) {
    char *buf = (char *)malloc(32);
    snprintf(buf, 32, "%lld", (long long)v);
    return buf;
}

__attribute__((weak))
char *ionic_float64_to_str(int64_t b) {
    double v = i642d(b);
    char *buf = (char *)malloc(32);
    snprintf(buf, 32, "%g", v);
    return buf;
}

int64_t ionic_println_float64(int64_t b) { printf("%g\n",  i642d(b)); fflush(stdout); return 0; }
int64_t ionic_print_float64(int64_t b)   { printf("%g",    i642d(b)); fflush(stdout); return 0; }

/* format("{} has {} items", s, int64_to_str(n)) — replaces {} placeholders in order */
char *ionic_format(const char *fmt,
    const char *a0, const char *a1, const char *a2, const char *a3,
    const char *a4, const char *a5, const char *a6) {
    const char *args[7] = {a0, a1, a2, a3, a4, a5, a6};
    /* first pass: compute output length */
    size_t out_sz = 0;
    int ai = 0;
    for (const char *p = fmt ? fmt : ""; *p; p++) {
        if (*p == '{' && *(p+1) == '}') {
            if (ai < 7 && args[ai]) out_sz += strlen(args[ai]);
            ai++; p++;
        } else {
            out_sz++;
        }
    }
    char *out = (char *)malloc(out_sz + 1);
    char *op = out;
    ai = 0;
    for (const char *p = fmt ? fmt : ""; *p; p++) {
        if (*p == '{' && *(p+1) == '}') {
            const char *a = (ai < 7 && args[ai]) ? args[ai] : "";
            size_t al = strlen(a);
            memcpy(op, a, al); op += al;
            ai++; p++;
        } else {
            *op++ = *p;
        }
    }
    *op = '\0';
    return out;
}

__attribute__((weak))
char *ionic_char_to_str(int64_t c) {
    /* One raw byte from a code point 0..255. The lexer rebuilds string
     * literals byte-by-byte through this, so it must be byte-identity for
     * 0..255 (no UTF-8 encoding, no 7-bit masking) or non-ASCII source bytes
     * would be corrupted. UTF-8 encoding lives in Ionic (u_utf8 / tui_utf8). */
    char *buf = (char *)malloc(2);
    buf[0] = (char)(c & 0xFF); buf[1] = '\0';
    return buf;
}

/* ── Array builtins ───────────────────────────────────────────────────────── */

__attribute__((weak))
int64_t ionic_array_len(IonicArray *arr) { return arr ? arr->len : 0; }

static void ionic_bounds_panic(int64_t idx, int64_t len) {
    fprintf(stderr, "ionic: index out of bounds: index %lld, length %lld\n",
            (long long)idx, (long long)len);
    exit(1);
}

__attribute__((weak))
int64_t ionic_array_get(IonicArray *arr, int64_t i) {
    if (!arr) { fprintf(stderr, "ionic: index into null array\n"); exit(1); }
    if (i < 0 || i >= arr->len) ionic_bounds_panic(i, arr->len);
    return ((int64_t *)arr->data)[i];
}

__attribute__((weak))
int64_t ionic_array_set(IonicArray *arr, int64_t i, int64_t v) {
    if (!arr) { fprintf(stderr, "ionic: index into null array\n"); exit(1); }
    if (i < 0 || i >= arr->len) ionic_bounds_panic(i, arr->len);
    ((int64_t *)arr->data)[i] = v; return 0;
}

__attribute__((weak))
void ionic_array_push(IonicArray *arr, int64_t v) { ionic_array_push_i64(arr, v); }

/* ── Model format detection ───────────────────────────────────────────────── */

typedef enum {
    MODEL_FMT_UNKNOWN = 0,
    MODEL_FMT_ONNX,
    MODEL_FMT_GGUF,
    MODEL_FMT_PT,
    MODEL_FMT_H5,
    MODEL_FMT_MLMODEL,
    MODEL_FMT_SAFETENSORS,
} ModelFormat;

static ModelFormat detect_format(const char *path) {
    /* Extension check first */
    const char *dot = strrchr(path, '.');
    if (dot) {
        if (strcmp(dot, ".onnx")        == 0) return MODEL_FMT_ONNX;
        if (strcmp(dot, ".gguf")        == 0) return MODEL_FMT_GGUF;
        if (strcmp(dot, ".pt")          == 0) return MODEL_FMT_PT;
        if (strcmp(dot, ".pth")         == 0) return MODEL_FMT_PT;
        if (strcmp(dot, ".h5")          == 0) return MODEL_FMT_H5;
        if (strcmp(dot, ".mlmodel")     == 0) return MODEL_FMT_MLMODEL;
        if (strcmp(dot, ".safetensors") == 0) return MODEL_FMT_SAFETENSORS;
    }
    /* Magic-byte fallback for extension-less files (e.g. Ollama blobs).
     * Only GGUF has an unambiguous magic; an extension-less ONNX file is not
     * identifiable this way, so it needs its ".onnx" name.
     * Note: safetensors has no magic — its first 8 bytes are the little-endian
     * length of a JSON header, so it cannot be identified here. Give it an
     * explicit extension, or load it with mlx_load_safetensors(). */
    FILE *f = fopen(path, "rb");
    if (!f) return MODEL_FMT_UNKNOWN;
    unsigned char magic[8] = {0};
    fread(magic, 1, sizeof(magic), f);
    fclose(f);
    /* GGUF: "GGUF" */
    if (magic[0]=='G' && magic[1]=='G' && magic[2]=='U' && magic[3]=='F') return MODEL_FMT_GGUF;
    /* ONNX protobuf: starts with 0x08 field-tag (common for ModelProto) — heuristic only */
    /* ORT also has its own binary format; skip for now */
    return MODEL_FMT_UNKNOWN;
}

/* ════════════════════════════════════════════════════════════════════════════
 * ONNX Runtime backend
 * ════════════════════════════════════════════════════════════════════════════ */

#ifdef IONIC_HAVE_ORT

static const OrtApi *g_ort = NULL;

static void ort_check(OrtStatus *status, const char *ctx) {
    if (!status) return;
    const char *msg = g_ort->GetErrorMessage(status);
    fprintf(stderr, "[ionic/ort] %s: %s\n", ctx, msg);
    g_ort->ReleaseStatus(status);
    exit(1);
}

static void ensure_ort(void) {
    if (g_ort) return;
    g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    if (!g_ort) { fprintf(stderr, "[ionic/ort] API init failed\n"); exit(1); }
}

typedef struct {
    ModelFormat  fmt;       /* MODEL_FMT_ONNX */
    OrtEnv      *env;
    OrtSession  *session;
} OnnxHandle;

static void *ort_load(const char *path) {
    ensure_ort();
    OnnxHandle *h = calloc(1, sizeof(OnnxHandle));
    h->fmt = MODEL_FMT_ONNX;
    ort_check(g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "ionic", &h->env), "CreateEnv");
    OrtSessionOptions *opts = NULL;
    ort_check(g_ort->CreateSessionOptions(&opts), "CreateSessionOptions");
#ifdef __APPLE__
    OrtStatus *cml_st = OrtSessionOptionsAppendExecutionProvider_CoreML(opts, 0);
    if (cml_st) { g_ort->ReleaseStatus(cml_st); }
    else        { fprintf(stderr, "[ionic/ort] CoreML EP enabled\n"); }
#endif
    ort_check(g_ort->CreateSession(h->env, path, opts, &h->session), "CreateSession");
    g_ort->ReleaseSessionOptions(opts);
    fprintf(stderr, "[ionic/ort] loaded: %s\n", path);
    return h;
}

static void ort_free(void *model) {
    ensure_ort();
    OnnxHandle *h = model;
    if (h->session) g_ort->ReleaseSession(h->session);
    if (h->env)     g_ort->ReleaseEnv(h->env);
    free(h);
}

#endif /* IONIC_HAVE_ORT */


/* ════════════════════════════════════════════════════════════════════════════
 * MLX tensor backend
 *
 * This is the *brick*: a stable, thin, over-complete surface over Apple's MLX
 * array runtime so an Ionic program can build and run model math on the GPU.
 * It deliberately contains NO training algorithm — no SGD, no Adam, no loss
 * functions, no backward pass. What it does contain is every primitive such a
 * loop is assembled from, so the algorithm can live in ordinary `.ionic`
 * source instead of being baked into the runtime.
 *
 * Design contract
 * ---------------
 *  - A tensor is an opaque `MlxTensor*` handle, never a raw `mlx_array`. The
 *    handle owns exactly one array; `mlx_array` is refcounted, so copies made
 *    with `mlx_array_set` are cheap and safe.
 *  - Every op returns a *new* handle. Ops never mutate their inputs, so an
 *    expression graph is built by chaining. This matches MLX's functional
 *    model: `c = mlx_matmul(a, b)` leaves `a` and `b` valid and unchanged.
 *  - MLX is lazy. Nothing is computed until `mlx_eval` (or an op that must
 *    materialise, like `mlx_to_array`). Chaining 50 ops and evaluating once
 *    is the fast path; calling `mlx_eval` after every op is correct but slow.
 *  - Errors do NOT unwind. MLX reports via a callback and returns non-zero,
 *    leaving the output array *empty*; touching an empty array segfaults. So
 *    every entry point below funnels through `mlx_ck()`, which turns a failed
 *    call into a NULL handle (or a 0/NaN return) and never yields a
 *    half-built tensor that would crash later.
 *
 * Availability
 * ------------
 * Compiled only when IONIC_HAVE_MLX is defined AND we're on Apple (Metal).
 * On Linux every entry point is present but returns the "unavailable" value,
 * so an Ionic program that mentions mlx_* still *links* — it just reports at
 * runtime that the backend isn't compiled in.
 * ════════════════════════════════════════════════════════════════════════════ */

#if IONIC_MLX_ACTIVE

/* ── dtype mapping: Ionic int <-> mlx_dtype ───────────────────────────────────
 * Exposed to Ionic as small integers so dtype can be an int64 argument.
 * The numbers are our own (stable) ABI; mapping to the real enum happens here.
 * Do NOT pass mlx_dtype ordinals across the boundary — the enum ordering is
 * an MLX implementation detail and has already changed between releases. */
typedef enum {
    IONIC_DT_F32 = 0,
    IONIC_DT_F16 = 1,
    IONIC_DT_BF16 = 2,
    IONIC_DT_F64 = 3,
    IONIC_DT_I32 = 4,
    IONIC_DT_I64 = 5,
    IONIC_DT_BOOL = 6,
    IONIC_DT_U8 = 7,
} IonicDtype;

static mlx_dtype ionic_dtype_to_mlx(int64_t d) {
    switch (d) {
        case IONIC_DT_F16:  return MLX_FLOAT16;
        case IONIC_DT_BF16: return MLX_BFLOAT16;
        case IONIC_DT_F64:  return MLX_FLOAT64;
        case IONIC_DT_I32:  return MLX_INT32;
        case IONIC_DT_I64:  return MLX_INT64;
        case IONIC_DT_BOOL: return MLX_BOOL;
        case IONIC_DT_U8:   return MLX_UINT8;
        case IONIC_DT_F32:
        default:            return MLX_FLOAT32;
    }
}

static int64_t mlx_dtype_to_ionic(mlx_dtype d) {
    switch (d) {
        case MLX_FLOAT16:   return IONIC_DT_F16;
        case MLX_BFLOAT16:  return IONIC_DT_BF16;
        case MLX_FLOAT64:   return IONIC_DT_F64;
        case MLX_INT32:     return IONIC_DT_I32;
        case MLX_INT64:     return IONIC_DT_I64;
        case MLX_BOOL:      return IONIC_DT_BOOL;
        case MLX_UINT8:     return IONIC_DT_U8;
        case MLX_FLOAT32:
        default:            return IONIC_DT_F32;
    }
}

/* ── Error plumbing ─────────────────────────────────────────────────────────
 * MLX's default handler aborts the process. We install our own once, record
 * the last message, and let the calling entry point decide what to return.
 * `g_mlx_err` is sticky-until-read so a failure is diagnosable from Ionic via
 * mlx_last_error(). */
static char  g_mlx_err[512] = {0};
static int   g_mlx_inited = 0;

/* ── Memory pools: MLX-side state ─────────────────────────────────────────────
 * Pool ids and declared capacity live near the top of this file, outside the
 * MLX conditional, because the pool model is a *language* concept: it is checked
 * and enforced whether or not a backend is linked in. What remains here is the
 * MLX half — the per-pool streams and the live byte accounting.
 *
 * On Apple silicon (unified memory) both pools address the same physical bytes:
 * a transfer changes which pool the value is *charged to* and moves no data.
 * On a discrete backend the streams differ and the transfer copies. Keeping the
 * pool in the handle is what makes those two cases distinguishable at all.
 */
static mlx_stream g_mlx_cpu_stream;   /* SRAM stream; created at init */
static mlx_stream g_mlx_gpu_stream;   /* VRAM stream; created at init */

/* Bytes currently charged to each pool, maintained by mlx_tensor_new/free.
 * This is *our* accounting, deliberately independent of mlx_get_active_memory:
 * MLX's number is process-global and includes intermediate graph buffers, so it
 * cannot answer "how much is in VRAM" once the pools are meant to be separate. */
static int64_t g_pool_bytes[IONIC_POOL_COUNT] = {0, 0};

static int pool_valid(int64_t p) { return p == IONIC_POOL_CPU || p == IONIC_POOL_GPU; }

static const char *pool_name(int64_t p) {
    return p == IONIC_POOL_GPU ? "VRAM" : (p == IONIC_POOL_CPU ? "SRAM" : "none");
}

static void ionic_mlx_error_handler(const char *msg, void *data) {
    (void)data;
    snprintf(g_mlx_err, sizeof(g_mlx_err), "%s", msg ? msg : "(unknown MLX error)");
    /* Trim MLX's build-path suffix so messages read cleanly in Ionic output. */
    char *at = strstr(g_mlx_err, " at /private/tmp/");
    if (at) *at = '\0';
    fprintf(stderr, "[ionic/mlx] %s\n", g_mlx_err);
}

static void ionic_mlx_init(void) {
    if (g_mlx_inited) return;
    mlx_set_error_handler(ionic_mlx_error_handler, NULL, NULL);
    g_mlx_cpu_stream = mlx_default_cpu_stream_new();
    g_mlx_gpu_stream = mlx_default_gpu_stream_new();
    g_mlx_inited = 1;
}

/* Run an MLX call, normalising failure. Returns 0 on success, -1 on error.
 * On error the caller must NOT use `*out` — MLX leaves it empty/dirty. */
#define mlx_ck(expr) ( (expr) == 0 ? 0 : -1 )

/* ── Tensor handle ──────────────────────────────────────────────────────────
 * The first field is NOT ModelFormat: tensors are not models, and keeping the
 * layouts distinct means a tensor can never be passed to model_free() by
 * accident (handle_fmt would read garbage). Tensors use their own magic. */
#define IONIC_MLX_MAGIC 0x4D4C5854  /* "MLXT" */

typedef struct {
    uint32_t   magic;
    uint32_t   alive;      /* 0 once freed or moved away from             */
    mlx_array  a;
    int64_t    pool;       /* IONIC_POOL_CPU / IONIC_POOL_GPU / _NONE      */
    int64_t    bytes;      /* charged against `pool` while this is alive  */
    uint32_t   owns_array; /* 0 after a move: another handle holds `a`    */
} MlxTensor;

/* The stream an op on this pool must run on. Keeping the pool in the handle —
 * rather than one process-global stream — is what makes `tensor@cpu` and
 * `tensor@gpu` distinguishable at runtime at all. Before this, all 36 call
 * sites hardcoded the GPU stream, so `tensor@cpu` was already VRAM and a
 * transfer could not have meant anything even if it had been implemented. */
static mlx_stream pool_stream(int64_t pool) {
    return pool == IONIC_POOL_GPU ? g_mlx_gpu_stream : g_mlx_cpu_stream;
}

/* 1 when a transfer between the pools moves no bytes: on Apple silicon both
 * pools address the same physical memory, so only the accountancy changes.
 * A discrete backend would define IONIC_POOLS_DISCRETE and pay the copy. */
static int pools_unified(void) {
#if defined(IONIC_POOLS_DISCRETE)
    return 0;
#else
    bool metal = false;
    if (mlx_ck(mlx_metal_is_available(&metal)) != 0) return 1;
    return metal ? 1 : 0;
#endif
}

/* Report and abort when `bytes` would not fit in `pool`.
 *
 * This is Stage D's backstop. The static checker rejects what it can *prove* is
 * over budget and exempts the rest with a warning (see budget_end() in
 * src/semantic/checker.ionic), so the exemptions must be caught here or the
 * declaration is a fiction: a 64 MiB `vram` happily charged 16 GiB and the
 * program exited 0. Failing at the allocation is the point — the alternative is
 * an OOM hours later, or silent overcommit on a device that has one.
 *
 * exit(1) rather than a NULL handle: Ionic has no way to observe NULL from an
 * mlx_* call (every one returns a tensor or a scalar), so returning NULL would
 * turn an over-budget program into a segfault at an unrelated line — exactly
 * the "corrupt memory later" this exists to prevent. `ionic_bounds_panic`
 * makes the same choice for the same reason.
 *
 * Refusing rather than falling back is also decision §4.4a of the plan: a
 * program that believes it ran on the accelerator must never silently not have.
 */
static void pool_capacity_panic(int64_t pool, int64_t bytes) {
    fprintf(stderr,
        "ionic: accelerator pool exhausted: a %lld-byte allocation does not fit "
        "%s (%lld of %lld bytes in use)\n"
        "       the declared budget came from `let vram = %lld;`; the static "
        "checker could not size this value, so what it measured was a lower "
        "bound. Raise the pool, narrow the scope's fraction, or free what is no "
        "longer live.\n",
        (long long)bytes, pool_name(pool),
        (long long)g_pool_bytes[pool], (long long)g_pool_capacity[pool],
        (long long)g_pool_capacity[pool]);
    exit(1);
}

/* Charge `bytes` to `pool`, refusing the allocation if it would not fit.
 *
 * Release is unconditional: giving bytes back can never over-fill. The
 * comparison avoids `used + bytes` so a huge request cannot overflow into a
 * negative sum and slip past the check. */
static void pool_charge(int64_t pool, int64_t bytes) {
    if (!pool_valid(pool)) return;
    if (bytes > 0 && g_pool_capacity[pool] > 0
        && bytes > g_pool_capacity[pool] - g_pool_bytes[pool]) {
        pool_capacity_panic(pool, bytes);
    }
    g_pool_bytes[pool] += bytes;
    if (g_pool_bytes[pool] < 0) g_pool_bytes[pool] = 0;  /* defensive */
}

/* Construct a handle in a given pool. Sizes the array so the pool's occupancy
 * is answerable from Ionic (`mlx_pool_bytes`). Constructors charge SRAM; only a
 * transfer charges VRAM, which is the operation a capacity budget bounds. */
static MlxTensor *mlx_tensor_new_at(mlx_array a, int64_t pool) {
    MlxTensor *t = (MlxTensor *)calloc(1, sizeof(MlxTensor));
    if (!t) return NULL;
    t->magic      = IONIC_MLX_MAGIC;
    t->alive      = 1;
    t->a          = a;
    t->pool       = pool;
    t->bytes      = (int64_t)mlx_array_nbytes(a);
    t->owns_array = 1;
    pool_charge(pool, t->bytes);
    return t;
}

/* Default: SRAM. Every constructor and every safetensors lookup lands here,
 * matching the checker's rule that constructors take no tensor operand and so
 * have no pool to infer one from (docs/plan-placement.md §3.3a). */
static MlxTensor *mlx_tensor_new(mlx_array a) {
    return mlx_tensor_new_at(a, IONIC_POOL_CPU);
}

/* Validate a handle coming from Ionic. A dangling/NULL/mistyped pointer is a
 * hard programmer error, so we report and return NULL rather than crash. */
static MlxTensor *mlx_tensor_of(void *h) {
    if (!h) { fprintf(stderr, "[ionic/mlx] null tensor handle\n"); return NULL; }
    MlxTensor *t = (MlxTensor *)h;
    if (t->magic != IONIC_MLX_MAGIC) {
        fprintf(stderr, "[ionic/mlx] bad tensor handle (not an mlx tensor)\n");
        return NULL;
    }
    if (!t->alive) {
        fprintf(stderr, "[ionic/mlx] use of a freed tensor\n");
        return NULL;
    }
    return t;
}

/* Unwrap for ops: returns the mlx_array, or bails out of the enclosing
 * function with NULL. Every op below is a `void *`-returning entry point, so
 * this is safe to use uniformly. */
#define MLX_ARG(h) \
    MlxTensor *_t = mlx_tensor_of(h); \
    if (!_t) return NULL; \
    mlx_array  _a    = _t->a; \
    int64_t    _pool = _t->pool; \
    mlx_stream _s    = pool_stream(_pool); \
    (void)_a; (void)_s

/* ── Shape helper ───────────────────────────────────────────────────────────
 * Ionic passes shapes as an `[int64]` array. MLX wants `const int*`. Convert
 * into a caller-provided stack buffer (capped so a malformed input can't
 * overflow). Returns the number of dims, or -1 on error. */
#define IONIC_MLX_MAX_DIMS 8

static int mlx_shape_from_arr(void *shape_arr, int *out, size_t *n_out) {
    if (!shape_arr) { *n_out = 0; return 0; }
    IonicArray *sa = (IonicArray *)shape_arr;
    if (sa->len > IONIC_MLX_MAX_DIMS) {
        fprintf(stderr, "[ionic/mlx] shape has %lld dims, max is %d\n",
                (long long)sa->len, IONIC_MLX_MAX_DIMS);
        return -1;
    }
    int64_t *slots = (int64_t *)sa->data;
    for (int64_t i = 0; i < sa->len; i++) out[i] = (int)slots[i];
    *n_out = (size_t)sa->len;
    return 0;
}

/* ── Construction ─────────────────────────────────────────────────────────── */

void *ionic_mlx_zeros(void *shape_arr, int64_t dtype) {
    ionic_mlx_init();
    int shape[IONIC_MLX_MAX_DIMS]; size_t n = 0;
    if (mlx_shape_from_arr(shape_arr, shape, &n) != 0) return NULL;
    mlx_array r = mlx_array_new();
    if (mlx_ck(mlx_zeros(&r, shape, n, ionic_dtype_to_mlx(dtype), g_mlx_cpu_stream)) != 0) return NULL;
    return mlx_tensor_new(r);
}

void *ionic_mlx_ones(void *shape_arr, int64_t dtype) {
    ionic_mlx_init();
    int shape[IONIC_MLX_MAX_DIMS]; size_t n = 0;
    if (mlx_shape_from_arr(shape_arr, shape, &n) != 0) return NULL;
    mlx_array r = mlx_array_new();
    if (mlx_ck(mlx_ones(&r, shape, n, ionic_dtype_to_mlx(dtype), g_mlx_cpu_stream)) != 0) return NULL;
    return mlx_tensor_new(r);
}

/* Float arguments arrive as int64 IEEE-754 bit patterns in GP registers, not in
 * D0-D7: the native codegen has no float register ABI and every other float
 * builtin in this runtime uses the same convention (see d2i64/i642d above).
 * Declaring these parameters `double` would read the wrong register bank — the
 * compiler would pass 4.0 in X1 while the callee looked in D1. */
void *ionic_mlx_full(void *shape_arr, int64_t val_bits, int64_t dtype) {
    ionic_mlx_init();
    int shape[IONIC_MLX_MAX_DIMS]; size_t n = 0;
    if (mlx_shape_from_arr(shape_arr, shape, &n) != 0) return NULL;
    mlx_array s = mlx_array_new_float32((float)i642d(val_bits));
    mlx_array r = mlx_array_new();
    int rc = mlx_ck(mlx_full(&r, shape, n, s, ionic_dtype_to_mlx(dtype), g_mlx_cpu_stream));
    mlx_array_free(s);
    if (rc != 0) return NULL;
    return mlx_tensor_new(r);
}

void *ionic_mlx_arange(int64_t start_b, int64_t stop_b, int64_t step_b, int64_t dtype) {
    ionic_mlx_init();
    mlx_array r = mlx_array_new();
    if (mlx_ck(mlx_arange(&r, i642d(start_b), i642d(stop_b), i642d(step_b),
                          ionic_dtype_to_mlx(dtype), g_mlx_cpu_stream)) != 0) return NULL;
    return mlx_tensor_new(r);
}

void *ionic_mlx_eye(int64_t n, int64_t m, int64_t dtype) {
    ionic_mlx_init();
    mlx_array r = mlx_array_new();
    if (mlx_ck(mlx_eye(&r, (int)n, (int)m, 0, ionic_dtype_to_mlx(dtype), g_mlx_cpu_stream)) != 0) return NULL;
    return mlx_tensor_new(r);
}

/* Random init — what parameter tensors are seeded with. This is a primitive,
 * not an initialisation policy: no Xavier/He scaling rule is applied here. */
void *ionic_mlx_random_normal(void *shape_arr, int64_t loc_b, int64_t scale_b, int64_t dtype) {
    ionic_mlx_init();
    int shape[IONIC_MLX_MAX_DIMS]; size_t n = 0;
    if (mlx_shape_from_arr(shape_arr, shape, &n) != 0) return NULL;
    mlx_array r = mlx_array_new();
    if (mlx_ck(mlx_random_normal(&r, shape, n, ionic_dtype_to_mlx(dtype),
                                 (float)i642d(loc_b), (float)i642d(scale_b), mlx_array_empty,
                                 g_mlx_cpu_stream)) != 0) return NULL;
    return mlx_tensor_new(r);
}

void *ionic_mlx_random_uniform(void *shape_arr, int64_t lo_b, int64_t hi_b, int64_t dtype) {
    ionic_mlx_init();
    int shape[IONIC_MLX_MAX_DIMS]; size_t n = 0;
    if (mlx_shape_from_arr(shape_arr, shape, &n) != 0) return NULL;
    mlx_array low  = mlx_array_new_float32((float)i642d(lo_b));
    mlx_array high = mlx_array_new_float32((float)i642d(hi_b));
    mlx_array r    = mlx_array_new();
    int rc = mlx_ck(mlx_random_uniform(&r, low, high, shape, n,
                                       ionic_dtype_to_mlx(dtype), mlx_array_empty,
                                       g_mlx_cpu_stream));
    mlx_array_free(low);
    mlx_array_free(high);
    if (rc != 0) return NULL;
    return mlx_tensor_new(r);
}

/* Build a tensor from a flat [float64] Ionic array plus a shape. This is the
 * bridge from ordinary Ionic data into GPU math. Values are copied, so the
 * caller may free/mutate the source array afterwards. */
void *ionic_mlx_from_array(void *flat_arr, void *shape_arr, int64_t dtype) {
    ionic_mlx_init();
    if (!flat_arr) { fprintf(stderr, "[ionic/mlx] from_array: null array\n"); return NULL; }
    int shape[IONIC_MLX_MAX_DIMS]; size_t n = 0;
    if (mlx_shape_from_arr(shape_arr, shape, &n) != 0) return NULL;

    IonicArray *fa = (IonicArray *)flat_arr;
    size_t total = 1;
    for (size_t i = 0; i < n; i++) total *= (size_t)(shape[i] > 0 ? shape[i] : 0);
    if ((int64_t)total != fa->len) {
        fprintf(stderr, "[ionic/mlx] from_array: shape implies %zu elements, array has %lld\n",
                total, (long long)fa->len);
        return NULL;
    }

    int64_t *slots = (int64_t *)fa->data;
    mlx_dtype mdt = ionic_dtype_to_mlx(dtype);
    int rc;

    /* Fast path: f32/f64 buffer straight in. Ionic stores float64s as their
     * bit pattern in i64 slots, so f64 needs no conversion at all. */
    if (mdt == MLX_FLOAT64) {
        mlx_array r = mlx_array_new_data(slots, shape, (int)n, MLX_FLOAT64);
        return mlx_tensor_new(r);
    }
    if (mdt == MLX_FLOAT32) {
        float *buf = (float *)malloc(total * sizeof(float));
        if (!buf) return NULL;
        for (size_t i = 0; i < total; i++) {
            double d; memcpy(&d, &slots[i], 8);
            buf[i] = (float)d;
        }
        mlx_array r = mlx_array_new_data(buf, shape, (int)n, MLX_FLOAT32);
        free(buf);              /* mlx_array_new_data copies the buffer */
        return mlx_tensor_new(r);
    }
    if (mdt == MLX_INT32 || mdt == MLX_INT64) {
        /* Integers arrive as plain i64 values (not bit patterns). */
        if (mdt == MLX_INT64) {
            mlx_array r = mlx_array_new_data(slots, shape, (int)n, MLX_INT64);
            return mlx_tensor_new(r);
        }
        int32_t *buf = (int32_t *)malloc(total * sizeof(int32_t));
        if (!buf) return NULL;
        for (size_t i = 0; i < total; i++) buf[i] = (int32_t)slots[i];
        mlx_array r = mlx_array_new_data(buf, shape, (int)n, MLX_INT32);
        free(buf);
        return mlx_tensor_new(r);
    }

    /* Generic path: build as f64 then cast. */
    mlx_array base = mlx_array_new_data(slots, shape, (int)n, MLX_FLOAT64);
    mlx_array r    = mlx_array_new();
    rc = mlx_ck(mlx_astype(&r, base, mdt, g_mlx_cpu_stream));
    mlx_array_free(base);
    if (rc != 0) return NULL;
    return mlx_tensor_new(r);
}

/* ── Materialisation and inspection ───────────────────────────────────────── */

/* Force the graph to run. Idempotent; safe on already-evaluated tensors. */
int64_t ionic_mlx_eval(void *h) {
    MlxTensor *t = mlx_tensor_of(h);
    if (!t) return -1;
    if (mlx_ck(mlx_array_eval(t->a)) != 0) return -1;
    return 0;
}

int64_t ionic_mlx_ndim(void *h) {
    MlxTensor *t = mlx_tensor_of(h);
    if (!t) return -1;
    return (int64_t)mlx_array_ndim(t->a);
}

/* Element count (product of dims), i.e. what a flattened [float64] needs. */
int64_t ionic_mlx_size(void *h) {
    MlxTensor *t = mlx_tensor_of(h);
    if (!t) return -1;
    return (int64_t)mlx_array_size(t->a);
}

/* Dimension `i`, or -1 if out of range. */
int64_t ionic_mlx_dim(void *h, int64_t i) {
    MlxTensor *t = mlx_tensor_of(h);
    if (!t) return -1;
    int nd = (int)mlx_array_ndim(t->a);
    if (i < 0 || i >= nd) return -1;
    return (int64_t)mlx_array_shape(t->a)[i];
}

/* Shape as an [int64] Ionic array, so the same value can be fed back into
 * reshape/zeros — the shapes compose. */
void *ionic_mlx_shape(void *h) {
    MlxTensor *t = mlx_tensor_of(h);
    if (!t) return ionic_array_alloc(1);
    int nd = (int)mlx_array_ndim(t->a);
    const int *sh = mlx_array_shape(t->a);
    IonicArray *out = ionic_array_alloc(nd > 0 ? nd : 1);
    for (int i = 0; i < nd; i++) ionic_array_push_i64(out, (int64_t)sh[i]);
    return out;
}

int64_t ionic_mlx_dtype(void *h) {
    MlxTensor *t = mlx_tensor_of(h);
    if (!t) return -1;
    return mlx_dtype_to_ionic(mlx_array_dtype(t->a));
}

/* Copy the tensor out into a flat [float64] Ionic array (row-major).
 * Evaluates first. Any dtype is converted to float64; use mlx_dtype() if you
 * need to know what it was. Note the element order is MLX's row-major order. */
void *ionic_mlx_to_array(void *h) {
    MlxTensor *t = mlx_tensor_of(h);
    if (!t) return ionic_array_alloc(1);
    if (mlx_ck(mlx_array_eval(t->a)) != 0) return ionic_array_alloc(1);

    size_t total = mlx_array_size(t->a);
    IonicArray *out = ionic_array_alloc((int64_t)(total > 0 ? total : 1));
    mlx_dtype dt = mlx_array_dtype(t->a);

    /* Read natively where possible, then widen to double bit patterns. */
    switch (dt) {
        case MLX_FLOAT32: {
            const float *p = mlx_array_data_float32(t->a);
            if (!p) break;
            for (size_t i = 0; i < total; i++) ionic_array_push_i64(out, d2i64((double)p[i]));
            return out;
        }
        case MLX_FLOAT64: {
            const double *p = mlx_array_data_float64(t->a);
            if (!p) break;
            for (size_t i = 0; i < total; i++) ionic_array_push_i64(out, d2i64(p[i]));
            return out;
        }
        case MLX_INT64: {
            const int64_t *p = mlx_array_data_int64(t->a);
            if (!p) break;
            for (size_t i = 0; i < total; i++) ionic_array_push_i64(out, d2i64((double)p[i]));
            return out;
        }
        case MLX_INT32: {
            const int32_t *p = mlx_array_data_int32(t->a);
            if (!p) break;
            for (size_t i = 0; i < total; i++) ionic_array_push_i64(out, d2i64((double)p[i]));
            return out;
        }
        case MLX_UINT8: {
            const uint8_t *p = mlx_array_data_uint8(t->a);
            if (!p) break;
            for (size_t i = 0; i < total; i++) ionic_array_push_i64(out, d2i64((double)p[i]));
            return out;
        }
        case MLX_BOOL: {
            const bool *p = mlx_array_data_bool(t->a);
            if (!p) break;
            for (size_t i = 0; i < total; i++) ionic_array_push_i64(out, d2i64(p[i] ? 1.0 : 0.0));
            return out;
        }
        default: break;   /* f16/bf16: go through a cast below */
    }

    /* Fallback: cast to f32 in-graph, then read. */
    mlx_array c = mlx_array_new();
    if (mlx_ck(mlx_astype(&c, t->a, MLX_FLOAT32, pool_stream(t->pool))) == 0
        && mlx_ck(mlx_array_eval(c)) == 0) {
        const float *p = mlx_array_data_float32(c);
        if (p) {
            for (size_t i = 0; i < total; i++) ionic_array_push_i64(out, d2i64((double)p[i]));
        }
    }
    mlx_array_free(c);
    return out;
}

/* Scalar extraction for size-1 tensors (a "loss value" read-out). Returns the
 * float64 bit pattern, matching every other float builtin in this runtime.
 * Returns 0 for non-scalar input; check mlx_size() first if that matters.
 *
 * The dtype must be dispatched on: mlx_array_item_float32() reinterprets the
 * stored bytes rather than converting, so reading an int32 tensor through it
 * yields the float whose bit pattern happens to match (argmax's 1 becomes
 * 1.4e-45) instead of 1.0. */
int64_t ionic_mlx_item(void *h) {
    MlxTensor *t = mlx_tensor_of(h);
    if (!t) return 0;
    if (mlx_array_size(t->a) != 1) {
        fprintf(stderr, "[ionic/mlx] item: tensor has %zu elements, not 1\n",
                mlx_array_size(t->a));
        return 0;
    }
    if (mlx_ck(mlx_array_eval(t->a)) != 0) return 0;

    switch (mlx_array_dtype(t->a)) {
        case MLX_INT64:  { int64_t v = 0; if (mlx_ck(mlx_array_item_int64(&v, t->a))   != 0) return 0; return d2i64((double)v); }
        case MLX_INT32:  { int32_t v = 0; if (mlx_ck(mlx_array_item_int32(&v, t->a))   != 0) return 0; return d2i64((double)v); }
        case MLX_INT16:  { int16_t v = 0; if (mlx_ck(mlx_array_item_int16(&v, t->a))   != 0) return 0; return d2i64((double)v); }
        case MLX_UINT32: { uint32_t v = 0; if (mlx_ck(mlx_array_item_uint32(&v, t->a)) != 0) return 0; return d2i64((double)v); }
        case MLX_UINT8:  { uint8_t v = 0; if (mlx_ck(mlx_array_item_uint8(&v, t->a))   != 0) return 0; return d2i64((double)v); }
        case MLX_BOOL:   { bool v = false; if (mlx_ck(mlx_array_item_bool(&v, t->a))   != 0) return 0; return d2i64(v ? 1.0 : 0.0); }
        case MLX_FLOAT64: { double v = 0.0; if (mlx_ck(mlx_array_item_float64(&v, t->a)) != 0) return 0; return d2i64(v); }
        case MLX_FLOAT32: default: {
            float v = 0.0f;
            if (mlx_ck(mlx_array_item_float32(&v, t->a)) != 0) return 0;
            return d2i64((double)v);
        }
    }
}

/* ── Lifetime ─────────────────────────────────────────────────────────────── */

/* Free a tensor. Null-tolerant and idempotent-ish: a second free on the same
 * handle is caught by the `alive` flag rather than double-freeing the array.
 *
 * A handle that no longer owns its array (it was moved away from) releases
 * nothing: the bytes belong to the handle the transfer produced. Releasing the
 * array twice would be a double free of the same refcount, which MLX turns
 * into a use-after-free the moment either handle is read. */
void ionic_mlx_free(void *h) {
    MlxTensor *t = mlx_tensor_of(h);
    if (!t) return;
    pool_charge(t->pool, -t->bytes);
    if (t->owns_array) mlx_array_free(t->a);
    t->alive = 0;
    free(t);
}

/* ── Transfers between pools ─────────────────────────────────────────────────
 * `.toGpu()` / `.toCpu()` compile to these. Before them, codegen emitted the
 * operand's value unchanged (`native.ionic:1539`), so a "transfer" was a no-op
 * and `tensor@cpu`/`tensor@gpu` were the same pool at runtime no matter what
 * the checker proved about them.
 *
 * The semantics are the ones docs/plan-placement.md §4.3 settled on: a
 * transfer is a MOVE. The destination handle takes the array reference over
 * and the source is left not owning it, so the checker's use-after-move rule
 * has something real behind it — the source's bytes are unaccounted from its
 * pool and accounted to the destination's.
 *
 * On unified memory (Apple silicon) both pools address the same physical bytes,
 * so no data is copied and the array reference is handed over as-is. On a
 * discrete backend the value is copied onto the destination stream and the
 * source's storage is released; that is what `pools_unified()` distinguishes.
 * `mlx_copy` is the primitive either way, so the two paths differ only in
 * whether the source is reclaimed.
 */
static void *mlx_transfer(void *h, int64_t to_pool) {
    MlxTensor *t = mlx_tensor_of(h);
    if (!t) return NULL;

    if (t->pool == to_pool) {
        /* Already there. Returning a *new* handle rather than `h` keeps the
         * move contract uniform: the caller always ends up owning the result,
         * and the source is always dead afterwards. */
        t->alive      = 0;
        t->owns_array = 0;
        pool_charge(t->pool, -t->bytes);
        return mlx_tensor_new_at(t->a, to_pool);
    }

    int64_t from_pool = t->pool;
    mlx_array moved;

    if (pools_unified()) {
        /* Same physical bytes, different accountancy. Bump the refcount so the
         * destination handle owns a reference of its own, then drop the
         * source's. */
        moved = mlx_array_new();
        if (mlx_ck(mlx_array_set(&moved, t->a)) != 0) return NULL;
    } else {
        /* Discrete pools: the value must physically land on the destination
         * stream before the source storage can be reclaimed. */
        moved = mlx_array_new();
        if (mlx_ck(mlx_copy(&moved, t->a, pool_stream(to_pool))) != 0) return NULL;
        if (mlx_ck(mlx_array_eval(moved)) != 0) {
            mlx_array_free(moved);
            return NULL;
        }
    }

    /* Move, not copy: the source stops owning its array and its bytes leave the
     * source pool. The checker already rejects reading the source afterwards;
     * this makes the runtime agree instead of leaving two handles over one
     * allocation. */
    t->alive      = 0;
    t->owns_array = 0;
    pool_charge(from_pool, -t->bytes);
    mlx_array_free(t->a);

    return mlx_tensor_new_at(moved, to_pool);
}

/* Ionic-facing: `.toGpu()` / `.toCpu()`. Both take the source handle and return
 * the destination handle; codegen reassigns the slot to the result. */
void *ionic_mlx_to_gpu(void *h) { return mlx_transfer(h, IONIC_POOL_GPU); }
void *ionic_mlx_to_cpu(void *h) { return mlx_transfer(h, IONIC_POOL_CPU); }

/* ── Pool introspection ──────────────────────────────────────────────────────
 * These are what make the pool model observable from Ionic, and therefore
 * testable. Without them a transfer that silently did nothing would look
 * exactly like one that worked: the existing master test only ever checked
 * that the *types* line up.
 *
 *   mlx_pool(h)         -> IONIC_POOL_CPU (0) / IONIC_POOL_GPU (1) / -1
 *   mlx_pool_bytes(p)   -> bytes currently charged to pool p, or -1
 *   mlx_pool_count()    -> number of pools (2)
 */
int64_t ionic_mlx_pool(void *h) {
    MlxTensor *t = mlx_tensor_of(h);
    if (!t) return -1;
    return t->pool;
}

int64_t ionic_mlx_pool_bytes(int64_t pool) {
    if (!pool_valid(pool)) return -1;
    return g_pool_bytes[pool];
}

int64_t ionic_mlx_pool_count(void) { return IONIC_POOL_COUNT; }

/* Whether a transfer between the two pools is free on this backend. Ionic uses
 * this to assert "unified memory means the byte count does not move" without
 * hard-coding the answer per platform. */
int64_t ionic_mlx_pools_unified(void) { return pools_unified(); }

/* ── Elementwise binary ops ───────────────────────────────────────────────── */

#define MLX_BINOP_IMPL(fname, mlxfn)                                       \
    void *fname(void *ha, void *hb) {                                      \
        ionic_mlx_init();                                                  \
        MLX_ARG(ha);                                                       \
        MlxTensor *_tb = mlx_tensor_of(hb);                                \
        if (!_tb) return NULL;                                             \
        mlx_array _r = mlx_array_new();                                    \
        if (mlx_ck(mlxfn(&_r, _a, _tb->a,                                  \
                         _s)) != 0) return NULL;             \
        return mlx_tensor_new_at(_r, _pool);                                         \
    }

MLX_BINOP_IMPL(ionic_mlx_add,      mlx_add)
MLX_BINOP_IMPL(ionic_mlx_subtract, mlx_subtract)
MLX_BINOP_IMPL(ionic_mlx_multiply, mlx_multiply)
MLX_BINOP_IMPL(ionic_mlx_divide,   mlx_divide)
MLX_BINOP_IMPL(ionic_mlx_power,    mlx_power)
MLX_BINOP_IMPL(ionic_mlx_maximum,  mlx_maximum)
MLX_BINOP_IMPL(ionic_mlx_minimum,  mlx_minimum)
MLX_BINOP_IMPL(ionic_mlx_matmul,   mlx_matmul)

#undef MLX_BINOP_IMPL

/* ── Elementwise unary ops ────────────────────────────────────────────────── */

#define MLX_UNOP_IMPL(fname, mlxfn)                                        \
    void *fname(void *h) {                                                 \
        ionic_mlx_init();                                                  \
        MLX_ARG(h);                                                        \
        mlx_array _r = mlx_array_new();                                    \
        if (mlx_ck(mlxfn(&_r, _a, _s)) != 0) return NULL;     \
        return mlx_tensor_new_at(_r, _pool);                                         \
    }

MLX_UNOP_IMPL(ionic_mlx_negative, mlx_negative)
MLX_UNOP_IMPL(ionic_mlx_square,   mlx_square)
MLX_UNOP_IMPL(ionic_mlx_sqrt,     mlx_sqrt)
MLX_UNOP_IMPL(ionic_mlx_exp,      mlx_exp)
MLX_UNOP_IMPL(ionic_mlx_log,      mlx_log)
MLX_UNOP_IMPL(ionic_mlx_abs,      mlx_abs)
MLX_UNOP_IMPL(ionic_mlx_tanh,     mlx_tanh)
MLX_UNOP_IMPL(ionic_mlx_erf,      mlx_erf)

#undef MLX_UNOP_IMPL

/* contiguous takes an extra bool in MLX; wrap it so the Ionic surface stays
 * uniform (one argument). */
void *ionic_mlx_contiguous(void *h) {
    ionic_mlx_init();
    MLX_ARG(h);
    mlx_array r = mlx_array_new();
    if (mlx_ck(mlx_contiguous(&r, _a, false, _s)) != 0) return NULL;
    return mlx_tensor_new_at(r, _pool);
}

/* ── Reductions ───────────────────────────────────────────────────────────── */

/* Sum of every element (optionally keeping dims). The classic
 * "sum the squared errors" ingredient. */
void *ionic_mlx_sum(void *h, int64_t keepdims) {
    ionic_mlx_init();
    MLX_ARG(h);
    mlx_array r = mlx_array_new();
    if (mlx_ck(mlx_sum(&r, _a, keepdims != 0, _s)) != 0) return NULL;
    return mlx_tensor_new_at(r, _pool);
}

void *ionic_mlx_mean(void *h, int64_t keepdims) {
    ionic_mlx_init();
    MLX_ARG(h);
    mlx_array r = mlx_array_new();
    if (mlx_ck(mlx_mean(&r, _a, keepdims != 0, _s)) != 0) return NULL;
    return mlx_tensor_new_at(r, _pool);
}

void *ionic_mlx_max(void *h, int64_t keepdims) {
    ionic_mlx_init();
    MLX_ARG(h);
    mlx_array r = mlx_array_new();
    if (mlx_ck(mlx_max(&r, _a, keepdims != 0, _s)) != 0) return NULL;
    return mlx_tensor_new_at(r, _pool);
}

void *ionic_mlx_min(void *h, int64_t keepdims) {
    ionic_mlx_init();
    MLX_ARG(h);
    mlx_array r = mlx_array_new();
    if (mlx_ck(mlx_min(&r, _a, keepdims != 0, _s)) != 0) return NULL;
    return mlx_tensor_new_at(r, _pool);
}

/* Reduction along one axis. axis may be negative (Python-style from the end),
 * which is what softmax over the last axis needs. */
void *ionic_mlx_sum_axis(void *h, int64_t axis, int64_t keepdims) {
    ionic_mlx_init();
    MLX_ARG(h);
    mlx_array r = mlx_array_new();
    if (mlx_ck(mlx_sum_axis(&r, _a, (int)axis, keepdims != 0, _s)) != 0) return NULL;
    return mlx_tensor_new_at(r, _pool);
}

void *ionic_mlx_mean_axis(void *h, int64_t axis, int64_t keepdims) {
    ionic_mlx_init();
    MLX_ARG(h);
    mlx_array r = mlx_array_new();
    if (mlx_ck(mlx_mean_axis(&r, _a, (int)axis, keepdims != 0, _s)) != 0) return NULL;
    return mlx_tensor_new_at(r, _pool);
}

void *ionic_mlx_argmax(void *h, int64_t axis, int64_t keepdims) {
    ionic_mlx_init();
    MLX_ARG(h);
    mlx_array r = mlx_array_new();
    if (mlx_ck(mlx_argmax_axis(&r, _a, (int)axis, keepdims != 0, _s)) != 0) return NULL;
    return mlx_tensor_new_at(r, _pool);
}

/* Softmax over an axis — the sampling/logits step of any generator. */
void *ionic_mlx_softmax(void *h, int64_t axis) {
    ionic_mlx_init();
    MLX_ARG(h);
    mlx_array r = mlx_array_new();
    if (mlx_ck(mlx_softmax_axis(&r, _a, (int)axis, false, _s)) != 0) return NULL;
    return mlx_tensor_new_at(r, _pool);
}

/* Log-softmax along an axis. MLX has no fused log_softmax, so this composes
 * softmax -> log on the graph (still lazy, still one eval at the end). Kept as
 * a named primitive because cross-entropy code reads better for it; it is not
 * the numerically-stable logsumexp form, which callers can build by hand from
 * mlx_max / mlx_exp / mlx_sum_axis. */
void *ionic_mlx_log_softmax(void *h, int64_t axis) {
    ionic_mlx_init();
    MLX_ARG(h);
    mlx_array sm = mlx_array_new();
    if (mlx_ck(mlx_softmax_axis(&sm, _a, (int)axis, false, _s)) != 0) return NULL;
    mlx_array r = mlx_array_new();
    int rc = mlx_ck(mlx_log(&r, sm, _s));
    mlx_array_free(sm);
    if (rc != 0) return NULL;
    return mlx_tensor_new_at(r, _pool);
}

/* ── Shape manipulation ───────────────────────────────────────────────────── */

void *ionic_mlx_reshape(void *h, void *shape_arr) {
    ionic_mlx_init();
    MLX_ARG(h);
    int shape[IONIC_MLX_MAX_DIMS]; size_t n = 0;
    if (mlx_shape_from_arr(shape_arr, shape, &n) != 0) return NULL;
    mlx_array r = mlx_array_new();
    if (mlx_ck(mlx_reshape(&r, _a, shape, n, _s)) != 0) return NULL;
    return mlx_tensor_new_at(r, _pool);
}

/* Transpose given an explicit permutation of dims ([1,0] for a 2-D matrix).
 * MLX rejects an empty axis list, so a full reversal is the sensible default
 * when the caller passes no permutation. */
void *ionic_mlx_transpose(void *h, void *axes_arr) {
    ionic_mlx_init();
    MLX_ARG(h);
    int axes[IONIC_MLX_MAX_DIMS]; size_t n = 0;
    if (mlx_shape_from_arr(axes_arr, axes, &n) != 0) return NULL;

    int nd = (int)mlx_array_ndim(_a);
    int rev[IONIC_MLX_MAX_DIMS];
    if (n == 0) {
        if (nd <= 0 || nd > IONIC_MLX_MAX_DIMS) {
            fprintf(stderr, "[ionic/mlx] transpose: bad rank %d\n", nd);
            return NULL;
        }
        for (int i = 0; i < nd; i++) rev[i] = nd - 1 - i;
        mlx_array r = mlx_array_new();
        if (mlx_ck(mlx_transpose_axes(&r, _a, rev, (size_t)nd, _s)) != 0) return NULL;
        return mlx_tensor_new_at(r, _pool);
    }

    mlx_array r = mlx_array_new();
    if (mlx_ck(mlx_transpose_axes(&r, _a, axes, n, _s)) != 0) return NULL;
    return mlx_tensor_new_at(r, _pool);
}

/* Cast element dtype. Kept separate from construction so precision can be
 * chosen per-step (e.g. f32 activations, bf16 weights). */
void *ionic_mlx_astype(void *h, int64_t dtype) {
    ionic_mlx_init();
    MLX_ARG(h);
    mlx_array r = mlx_array_new();
    if (mlx_ck(mlx_astype(&r, _a, ionic_dtype_to_mlx(dtype), _s)) != 0) return NULL;
    return mlx_tensor_new_at(r, _pool);
}

/* Slice: take elements [start, stop) along `axis`, leaving other dims whole.
 * MLX's raw mlx_slice takes full-rank start/stop/strides vectors, so we build
 * them here from the single-axis form the Ionic side exposes. */
void *ionic_mlx_slice(void *h, int64_t start, int64_t stop, int64_t axis) {
    ionic_mlx_init();
    MLX_ARG(h);
    int nd = (int)mlx_array_ndim(_a);
    if (nd <= 0 || nd > IONIC_MLX_MAX_DIMS) {
        fprintf(stderr, "[ionic/mlx] slice: bad rank %d\n", nd);
        return NULL;
    }
    int ax = (int)axis;
    if (ax < 0) ax += nd;
    if (ax < 0 || ax >= nd) {
        fprintf(stderr, "[ionic/mlx] slice: axis %lld out of range for rank %d\n",
                (long long)axis, nd);
        return NULL;
    }

    const int *shape = mlx_array_shape(_a);
    int st[IONIC_MLX_MAX_DIMS], sp[IONIC_MLX_MAX_DIMS], sd[IONIC_MLX_MAX_DIMS];
    for (int i = 0; i < nd; i++) { st[i] = 0; sp[i] = shape[i]; sd[i] = 1; }
    st[ax] = (int)start;
    sp[ax] = (int)stop;

    /* strides must have the same length as start/stop; passing NULL with
     * strides_num 0 makes MLX reject the call as a rank mismatch. */
    mlx_array r = mlx_array_new();
    if (mlx_ck(mlx_slice(&r, _a, st, (size_t)nd, sp, (size_t)nd, sd, (size_t)nd,
                         _s)) != 0) return NULL;
    return mlx_tensor_new_at(r, _pool);
}

/* Gather rows by index along an axis (embedding lookup / token selection). */
void *ionic_mlx_take(void *h, void *idx_arr, int64_t axis) {
    ionic_mlx_init();
    MLX_ARG(h);
    if (!idx_arr) { fprintf(stderr, "[ionic/mlx] take: null indices\n"); return NULL; }
    IonicArray *ia = (IonicArray *)idx_arr;
    int32_t *buf = (int32_t *)malloc((size_t)(ia->len > 0 ? ia->len : 1) * sizeof(int32_t));
    if (!buf) return NULL;
    int64_t *slots = (int64_t *)ia->data;
    for (int64_t i = 0; i < ia->len; i++) buf[i] = (int32_t)slots[i];
    int ish[1] = { (int)ia->len };
    mlx_array idx = mlx_array_new_data(buf, ish, 1, MLX_INT32);
    free(buf);

    mlx_array r = mlx_array_new();
    int rc = mlx_ck(mlx_take_axis(&r, _a, idx, (int)axis, _s));
    mlx_array_free(idx);
    if (rc != 0) return NULL;
    return mlx_tensor_new_at(r, _pool);
}

/* Concatenate two tensors along an axis. mlx_concatenate() only exists as the
 * flat (axis 0) convenience form; the explicit-axis variant is a separate
 * symbol, so don't silently drop the argument. */
void *ionic_mlx_concatenate(void *ha, void *hb, int64_t axis) {
    ionic_mlx_init();
    MLX_ARG(ha);
    MlxTensor *tb = mlx_tensor_of(hb);
    if (!tb) return NULL;
    mlx_array pair[2] = { _a, tb->a };
    mlx_vector_array va = mlx_vector_array_new_data(pair, 2);
    mlx_array r = mlx_array_new();
    int rc = mlx_ck(mlx_concatenate_axis(&r, va, (int)axis, _s));
    mlx_vector_array_free(va);
    if (rc != 0) return NULL;
    return mlx_tensor_new_at(r, _pool);
}

/* ── NN primitives ──────────────────────────────────────────────────────────
 * These are the fused Metal kernels a transformer needs. They are building
 * blocks: no attention mask policy, no layer stack, no initialisation scheme.
 * ------------------------------------------------------------------------ */

/* Scaled dot-product attention. `causal` selects MLX's "causal" mask mode,
 * which is the only mask policy exposed — anything finer-grained should be
 * built from the tensor ops above. */
void *ionic_mlx_sdpa(void *q, void *k, void *v, int64_t scale_b, int64_t causal) {
    ionic_mlx_init();
    MlxTensor *tq = mlx_tensor_of(q);
    MlxTensor *tk = mlx_tensor_of(k);
    MlxTensor *tv = mlx_tensor_of(v);
    if (!tq || !tk || !tv) return NULL;
    mlx_array r = mlx_array_new();
    if (mlx_ck(mlx_fast_scaled_dot_product_attention(
            &r, tq->a, tk->a, tv->a, (float)i642d(scale_b),
            causal ? "causal" : "", mlx_array_empty, mlx_array_empty,
            pool_stream(tq->pool))) != 0) return NULL;
    return mlx_tensor_new_at(r, tq->pool);
}

/* RMSNorm. `weight` may be NULL for the weightless form. */
void *ionic_mlx_rms_norm(void *h, void *weight, int64_t eps_b) {
    ionic_mlx_init();
    MLX_ARG(h);
    mlx_array w = mlx_array_empty;
    MlxTensor *tw = NULL;
    if (weight) {
        tw = mlx_tensor_of(weight);
        if (!tw) return NULL;
        w = tw->a;
    }
    mlx_array r = mlx_array_new();
    if (mlx_ck(mlx_fast_rms_norm(&r, _a, w, (float)i642d(eps_b), _s)) != 0) return NULL;
    return mlx_tensor_new_at(r, _pool);
}

/* LayerNorm. weight and bias may both be NULL. */
void *ionic_mlx_layer_norm(void *h, void *weight, void *bias, int64_t eps_b) {
    ionic_mlx_init();
    MLX_ARG(h);
    mlx_array w = mlx_array_empty, b = mlx_array_empty;
    if (weight) { MlxTensor *t = mlx_tensor_of(weight); if (!t) return NULL; w = t->a; }
    if (bias)   { MlxTensor *t = mlx_tensor_of(bias);   if (!t) return NULL; b = t->a; }
    mlx_array r = mlx_array_new();
    if (mlx_ck(mlx_fast_layer_norm(&r, _a, w, b, (float)i642d(eps_b), _s)) != 0) return NULL;
    return mlx_tensor_new_at(r, _pool);
}

/* Rotary position embedding. `dims` is the number of dimensions to rotate;
 * `traditional` picks the GPT-NeoX vs. original sign convention. */
void *ionic_mlx_rope(void *h, int64_t dims, int64_t base_b, int64_t traditional, int64_t offset) {
    ionic_mlx_init();
    MLX_ARG(h);
    mlx_optional_float opt;
    opt.value = (float)i642d(base_b);
    opt.has_value = true;
    mlx_array r = mlx_array_new();
    if (mlx_ck(mlx_fast_rope(&r, _a, (int)dims, traditional != 0, opt, 1.0f,
                             (int)offset, mlx_array_empty, _s)) != 0) return NULL;
    return mlx_tensor_new_at(r, _pool);
}

/* ── Autodiff seam (documented, not yet wired) ──────────────────────────────
 * MLX computes gradients through C closures:
 *
 *     mlx_closure loss = mlx_closure_new_func(loss_fn);
 *     mlx_value_and_grad(&vg, loss, argnums, n);
 *
 * `loss_fn` is a C function pointer called by MLX. Ionic has no function
 * values, so exposing mlx_value_and_grad directly requires teaching codegen to
 * take the address of a user function (it already emits one defined symbol per
 * `fn`, so this is mechanical: ADRP+ADD over the existing symbol).
 *
 * That bridge is intentionally NOT built here: it changes the language surface
 * (a new "function reference" value) and deserves its own design pass. Until
 * then gradient support is absent by design, not by accident — the primitives
 * above are exactly the set a hand-written forward/backward would use.
 * ------------------------------------------------------------------------- */

/* ── Diagnostics & memory ─────────────────────────────────────────────────── */

/* Last MLX error message (empty when the last op succeeded). */
const char *ionic_mlx_last_error(void) { return g_mlx_err; }

int64_t ionic_mlx_clear_error(void) { g_mlx_err[0] = '\0'; return 0; }

/* 1 when the backend is compiled in, 0 otherwise. Lets a program branch
 * instead of hard-failing on a platform without MLX. */
int64_t ionic_mlx_available(void) {
    return 1;
}

/* 1 when a Metal device is actually present (e.g. 0 under Rosetta / CI). */
int64_t ionic_mlx_metal_available(void) {
    ionic_mlx_init();
    bool avail = false;
    if (mlx_ck(mlx_metal_is_available(&avail)) != 0) return 0;
    return avail ? 1 : 0;
}

/* Bytes of live array memory, and the high-water mark since process start. */
int64_t ionic_mlx_active_memory(void) {
    ionic_mlx_init();
    size_t v = 0;
    if (mlx_ck(mlx_get_active_memory(&v)) != 0) return -1;
    return (int64_t)v;
}

int64_t ionic_mlx_peak_memory(void) {
    ionic_mlx_init();
    size_t v = 0;
    if (mlx_ck(mlx_get_peak_memory(&v)) != 0) return -1;
    return (int64_t)v;
}

int64_t ionic_mlx_reset_peak_memory(void) {
    ionic_mlx_init();
    return mlx_ck(mlx_reset_peak_memory());
}

/* Release cached buffers back to the OS. Useful in a loop so the allocator's
 * cache does not grow without bound (this repo gates on RSS). */
int64_t ionic_mlx_clear_cache(void) {
    ionic_mlx_init();
    return mlx_ck(mlx_clear_cache());
}

/* Cap how much memory MLX may allocate. 0 means "no limit". */
int64_t ionic_mlx_set_memory_limit(int64_t bytes) {
    ionic_mlx_init();
    size_t prev = 0;
    return mlx_ck(mlx_set_memory_limit(&prev, (size_t)bytes));
}

/* ── Safetensors I/O ────────────────────────────────────────────────────────
 * Load a .safetensors file and expose (a) the number of tensors and (b) the
 * Nth tensor by index, so Ionic can walk the whole dict without a hash map
 * crossing the FFI boundary. A named lookup is also provided for the common
 * "fetch this one weight" case.
 *
 * Handles own the underlying map; mlx_weights_free releases everything.
 * ------------------------------------------------------------------------ */

typedef struct {
    uint32_t                 magic;
    mlx_map_string_to_array  params;
    mlx_map_string_to_string meta;
} MlxWeights;

#define IONIC_MLX_WEIGHTS_MAGIC 0x4D4C5857  /* "MLXW" */

static MlxWeights *mlx_weights_of(void *h) {
    if (!h) { fprintf(stderr, "[ionic/mlx] null weights handle\n"); return NULL; }
    MlxWeights *w = (MlxWeights *)h;
    if (w->magic != IONIC_MLX_WEIGHTS_MAGIC) {
        fprintf(stderr, "[ionic/mlx] bad weights handle\n");
        return NULL;
    }
    return w;
}

void *ionic_mlx_load_safetensors(const char *path) {
    ionic_mlx_init();
    MlxWeights *w = (MlxWeights *)calloc(1, sizeof(MlxWeights));
    w->magic  = IONIC_MLX_WEIGHTS_MAGIC;
    w->params = mlx_map_string_to_array_new();
    w->meta   = mlx_map_string_to_string_new();

    if (mlx_ck(mlx_load_safetensors(&w->params, &w->meta, path, g_mlx_cpu_stream)) != 0) {
        mlx_map_string_to_array_free(w->params);
        mlx_map_string_to_string_free(w->meta);
        free(w);
        return NULL;
    }
    fprintf(stderr, "[ionic/mlx] loaded safetensors: %s\n", path);
    return w;
}

/* Number of tensors in the file. mlx-c exposes no map size, so we count by
 * walking the iterator (cheap: it only visits keys, does not materialise
 * data). */
static int64_t mlx_weights_count_impl(MlxWeights *w) {
    mlx_map_string_to_array_iterator it =
        mlx_map_string_to_array_iterator_new(w->params);
    const char *key = NULL;
    mlx_array   val = mlx_array_new();
    int64_t     n   = 0;
    while (mlx_map_string_to_array_iterator_next(&key, &val, it) == 0) {
        mlx_array_free(val);
        val = mlx_array_new();
        n++;
    }
    mlx_array_free(val);
    mlx_map_string_to_array_iterator_free(it);
    return n;
}

int64_t ionic_mlx_weights_count(void *h) {
    MlxWeights *w = mlx_weights_of(h);
    if (!w) return -1;
    return mlx_weights_count_impl(w);
}

/* Tensor at index i, or NULL. Iteration order is MLX's map order. */
void *ionic_mlx_weights_at(void *h, int64_t i) {
    MlxWeights *w = mlx_weights_of(h);
    if (!w) return NULL;

    mlx_map_string_to_array_iterator it = mlx_map_string_to_array_iterator_new(w->params);
    const char *key = NULL;
    mlx_array   val = mlx_array_new();
    void       *result = NULL;
    int64_t     idx = 0;

    while (mlx_map_string_to_array_iterator_next(&key, &val, it) == 0) {
        if (idx == i) {
            /* val owns a reference from the iterator; wrap it in a tensor.
             * The tensor takes the reference over, so it stays valid after
             * the iterator is freed. */
            result = mlx_tensor_new(val);
            val = mlx_array_new();   /* don't free the one we handed off */
            break;
        }
        mlx_array_free(val);
        val = mlx_array_new();
        idx++;
    }

    mlx_array_free(val);
    mlx_map_string_to_array_iterator_free(it);
    if (!result) {
        fprintf(stderr, "[ionic/mlx] weights_at: index %lld out of range (%lld tensors)\n",
                (long long)i, (long long)mlx_weights_count_impl(w));
    }
    return result;
}

/* Key at index i, or "" — pair with weights_at to walk the dict.
 *
 * Returns a fresh copy, not `keybuf`. A static buffer would be clobbered by
 * the next call, so `a = key_at(w, 0); b = key_at(w, 1)` left `a` reading as
 * `b` — and the caller owns a TY_STR, so there is nothing else it could hold.
 * See docs/memory-model.md §11.2. */
char *ionic_mlx_weights_key_at(void *h, int64_t i) {
    MlxWeights *w = mlx_weights_of(h);
    if (!w) return rt_empty_str();

    static char keybuf[512];
    keybuf[0] = '\0';
    mlx_map_string_to_array_iterator it = mlx_map_string_to_array_iterator_new(w->params);
    const char *key = NULL;
    mlx_array   val = mlx_array_new();
    int64_t     idx = 0;

    while (mlx_map_string_to_array_iterator_next(&key, &val, it) == 0) {
        if (idx == i) {
            snprintf(keybuf, sizeof(keybuf), "%s", key ? key : "");
            break;
        }
        mlx_array_free(val);
        val = mlx_array_new();
        idx++;
    }

    mlx_array_free(val);
    mlx_map_string_to_array_iterator_free(it);
    return rt_dup_str(keybuf);
}

/* Look up one tensor by name, or NULL when absent. */
void *ionic_mlx_weights_get(void *h, const char *name) {
    MlxWeights *w = mlx_weights_of(h);
    if (!w) return NULL;
    mlx_array v = mlx_array_new();
    if (mlx_ck(mlx_map_string_to_array_get(&v, w->params, name)) != 0) return NULL;
    return mlx_tensor_new(v);
}

void ionic_mlx_weights_free(void *h) {
    MlxWeights *w = mlx_weights_of(h);
    if (!w) return;
    mlx_map_string_to_array_free(w->params);
    mlx_map_string_to_string_free(w->meta);
    w->magic = 0;
    free(w);
}

/* Save a single tensor as .safetensors under one key. */
int64_t ionic_mlx_save_safetensors(void *h, const char *path, const char *key) {
    ionic_mlx_init();
    MlxTensor *t = mlx_tensor_of(h);
    if (!t) return -1;
    mlx_map_string_to_array  params = mlx_map_string_to_array_new();
    mlx_map_string_to_string meta   = mlx_map_string_to_string_new();
    mlx_map_string_to_array_insert(params, key ? key : "tensor", t->a);
    int rc = mlx_ck(mlx_save_safetensors(path, params, meta));
    mlx_map_string_to_array_free(params);
    mlx_map_string_to_string_free(meta);
    return rc;
}

#else  /* !IONIC_MLX_ACTIVE — Linux / non-Apple, or MLX not requested */

/* Every MLX entry point exists on every platform so Ionic programs that use
 * mlx_* still *link* — it just reports at runtime that the backend isn't
 * compiled in.
 *
 * The stub set is generated by hand from the real definitions above; keep the
 * two in sync when adding an op. A missing stub is a LINK error on Linux, not a
 * runtime error, which is why this list is exhaustive rather than minimal.
 *
 * Return conventions follow the real implementations so callers behave the same
 * way: `void *` ops yield NULL, count/length ops yield -1, predicates yield 0.
 */
static void mlx_unavailable(void) {
    static int warned = 0;
    if (warned) return;
    warned = 1;
    fprintf(stderr, "[ionic/mlx] MLX backend not compiled in on this platform "
                    "(Apple silicon + mlx-c required)\n");
}

/* ── Predicates / diagnostics ─────────────────────────────────────────────── */
int64_t ionic_mlx_available(void) { return 0; }
int64_t ionic_mlx_metal_available(void) { return 0; }
const char *ionic_mlx_last_error(void) { return "MLX backend not compiled in"; }
int64_t ionic_mlx_clear_error(void) { return 0; }
int64_t ionic_mlx_active_memory(void) { return -1; }
int64_t ionic_mlx_peak_memory(void) { return -1; }
int64_t ionic_mlx_reset_peak_memory(void) { return -1; }
int64_t ionic_mlx_clear_cache(void) { return 0; }
int64_t ionic_mlx_set_memory_limit(int64_t b) { (void)b; return -1; }

/* ── Lifetime / inspection ────────────────────────────────────────────────── */
void    ionic_mlx_free(void *h) { (void)h; }
int64_t ionic_mlx_eval(void *h) { (void)h; mlx_unavailable(); return -1; }
int64_t ionic_mlx_ndim(void *h) { (void)h; return -1; }
int64_t ionic_mlx_size(void *h) { (void)h; return -1; }
int64_t ionic_mlx_dim(void *h, int64_t i) { (void)h; (void)i; return -1; }
int64_t ionic_mlx_dtype(void *h) { (void)h; return -1; }
int64_t ionic_mlx_item(void *h) { (void)h; return 0; }
void   *ionic_mlx_shape(void *h) { (void)h; return ionic_array_alloc(1); }
void   *ionic_mlx_to_array(void *h) { (void)h; return ionic_array_alloc(1); }

/* ── Pools (Stage C) ─────────────────────────────────────────────────────────
 * The pool model is language-level, so its *introspection* must exist even
 * where no backend does: a program can ask whether transfers are free and how
 * much is charged to each pool on any platform. What it cannot do without a
 * backend is transfer anything — hence NULL from to_gpu/to_cpu, which the
 * checker's placement rules have already made unreachable in practice. */
void   *ionic_mlx_to_gpu(void *h) { (void)h; mlx_unavailable(); return NULL; }
void   *ionic_mlx_to_cpu(void *h) { (void)h; mlx_unavailable(); return NULL; }
int64_t ionic_mlx_pool(void *h) { (void)h; return -1; }
int64_t ionic_mlx_pool_bytes(int64_t p) { (void)p; return -1; }
/* No ionic_mlx_pool_capacity stub is needed here: capacity is a language-level
 * fact, so it lives outside the MLX conditional with ionic_pool_set_capacity and
 * answers correctly on every platform. A program that declares `vram` can read
 * it back even where the backend is absent. */
int64_t ionic_mlx_pool_count(void) { return IONIC_POOL_COUNT; }
int64_t ionic_mlx_pools_unified(void) { return 0; }

/* ── Construction ─────────────────────────────────────────────────────────── */
void *ionic_mlx_zeros(void *shape, int64_t dt)               { (void)shape; (void)dt; mlx_unavailable(); return NULL; }
void *ionic_mlx_ones(void *shape, int64_t dt)                { (void)shape; (void)dt; mlx_unavailable(); return NULL; }
void *ionic_mlx_full(void *shape, int64_t v, int64_t dt)     { (void)shape; (void)v; (void)dt; mlx_unavailable(); return NULL; }
void *ionic_mlx_arange(int64_t a, int64_t b, int64_t c, int64_t dt) { (void)a;(void)b;(void)c;(void)dt; mlx_unavailable(); return NULL; }
void *ionic_mlx_eye(int64_t n, int64_t m, int64_t dt)        { (void)n; (void)m; (void)dt; mlx_unavailable(); return NULL; }
void *ionic_mlx_random_normal(void *shape, int64_t l, int64_t s, int64_t dt)  { (void)shape;(void)l;(void)s;(void)dt; mlx_unavailable(); return NULL; }
void *ionic_mlx_random_uniform(void *shape, int64_t lo, int64_t hi, int64_t dt) { (void)shape;(void)lo;(void)hi;(void)dt; mlx_unavailable(); return NULL; }
void *ionic_mlx_from_array(void *flat, void *shape, int64_t dt) { (void)flat; (void)shape; (void)dt; mlx_unavailable(); return NULL; }

/* ── Elementwise / linear algebra ─────────────────────────────────────────── */
void *ionic_mlx_add(void *a, void *b)          { (void)a; (void)b; mlx_unavailable(); return NULL; }
void *ionic_mlx_subtract(void *a, void *b)     { (void)a; (void)b; mlx_unavailable(); return NULL; }
void *ionic_mlx_multiply(void *a, void *b)     { (void)a; (void)b; mlx_unavailable(); return NULL; }
void *ionic_mlx_divide(void *a, void *b)       { (void)a; (void)b; mlx_unavailable(); return NULL; }
void *ionic_mlx_power(void *a, void *b)        { (void)a; (void)b; mlx_unavailable(); return NULL; }
void *ionic_mlx_maximum(void *a, void *b)      { (void)a; (void)b; mlx_unavailable(); return NULL; }
void *ionic_mlx_minimum(void *a, void *b)      { (void)a; (void)b; mlx_unavailable(); return NULL; }
void *ionic_mlx_matmul(void *a, void *b)       { (void)a; (void)b; mlx_unavailable(); return NULL; }
void *ionic_mlx_concatenate(void *a, void *b, int64_t ax) { (void)a; (void)b; (void)ax; mlx_unavailable(); return NULL; }

void *ionic_mlx_negative(void *h)   { (void)h; mlx_unavailable(); return NULL; }
void *ionic_mlx_square(void *h)     { (void)h; mlx_unavailable(); return NULL; }
void *ionic_mlx_sqrt(void *h)       { (void)h; mlx_unavailable(); return NULL; }
void *ionic_mlx_exp(void *h)        { (void)h; mlx_unavailable(); return NULL; }
void *ionic_mlx_log(void *h)        { (void)h; mlx_unavailable(); return NULL; }
void *ionic_mlx_abs(void *h)        { (void)h; mlx_unavailable(); return NULL; }
void *ionic_mlx_tanh(void *h)       { (void)h; mlx_unavailable(); return NULL; }
void *ionic_mlx_erf(void *h)        { (void)h; mlx_unavailable(); return NULL; }
void *ionic_mlx_contiguous(void *h) { (void)h; mlx_unavailable(); return NULL; }

/* ── Reductions ───────────────────────────────────────────────────────────── */
void *ionic_mlx_sum(void *h, int64_t kd)   { (void)h; (void)kd; mlx_unavailable(); return NULL; }
void *ionic_mlx_mean(void *h, int64_t kd)  { (void)h; (void)kd; mlx_unavailable(); return NULL; }
void *ionic_mlx_max(void *h, int64_t kd)   { (void)h; (void)kd; mlx_unavailable(); return NULL; }
void *ionic_mlx_min(void *h, int64_t kd)   { (void)h; (void)kd; mlx_unavailable(); return NULL; }
void *ionic_mlx_sum_axis(void *h, int64_t ax, int64_t kd)  { (void)h;(void)ax;(void)kd; mlx_unavailable(); return NULL; }
void *ionic_mlx_mean_axis(void *h, int64_t ax, int64_t kd) { (void)h;(void)ax;(void)kd; mlx_unavailable(); return NULL; }
void *ionic_mlx_argmax(void *h, int64_t ax, int64_t kd)    { (void)h;(void)ax;(void)kd; mlx_unavailable(); return NULL; }
void *ionic_mlx_softmax(void *h, int64_t ax)      { (void)h; (void)ax; mlx_unavailable(); return NULL; }
void *ionic_mlx_log_softmax(void *h, int64_t ax)  { (void)h; (void)ax; mlx_unavailable(); return NULL; }

/* ── Shape ────────────────────────────────────────────────────────────────── */
void *ionic_mlx_reshape(void *h, void *shape)   { (void)h; (void)shape; mlx_unavailable(); return NULL; }
void *ionic_mlx_transpose(void *h, void *axes)  { (void)h; (void)axes; mlx_unavailable(); return NULL; }
void *ionic_mlx_astype(void *h, int64_t dt)     { (void)h; (void)dt; mlx_unavailable(); return NULL; }
void *ionic_mlx_slice(void *h, int64_t a, int64_t b, int64_t ax) { (void)h;(void)a;(void)b;(void)ax; mlx_unavailable(); return NULL; }
void *ionic_mlx_take(void *h, void *idx, int64_t ax) { (void)h; (void)idx; (void)ax; mlx_unavailable(); return NULL; }

/* ── NN primitives ────────────────────────────────────────────────────────── */
void *ionic_mlx_sdpa(void *q, void *k, void *v, int64_t sc, int64_t causal) {
    (void)q; (void)k; (void)v; (void)sc; (void)causal; mlx_unavailable(); return NULL;
}
void *ionic_mlx_rms_norm(void *h, void *w, int64_t eps)          { (void)h;(void)w;(void)eps; mlx_unavailable(); return NULL; }
void *ionic_mlx_layer_norm(void *h, void *w, void *b, int64_t eps) { (void)h;(void)w;(void)b;(void)eps; mlx_unavailable(); return NULL; }
void *ionic_mlx_rope(void *h, int64_t d, int64_t base, int64_t trad, int64_t off) {
    (void)h; (void)d; (void)base; (void)trad; (void)off; mlx_unavailable(); return NULL;
}

/* ── Safetensors ──────────────────────────────────────────────────────────── */
void      *ionic_mlx_load_safetensors(const char *path) { (void)path; mlx_unavailable(); return NULL; }
void       ionic_mlx_weights_free(void *h)              { (void)h; }
int64_t    ionic_mlx_weights_count(void *h)             { (void)h; return -1; }
void      *ionic_mlx_weights_at(void *h, int64_t i)     { (void)h; (void)i; return NULL; }
char       *ionic_mlx_weights_key_at(void *h, int64_t i) { (void)h; (void)i; return rt_empty_str(); }
void      *ionic_mlx_weights_get(void *h, const char *n) { (void)h; (void)n; return NULL; }
int64_t    ionic_mlx_save_safetensors(void *h, const char *p, const char *k) {
    (void)h; (void)p; (void)k; return -1;
}

#endif /* IONIC_MLX_ACTIVE */

/* ════════════════════════════════════════════════════════════════════════════
 * Public API — dispatch on ModelFormat stored in handle
 * ════════════════════════════════════════════════════════════════════════════ */

/* The first field of every handle struct is ModelFormat fmt — safe to cast */
static ModelFormat handle_fmt(void *model) {
    if (!model) return MODEL_FMT_UNKNOWN;
    return *(ModelFormat *)model;
}

void *ionic_load_model(const char *path) {
    ModelFormat fmt = detect_format(path);
    switch (fmt) {
#ifdef IONIC_HAVE_ORT
        case MODEL_FMT_ONNX:
            return ort_load(path);
#endif
        case MODEL_FMT_GGUF:
            /* Recognised, but there is no runner: llama.cpp was removed because
             * its only placement control (n_gpu_layers) is a single load-time
             * integer and cannot express per-value tensor residency. Use an
             * ONNX export, or wait for the low-level ggml path. */
            fprintf(stderr, "[ionic] load_model: GGUF inference was removed with llama.cpp; "
                            "export the model to ONNX instead\n");
            return NULL;
        case MODEL_FMT_PT:
            fprintf(stderr, "[ionic] load_model: PyTorch (.pt/.pth) not yet supported\n");
            return NULL;
        case MODEL_FMT_H5:
            fprintf(stderr, "[ionic] load_model: Keras/HDF5 (.h5) not yet supported\n");
            return NULL;
        case MODEL_FMT_MLMODEL:
            fprintf(stderr, "[ionic] load_model: CoreML (.mlmodel) not yet supported\n");
            return NULL;
        default:
            fprintf(stderr, "[ionic] load_model: unknown format '%s'\n", path);
            return NULL;
    }
}

void ionic_model_free(void *model) {
    if (!model) return;
    switch (handle_fmt(model)) {
#ifdef IONIC_HAVE_ORT
        case MODEL_FMT_ONNX: ort_free(model); break;
#endif
        default: free(model); break;
    }
}

void *ionic_model_forward(void *model, void *input) {
    if (!model) return input;
    fprintf(stderr, "[ionic] ionic_model_forward: model handles are backend-specific; "
                    "use ionic_piper_forward for ONNX\n");
    return input;
}

/* ── Piper TTS forward (ONNX) ─────────────────────────────────────────────── */

void *ionic_piper_forward(void *model,
                          void *phoneme_arr,
                          double noise_scale,
                          double length_scale,
                          double noise_w) {
#ifdef IONIC_HAVE_ORT
    ensure_ort();
    if (!model || handle_fmt(model) != MODEL_FMT_ONNX) {
        fprintf(stderr, "[ionic] piper_forward: not an ONNX model\n");
        return ionic_array_alloc(1);
    }

    OnnxHandle  *h   = model;
    IonicArray  *arr = phoneme_arr;
    int64_t      T   = arr->len;
    int64_t     *ids = (int64_t *)arr->data;

    OrtMemoryInfo *mem_info = NULL;
    ort_check(g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &mem_info),
              "CreateCpuMemoryInfo");

    int64_t shape_input[2] = {1, T};
    OrtValue *ov_input = NULL;
    ort_check(g_ort->CreateTensorWithDataAsOrtValue(
        mem_info, ids, (size_t)(T * sizeof(int64_t)),
        shape_input, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &ov_input), "input tensor");

    int64_t lengths_data[1] = {T};
    int64_t shape_len[1]    = {1};
    OrtValue *ov_lengths = NULL;
    ort_check(g_ort->CreateTensorWithDataAsOrtValue(
        mem_info, lengths_data, sizeof(int64_t),
        shape_len, 1, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &ov_lengths), "lengths tensor");

    float scales_data[3] = {(float)noise_scale, (float)length_scale, (float)noise_w};
    int64_t shape_scales[1] = {3};
    OrtValue *ov_scales = NULL;
    ort_check(g_ort->CreateTensorWithDataAsOrtValue(
        mem_info, scales_data, 3 * sizeof(float),
        shape_scales, 1, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &ov_scales), "scales tensor");

    const char *input_names[]  = {"input", "input_lengths", "scales"};
    const char *output_names[] = {"output"};
    OrtValue   *inputs[3]      = {ov_input, ov_lengths, ov_scales};
    OrtValue   *ov_output      = NULL;
    ort_check(g_ort->Run(h->session, NULL,
        input_names, (const OrtValue *const *)inputs, 3,
        output_names, 1, &ov_output), "Run");

    float *out_data = NULL;
    ort_check(g_ort->GetTensorMutableData(ov_output, (void **)&out_data), "GetTensorMutableData");

    OrtTensorTypeAndShapeInfo *shape_info = NULL;
    ort_check(g_ort->GetTensorTypeAndShape(ov_output, &shape_info), "GetTensorTypeAndShape");
    size_t elem_count = 0;
    ort_check(g_ort->GetTensorShapeElementCount(shape_info, &elem_count), "GetElementCount");
    g_ort->ReleaseTensorTypeAndShapeInfo(shape_info);

    IonicArray *result = ionic_array_alloc((int64_t)elem_count);
    for (size_t i = 0; i < elem_count; i++) {
        double d = (double)out_data[i];
        int64_t bits;
        memcpy(&bits, &d, 8);
        ionic_array_push_i64(result, bits);
    }

    fprintf(stderr, "[ionic/ort] piper: %zu samples\n", elem_count);

    g_ort->ReleaseValue(ov_output);
    g_ort->ReleaseValue(ov_scales);
    g_ort->ReleaseValue(ov_lengths);
    g_ort->ReleaseValue(ov_input);
    g_ort->ReleaseMemoryInfo(mem_info);
    return result;
#else
    (void)model; (void)phoneme_arr;
    (void)noise_scale; (void)length_scale; (void)noise_w;
    fprintf(stderr, "[ionic] piper_forward: ONNX Runtime not compiled in\n");
    return ionic_array_alloc(1);
#endif
}

/* ── WAV writer ───────────────────────────────────────────────────────────── */

static void write_le16(FILE *f, uint16_t v) { fputc(v & 0xff, f); fputc(v >> 8, f); }
static void write_le32(FILE *f, uint32_t v) {
    fputc(v & 0xff, f); fputc((v >> 8) & 0xff, f);
    fputc((v >> 16) & 0xff, f); fputc((v >> 24) & 0xff, f);
}

void ionic_write_wav(const char *path, void *samples_arr,
                     int64_t num_samples, int64_t sample_rate) {
    IonicArray *arr = samples_arr;
    int64_t     n   = (num_samples > 0 && num_samples <= arr->len) ? num_samples : arr->len;

    FILE *f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "[ionic] write_wav: cannot open '%s'\n", path); return; }

    uint32_t data_bytes = (uint32_t)(n * 2);
    fwrite("RIFF", 1, 4, f); write_le32(f, 36 + data_bytes);
    fwrite("WAVE", 1, 4, f);
    fwrite("fmt ", 1, 4, f); write_le32(f, 16);
    write_le16(f, 1);                        /* PCM */
    write_le16(f, 1);                        /* mono */
    write_le32(f, (uint32_t)sample_rate);
    write_le32(f, (uint32_t)(sample_rate * 2)); /* byte rate */
    write_le16(f, 2);                        /* block align */
    write_le16(f, 16);                       /* bits per sample */
    fwrite("data", 1, 4, f); write_le32(f, data_bytes);

    int64_t *slots = arr->data;
    for (int64_t i = 0; i < n; i++) {
        double d; memcpy(&d, &slots[i], 8);
        if (d >  1.0) d =  1.0;
        if (d < -1.0) d = -1.0;
        int16_t s = (int16_t)(d * 32767.0);
        write_le16(f, (uint16_t)s);
    }
    fclose(f);
    fprintf(stderr, "[ionic] WAV: %s (%lld samples @ %lld Hz)\n",
            path, (long long)n, (long long)sample_rate);
}

/* ════════════════════════════════════════════════════════════════════════════
 * Web request primitives
 *
 * A minimal dependency-free HTTP/1.1 client over POSIX sockets. Only the
 * blocking HTTP(S-free) path is implemented here; HTTPS would require TLS
 * (a separate dependency) and is stubbed to return an empty body.
 * ════════════════════════════════════════════════════════════════════════════ */

#ifdef _WIN32
#  include <winsock2.h>
#  include <ws2tcpip.h>
#else
#  include <sys/socket.h>
#  include <sys/types.h>
#  include <netinet/in.h>
#  include <arpa/inet.h>
#  include <netdb.h>
#  include <errno.h>
#endif

/*
 * Split a URL into host and port.
 *   "http://host:port/path"  -> host, port, rest (path)
 *   "https://host/..."       -> returns 0 (unsupported)
 */
static void url_parse(const char *url, char *host, int host_sz,
                      int *port, const char **path) {
    const char *h = url;
    *port = 80;
    *path = "/";

    if (strncmp(h, "http://", 7) == 0) { h += 7; }
    else if (strncmp(h, "https://", 8) == 0) {
        // TLS not implemented — mark port as unsupported by using -1
        *port = -1;
        h += 8;
    }

    /* host:port/path */
    const char *colon = strchr(h, ':');
    const char *slash = strchr(h, '/');
    const char *end = colon;
    if (end == NULL || (slash && slash < end)) end = slash;
    if (end == NULL) end = h + strlen(h);

    size_t hlen = (size_t)(end - h);
    if (hlen >= (size_t)host_sz) hlen = (size_t)(host_sz - 1);
    memcpy(host, h, hlen);
    host[hlen] = '\0';

    if (colon && (*colon == ':')) {
        *port = atoi(colon + 1);
        const char *p2 = strchr(colon + 1, '/');
        if (p2) *path = p2; else *path = "/";
    } else if (slash) {
        *path = slash;
    }
}

static void http_write_all(int sock, const char *buf, size_t n) {
    size_t off = 0;
    while (off < n) {
        ssize_t w = send(sock, buf + off, n - off, 0);
        if (w <= 0) return;
        off += (size_t)w;
    }
}

/* Perform a request; returns malloc'd body (or "" on failure). */
static char *http_request(const char *method, const char *url,
                          const char *headers, const char *body) {
    char host[256];
    int  port = 80;
    const char *path = "/";
    url_parse(url, host, sizeof(host), &port, &path);
    if (port < 0) {
        fprintf(stderr, "[ionic/http] https not supported (no TLS): %s\n", url);
        char *e = (char *)malloc(1); e[0] = '\0'; return e;
    }

#ifdef _WIN32
    WSADATA ws; (void)ws;
#else
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) { fprintf(stderr, "[ionic/http] socket: %s\n", strerror(errno)); return NULL; }

    struct hostent *he = gethostbyname(host);
    if (!he) { fprintf(stderr, "[ionic/http] lookup failed: %s\n", host); close(sock); return NULL; }

    struct sockaddr_in sa;
    memset(&sa, 0, sizeof(sa));
    sa.sin_family = AF_INET;
    sa.sin_port   = htons((uint16_t)port);
    memcpy(&sa.sin_addr, he->h_addr_list[0], he->h_length);

    if (connect(sock, (struct sockaddr *)&sa, sizeof(sa)) < 0) {
        fprintf(stderr, "[ionic/http] connect: %s\n", strerror(errno));
        close(sock);
        return NULL;
    }

    /* Build request line + headers */
    char req[4096];
    {
        int n = snprintf(req, sizeof(req), "%s %s HTTP/1.1\r\nHost: %s\r\n",
                         method, path, host);
        if (n < 0) n = 0;
        if (n >= 0) {
            if (headers) {
                size_t hl = strlen(headers);
                if (n + (int)hl < (int)sizeof(req)) {
                    memcpy(req + n, headers, hl);
                    n += (int)hl;
                }
            }
            if (body) {
                char cl[64];
                snprintf(cl, sizeof(cl), "Content-Length: %lld\r\n", (long long)strlen(body));
                size_t cll = strlen(cl);
                if (n + (int)cll < (int)sizeof(req)) {
                    memcpy(req + n, cl, cll);
                    n += (int)cll;
                }
            }
            /* Ask the server to close after the response so we can read to EOF. */
            const char *closehdr = "Connection: close\r\n";
            size_t chl = strlen(closehdr);
            if (n + (int)chl < (int)sizeof(req)) {
                memcpy(req + n, closehdr, chl);
                n += (int)chl;
            }
            if (n + 2 < (int)sizeof(req)) {
                memcpy(req + n, "\r\n", 2); n += 2;
            }
            http_write_all(sock, req, n);
            if (body) http_write_all(sock, body, strlen(body));
        }
    }

    /* Read the full response into a growing buffer.
     * Return the ENTIRE raw response (status line + headers + body) so that
     * ionic_http_status / ionic_http_body can parse it. */
    size_t cap = 8192, len = 0;
    char *buf = (char *)malloc(cap);
    if (!buf) { close(sock); return NULL; }
    char tmp[4096];
    for (;;) {
        ssize_t r = recv(sock, tmp, sizeof(tmp), 0);
        if (r <= 0) break;
        if (len + (size_t)r + 1 > cap) {
            cap *= 2;
            char *nb = (char *)realloc(buf, cap);
            if (!nb) break;
            buf = nb;
        }
        memcpy(buf + len, tmp, (size_t)r);
        len += (size_t)r;
    }
    buf[len] = '\0';
    close(sock);
    return buf;
#endif
}

/* Both wrappers fall back to an *allocated* empty string, not a rodata one:
 * http_request() returns NULL on socket/lookup/connect failure, so this is a
 * reachable path and the caller owns whatever comes back. See §11.1. */
char *ionic_http_get(const char *url) {
    char *b = http_request("GET", url, "", NULL);
    return b ? b : rt_empty_str();
}

char *ionic_http_post(const char *url, const char *body) {
    char *b = http_request("POST", url, "Content-Type: application/json\r\n", body);
    return b ? b : rt_empty_str();
}

/*
 * Return the HTTP status code from a full raw response. Callers that want the
 * code call this on the raw response string (not the body).
 */
int64_t ionic_http_status(const char *raw) {
    if (strncmp(raw, "HTTP/", 5) != 0) return -1;
    const char *sp = strchr(raw, ' ');
    if (!sp) return -1;
    return atoi(sp + 1);
}

/* Split a full HTTP response into headers and body. Returns the body.
 *
 * The no-separator path copies rather than returning `raw`. Returning the input
 * made "the caller owns the result" false: releasing both `raw` and the body
 * would be a double free, and on a literal response it would be a free of
 * rodata. See docs/memory-model.md §11.1. */
char *ionic_http_body(const char *raw) {
    const char *sep = strstr(raw, "\r\n\r\n");
    if (!sep) {
        const char *src = raw ? raw : "";
        size_t n = strlen(src);
        char *c = (char *)malloc(n + 1);
        if (c) memcpy(c, src, n + 1);
        return c;
    }
    char *b = (char *)malloc(strlen(sep + 4) + 1);
    strcpy(b, sep + 4);
    return b;
}

/* URL-encode a string (percent-encoding). */
char *ionic_http_urlencode(const char *s) {
    if (!s) s = "";
    char *out = (char *)malloc(strlen(s) * 3 + 1);
    char *p = out;
    static const char hex[] = "0123456789ABCDEF";
    for (const unsigned char *c = (const unsigned char *)s; *c; c++) {
        unsigned char ch = *c;
        if ((ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') ||
            (ch >= '0' && ch <= '9') || ch == '-' || ch == '_' ||
            ch == '.' || ch == '~') {
            *p++ = (char)ch;
        } else {
            *p++ = '%'; *p++ = hex[(ch >> 4) & 0xF]; *p++ = hex[ch & 0xF];
        }
    }
    *p = '\0';
    return out;
}

/* ═══ Terminal / alternate-screen TUI runtime ═══════════════════════════════
 * Raw-mode terminal control for building full-screen TUIs.
 *
 *   ionic_term_enter()  — save termios, switch to raw mode, enter the alternate
 *                         screen (?1049h), hide the cursor, clear. Installs an
 *                         atexit handler so the terminal is always restored.
 *   ionic_term_exit()   — leave the alternate screen, show cursor, restore raw.
 *   ionic_term_write(s) — write a raw string to the TTY (no stdio buffering).
 *   ionic_term_flush()  — no-op; present so TUI code can be explicit about it.
 *   ionic_term_size()   — cols*1000 + rows (falls back to 80x24).
 *   ionic_term_poll(ms) — 1 if input readable within ms, 0 on timeout, -1 error.
 *   ionic_term_key()    — block for one keypress; codepoint, or a KEY_* sentinel
 *                         (>=1000) for arrows/Home/End/etc; -1 on EOF/error.
 *
 * Input is read from /dev/tty (falling back to stdin) and output written to
 * /dev/tty (falling back to stdout), so redirecting stdin/stdout does not
 * corrupt a TUI session.
 *
 * Design notes:
 *   - The read fd is put in O_NONBLOCK so no read can ever block, and reads are
 *     issued only after poll() reports readable. This is required because a raw
 *     fd can report POLLIN (e.g. after an ESC) yet still block a VMIN=1 read.
 *   - The read-side flags and termios are snapshot once and both fully restored
 *     on exit/atexit, so the host shell is left exactly as it was.
 * ═══════════════════════════════════════════════════════════════════════════ */
#ifndef _WIN32
#  include <termios.h>
#  include <fcntl.h>
#  include <sys/select.h>
#  include <errno.h>
#  include <sys/ioctl.h>

/* KEY_* sentinels (must match lib/tui/term.ionic). */
#define IONIC_KEY_UP    1000
#define IONIC_KEY_DOWN  1001
#define IONIC_KEY_RIGHT 1002
#define IONIC_KEY_LEFT  1003
#define IONIC_KEY_HOME  1004
#define IONIC_KEY_END   1005
#define IONIC_KEY_PGUP  1006
#define IONIC_KEY_PGDN  1007
#define IONIC_KEY_INS   1008
#define IONIC_KEY_DEL   1009
#define IONIC_KEY_ESC   27

static int  g_tui_in_fd        = -1;  /* read side  */
static int  g_tui_out_fd       = -1;  /* write side */
static int  g_tui_raw          = 0;   /* raw mode active */
static int  g_tui_have_saved   = 0;   /* g_tui_saved holds original termios */
static int  g_tui_saved_flags  = 0;   /* original fcntl(O_*) flags of in_fd */
static int  g_tui_have_flags   = 0;
static int  g_tui_hooked       = 0;   /* atexit installed */
static int  g_tui_alt          = 0;   /* alternate screen active */
static struct termios g_tui_saved;

static int tui_dbg(void) { static int d = -1; if (d < 0) d = getenv("IONIC_TUI_DEBUG") ? 1 : 0; return d; }
#define TUIDBG(...) do { if (tui_dbg()) fprintf(stderr, __VA_ARGS__); } while (0)

/* Open the TTY fds once; fall back to stdin/stdout when /dev/tty is absent. */
static void tui_open(void) {
    if (g_tui_in_fd < 0) {
        g_tui_in_fd = open("/dev/tty", O_RDONLY | O_NONBLOCK);
        TUIDBG("tui_open in fd=%d errno=%d\n", g_tui_in_fd, errno);
        if (g_tui_in_fd < 0) g_tui_in_fd = 0;
    }
    if (g_tui_out_fd < 0) {
        g_tui_out_fd = open("/dev/tty", O_WRONLY);
        TUIDBG("tui_open out fd=%d errno=%d\n", g_tui_out_fd, errno);
        if (g_tui_out_fd < 0) g_tui_out_fd = 1;
    }
}

/* Poll g_tui_in_fd for readability, up to timeout_ms. 1 ready, 0 timeout. */
/* NB: select(), not poll(). On macOS poll() reports POLLNVAL on a
 * non-blocking tty fd even when it is perfectly valid and readable. */
static int tui_wait(int timeout_ms) {
    fd_set rf; FD_ZERO(&rf); FD_SET(g_tui_in_fd, &rf);
    struct timeval tv, *ptv = NULL;
    if (timeout_ms >= 0) {
        tv.tv_sec  = timeout_ms / 1000;
        tv.tv_usec = (timeout_ms % 1000) * 1000;
        ptv = &tv;
    }
    int r = select(g_tui_in_fd + 1, &rf, NULL, NULL, ptv);
    int ready = (r > 0 && FD_ISSET(g_tui_in_fd, &rf)) ? 1 : 0;
    TUIDBG("tui_wait fd=%d to=%d r=%d ready=%d\n", g_tui_in_fd, timeout_ms, r, ready);
    return ready;
}

/* Non-blocking single-byte read; -1 when nothing is ready or on EOF/error. */
static int tui_try_byte(void) {
    unsigned char b;
    ssize_t n = read(g_tui_in_fd, &b, 1);
    if (n == 1) return (int)b;
    return -1;
}

/* Block until a byte is available (or timeout_ms < 0 for forever), then read it. */
static int tui_getbyte(int timeout_ms) {
    if (!tui_wait(timeout_ms)) return -1;
    return tui_try_byte();
}

/* Restore the terminal no matter how the program exits. */
static void tui_restore_at_exit(void) {
    if (g_tui_alt) {
        const char *leave = "\033[?25h\033[?1049l";
        if (g_tui_out_fd >= 0) { ssize_t ig = write(g_tui_out_fd, leave, strlen(leave)); (void)ig; }
        g_tui_alt = 0;
    }
    if (g_tui_raw && g_tui_have_saved && g_tui_in_fd >= 0) {
        tcsetattr(g_tui_in_fd, TCSANOW, &g_tui_saved);
        g_tui_raw = 0;
    }
    if (g_tui_have_flags && g_tui_in_fd >= 0) {
        fcntl(g_tui_in_fd, F_SETFL, g_tui_saved_flags);
    }
}

int64_t ionic_term_write(const char *s) {
    tui_open();
    if (!s) return 0;
    size_t len = strlen(s);
    if (g_tui_out_fd < 0) { fputs(s, stdout); fflush(stdout); return (int64_t)len; }
    size_t off = 0;
    while (off < len) {
        ssize_t w = write(g_tui_out_fd, s + off, len - off);
        if (w < 0) { if (errno == EINTR) continue; break; }
        off += (size_t)w;
    }
    return (int64_t)off;
}

int64_t ionic_term_enter(void) {
    tui_open();
    if (!g_tui_have_saved && g_tui_in_fd >= 0) {
        if (tcgetattr(g_tui_in_fd, &g_tui_saved) == 0) g_tui_have_saved = 1;
    }
    if (!g_tui_have_flags && g_tui_in_fd >= 0) {
        int fl = fcntl(g_tui_in_fd, F_GETFL, 0);
        if (fl >= 0) { g_tui_saved_flags = fl; g_tui_have_flags = 1; }
    }
    if (g_tui_have_saved) {
        struct termios raw = g_tui_saved;
        raw.c_lflag &= ~(tcflag_t)(ECHO | ICANON | ISIG | IEXTEN);
        raw.c_iflag &= ~(tcflag_t)(IXON | ICRNL | BRKINT | INPCK | ISTRIP);
        raw.c_oflag &= ~(tcflag_t)(OPOST);
        raw.c_cflag |=  (tcflag_t)(CS8);
        raw.c_cc[VMIN]  = 1;
        raw.c_cc[VTIME] = 0;
        if (tcsetattr(g_tui_in_fd, TCSANOW, &raw) == 0) g_tui_raw = 1;
    }
    if (g_tui_in_fd >= 0) {
        int fl = fcntl(g_tui_in_fd, F_GETFL, 0);
        if (fl >= 0) fcntl(g_tui_in_fd, F_SETFL, fl | O_NONBLOCK);
    }
    if (!g_tui_hooked) { atexit(tui_restore_at_exit); g_tui_hooked = 1; }
    ionic_term_write("\033[?1049h\033[2J\033[H\033[?25l");
    g_tui_alt = 1;
    return 0;
}

int64_t ionic_term_exit(void) {
    ionic_term_write("\033[?25h\033[?1049l");
    g_tui_alt = 0;
    if (g_tui_raw && g_tui_have_saved && g_tui_in_fd >= 0) {
        tcsetattr(g_tui_in_fd, TCSANOW, &g_tui_saved);
        g_tui_raw = 0;
    }
    if (g_tui_have_flags && g_tui_in_fd >= 0) {
        fcntl(g_tui_in_fd, F_SETFL, g_tui_saved_flags);
    }
    return 0;
}

int64_t ionic_term_size(void) {
    tui_open();
    int fd = (g_tui_out_fd >= 0) ? g_tui_out_fd : 1;
    struct winsize ws;
    if (ioctl(fd, TIOCGWINSZ, &ws) == 0 && ws.ws_col > 0 && ws.ws_row > 0) {
        return (int64_t)ws.ws_col * 1000 + (int64_t)ws.ws_row;
    }
    return 80 * 1000 + 24;
}

int64_t ionic_term_flush(void) { return 0; }

int64_t ionic_term_poll(int64_t timeout_ms) {
    tui_open();
    if (g_tui_in_fd < 0) return -1;
    return tui_wait((int)timeout_ms);
}

int64_t ionic_term_key(void) {
    tui_open();
    if (g_tui_in_fd < 0) return -1;

    TUIDBG("term_key: blocking on fd=%d\n", g_tui_in_fd);
    int c = tui_getbyte(-1);      /* block for the first byte */
    TUIDBG("term_key: first byte=%d\n", c);
    if (c < 0) return -1;

    if (c == 0x1b) {  /* ESC: a lone Esc or the start of an escape sequence */
        if (!tui_wait(25)) return IONIC_KEY_ESC;
        int c2 = tui_try_byte();
        if (c2 < 0) return IONIC_KEY_ESC;
        if (c2 == '[' || c2 == 'O') {
            if (!tui_wait(25)) return IONIC_KEY_ESC;
            int c3 = tui_try_byte();
            if (c3 < 0) return IONIC_KEY_ESC;
            if (c3 >= '0' && c3 <= '9') {  /* CSI with numeric param, e.g. [3~ */
                int num = c3 - '0';
                int fin = c3;
                while (tui_wait(25)) {
                    int nc = tui_try_byte();
                    if (nc < 0) break;
                    if (nc >= '0' && nc <= '9') { num = num * 10 + (nc - '0'); continue; }
                    fin = nc; break;
                }
                if (fin == '~') {
                    if (num == 1) return IONIC_KEY_HOME;
                    if (num == 2) return IONIC_KEY_INS;
                    if (num == 3) return IONIC_KEY_DEL;
                    if (num == 4) return IONIC_KEY_END;
                    if (num == 5) return IONIC_KEY_PGUP;
                    if (num == 6) return IONIC_KEY_PGDN;
                }
                return IONIC_KEY_ESC;
            }
            if (c3 == 'A') return IONIC_KEY_UP;
            if (c3 == 'B') return IONIC_KEY_DOWN;
            if (c3 == 'C') return IONIC_KEY_RIGHT;
            if (c3 == 'D') return IONIC_KEY_LEFT;
            if (c3 == 'H') return IONIC_KEY_HOME;
            if (c3 == 'F') return IONIC_KEY_END;
            return IONIC_KEY_ESC;
        }
        return IONIC_KEY_ESC;
    }

    if (c < 0x80) return c;  /* ASCII */

    /* UTF-8: decode a multibyte sequence into a codepoint. */
    int extra = 0; int cp = 0;
    if      ((c & 0xE0) == 0xC0) { cp = c & 0x1F; extra = 1; }
    else if ((c & 0xF0) == 0xE0) { cp = c & 0x0F; extra = 2; }
    else if ((c & 0xF8) == 0xF0) { cp = c & 0x07; extra = 3; }
    else return -1;
    for (int i = 0; i < extra; i++) {
        int nb = tui_getbyte(25);   /* continuation bytes usually arrive at once */
        if (nb < 0) break;
        cp = (cp << 6) | (nb & 0x3F);
    }
    return cp;
}

#else  /* _WIN32 — no raw-terminal support; safe stubs. */
int64_t ionic_term_write(const char *s) { (void)s; return -1; }
int64_t ionic_term_enter(void)  { return -1; }
int64_t ionic_term_exit(void)   { return -1; }
int64_t ionic_term_size(void)   { return 80 * 1000 + 24; }
int64_t ionic_term_flush(void)  { return -1; }
int64_t ionic_term_poll(int64_t timeout_ms) { (void)timeout_ms; return -1; }
int64_t ionic_term_key(void)    { return -1; }
#endif

/* ════════════════════════════════════════════════════════════════════════════
 * Minimal JSON — just enough to read tokenizer.json and model config.json
 *
 * A full-featured parser is unnecessary: the only consumers are the BPE
 * tokenizer (one big object of string->int plus an array of merge rules) and a
 * config lookup by key. This parses the whole document into a small tree and
 * provides typed accessors. It handles \uXXXX escapes because tokenizer.json
 * encodes the SentencePiece word marker U+2581 ("▁") that way in places.
 * ════════════════════════════════════════════════════════════════════════════ */

typedef enum { JNULL, JBOOL, JNUM, JSTR, JARR, JOBJ } JType;

typedef struct JVal {
    JType      type;
    double     num;
    int        boolean;
    char      *str;        /* JSTR payload, or the key in a JOBJ parent */
    struct JVal **items;   /* JARR elements / JOBJ values */
    char     **keys;       /* JOBJ keys (parallel to items) */
    int        n, cap;
} JVal;

static JVal *jnew(JType t) {
    JVal *v = (JVal *)calloc(1, sizeof(JVal));
    v->type = t;
    return v;
}

static void jfree(JVal *v) {
    if (!v) return;
    if (v->str) free(v->str);
    for (int i = 0; i < v->n; i++) {
        if (v->keys) free(v->keys[i]);
        jfree(v->items[i]);
    }
    free(v->items);
    free(v->keys);
    free(v);
}

static void jpush(JVal *v, char *key, JVal *item) {
    if (v->n >= v->cap) {
        v->cap = v->cap ? v->cap * 2 : 8;
        v->items = (JVal **)realloc(v->items, (size_t)v->cap * sizeof(JVal *));
        v->keys  = (char **)realloc(v->keys, (size_t)v->cap * sizeof(char *));
    }
    v->keys[v->n]  = key;
    v->items[v->n] = item;
    v->n++;
}

/* Encode a codepoint as UTF-8 into buf; returns bytes written. */
static int utf8_put(char *buf, unsigned cp) {
    if (cp < 0x80) { buf[0] = (char)cp; return 1; }
    if (cp < 0x800) {
        buf[0] = (char)(0xC0 | (cp >> 6));
        buf[1] = (char)(0x80 | (cp & 0x3F));
        return 2;
    }
    if (cp < 0x10000) {
        buf[0] = (char)(0xE0 | (cp >> 12));
        buf[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
        buf[2] = (char)(0x80 | (cp & 0x3F));
        return 3;
    }
    buf[0] = (char)(0xF0 | (cp >> 18));
    buf[1] = (char)(0x80 | ((cp >> 12) & 0x3F));
    buf[2] = (char)(0x80 | ((cp >> 6) & 0x3F));
    buf[3] = (char)(0x80 | (cp & 0x3F));
    return 4;
}

static JVal *jparse(const char **pp);
static void jws(const char **pp) {
    while (**pp == ' ' || **pp == '\t' || **pp == '\n' || **pp == '\r') (*pp)++;
}

static char *jparse_string(const char **pp) {
    if (**pp != '"') return NULL;
    (*pp)++;
    size_t cap = 16, len = 0;
    char *out = (char *)malloc(cap);
    while (**pp && **pp != '"') {
        unsigned char c = (unsigned char)**pp;
        char tmp[4];
        int tn = 0;
        if (c == '\\') {
            (*pp)++;
            char e = **pp;
            if (e == 'n') { tmp[0] = '\n'; tn = 1; (*pp)++; }
            else if (e == 't') { tmp[0] = '\t'; tn = 1; (*pp)++; }
            else if (e == 'r') { tmp[0] = '\r'; tn = 1; (*pp)++; }
            else if (e == 'b') { tmp[0] = '\b'; tn = 1; (*pp)++; }
            else if (e == 'f') { tmp[0] = '\f'; tn = 1; (*pp)++; }
            else if (e == 'u') {
                (*pp)++;
                unsigned cp = 0;
                for (int k = 0; k < 4 && **pp; k++) {
                    char h = **pp; (*pp)++;
                    cp <<= 4;
                    if (h >= '0' && h <= '9') cp |= (unsigned)(h - '0');
                    else if (h >= 'a' && h <= 'f') cp |= (unsigned)(h - 'a' + 10);
                    else if (h >= 'A' && h <= 'F') cp |= (unsigned)(h - 'A' + 10);
                }
                /* surrogate pair */
                if (cp >= 0xD800 && cp <= 0xDBFF && **pp == '\\') {
                    const char *save = *pp;
                    (*pp)++;
                    if (**pp == 'u') {
                        (*pp)++;
                        unsigned lo = 0;
                        for (int k = 0; k < 4 && **pp; k++) {
                            char h = **pp; (*pp)++;
                            lo <<= 4;
                            if (h >= '0' && h <= '9') lo |= (unsigned)(h - '0');
                            else if (h >= 'a' && h <= 'f') lo |= (unsigned)(h - 'a' + 10);
                            else if (h >= 'A' && h <= 'F') lo |= (unsigned)(h - 'A' + 10);
                        }
                        cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                    } else {
                        *pp = save;
                    }
                }
                tn = utf8_put(tmp, cp);
            }
            else { tmp[0] = e; tn = 1; if (**pp) (*pp)++; }
        } else {
            tmp[0] = (char)c; tn = 1; (*pp)++;
        }
        if (len + (size_t)tn + 1 > cap) {
            while (len + (size_t)tn + 1 > cap) cap *= 2;
            out = (char *)realloc(out, cap);
        }
        memcpy(out + len, tmp, (size_t)tn);
        len += (size_t)tn;
    }
    if (**pp == '"') (*pp)++;
    out[len] = '\0';
    return out;
}

static JVal *jparse(const char **pp) {
    jws(pp);
    char c = **pp;
    if (c == '{') {
        (*pp)++;
        JVal *o = jnew(JOBJ);
        jws(pp);
        if (**pp == '}') { (*pp)++; return o; }
        for (;;) {
            jws(pp);
            char *k = jparse_string(pp);
            jws(pp);
            if (**pp == ':') (*pp)++;
            JVal *v = jparse(pp);
            jpush(o, k, v);
            jws(pp);
            if (**pp == ',') { (*pp)++; continue; }
            if (**pp == '}') { (*pp)++; break; }
            break;
        }
        return o;
    }
    if (c == '[') {
        (*pp)++;
        JVal *a = jnew(JARR);
        jws(pp);
        if (**pp == ']') { (*pp)++; return a; }
        for (;;) {
            JVal *v = jparse(pp);
            jpush(a, NULL, v);
            jws(pp);
            if (**pp == ',') { (*pp)++; continue; }
            if (**pp == ']') { (*pp)++; break; }
            break;
        }
        return a;
    }
    if (c == '"') { JVal *s = jnew(JSTR); s->str = jparse_string(pp); return s; }
    if (!strncmp(*pp, "true", 4))  { *pp += 4; JVal *b = jnew(JBOOL); b->boolean = 1; return b; }
    if (!strncmp(*pp, "false", 5)) { *pp += 5; JVal *b = jnew(JBOOL); b->boolean = 0; return b; }
    if (!strncmp(*pp, "null", 4))  { *pp += 4; return jnew(JNULL); }
    {
        char *end = NULL;
        double d = strtod(*pp, &end);
        if (end == *pp) { (*pp)++; return jnew(JNULL); }
        *pp = end;
        JVal *n = jnew(JNUM);
        n->num = d;
        return n;
    }
}

static JVal *jget(JVal *o, const char *key) {
    if (!o || o->type != JOBJ) return NULL;
    for (int i = 0; i < o->n; i++)
        if (o->keys[i] && strcmp(o->keys[i], key) == 0) return o->items[i];
    return NULL;
}

static JVal *jparse_file(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz <= 0) { fclose(f); return NULL; }
    char *buf = (char *)malloc((size_t)sz + 1);
    if (fread(buf, 1, (size_t)sz, f) != (size_t)sz) { free(buf); fclose(f); return NULL; }
    buf[sz] = '\0';
    fclose(f);
    const char *p = buf;
    JVal *v = jparse(&p);
    free(buf);
    return v;
}

/* ════════════════════════════════════════════════════════════════════════════
 * BPE tokenizer (SentencePiece/HF `tokenizer.json`, byte_fallback)
 *
 * This is the piece that lets Ionic *generate text* rather than only emit
 * token ids. It lives in C because Ionic has no hash map, and the vocab is a
 * 32k string->int table that must be probed per token.
 *
 * Algorithm, matching the HF `tokenizers` reference for this family:
 *   1. normalizer: prepend "▁" (U+2581) and replace every ' ' with "▁"
 *   2. seed the symbol list with *bytes* — but any run that forms a known
 *      vocab entry is merged first, via greedy longest-match over the raw
 *      bytes; this reproduces SentencePiece's pre-tokenisation without
 *      shipping a regex engine.
 *   3. repeatedly merge the adjacent pair with the lowest merge rank,
 *      leftmost first, until no pair remains.
 *   4. any symbol that is not in the vocab becomes <0xNN> byte tokens.
 *   5. post_processor: prepend BOS (<s>, id 1).
 *
 * Decode inverts it: ids -> strings, "▁" -> ' ', <0xNN> -> raw byte.
 * ════════════════════════════════════════════════════════════════════════════ */

#define BPE_HASH_BITS 16
#define BPE_HASH_SIZE (1 << BPE_HASH_BITS)

typedef struct BpeEntry {
    char *key;
    int   id;
    struct BpeEntry *next;
} BpeEntry;

typedef struct {
    BpeEntry *bucket[BPE_HASH_SIZE];
    /* merge ranks: "a b" -> rank. Only the rank matters, not the result. */
    BpeEntry *merges[BPE_HASH_SIZE];

    char **id_to_str;      /* vocab_size strings, owned */
    int    vocab_size;
    int    merge_count;    /* number of merge rules loaded (for the log line) */
    int    bos_id, eos_id, unk_id;
    int    byte_fallback;
    /* byte token ids, or -1 when the token is absent */
    int    byte_ids[256];
    int    loaded;
} BpeTokenizer;

static BpeTokenizer g_bpe;

static unsigned bpe_hash(const char *s) {
    unsigned h = 2166136261u;
    while (*s) { h ^= (unsigned char)*s++; h *= 16777619u; }
    return h & (BPE_HASH_SIZE - 1);
}

static void bpe_bucket_put(BpeEntry **bucket, const char *key, int id) {
    unsigned b = bpe_hash(key);
    BpeEntry *e = (BpeEntry *)malloc(sizeof(BpeEntry));
    e->key  = strdup(key);
    e->id   = id;
    e->next = bucket[b];
    bucket[b] = e;
}

static int bpe_bucket_get(BpeEntry **bucket, const char *key) {
    for (BpeEntry *e = bucket[bpe_hash(key)]; e; e = e->next)
        if (strcmp(e->key, key) == 0) return e->id;
    return -1;
}

static void bpe_free_bucket(BpeEntry **bucket) {
    for (int i = 0; i < BPE_HASH_SIZE; i++) {
        BpeEntry *e = bucket[i];
        while (e) { BpeEntry *n = e->next; free(e->key); free(e); e = n; }
        bucket[i] = NULL;
    }
}

/* Encode a merge rule "a b" as key "a\x01b" so it can share the hash table. */
static void bpe_merge_key(char *dst, size_t cap, const char *a, const char *b) {
    snprintf(dst, cap, "%s\x01%s", a, b);
}

/* Look up a single symbol in the vocab. */
static int bpe_lookup(const char *sym) {
    return bpe_bucket_get(g_bpe.bucket, sym);
}

/* Longest vocab match starting at s, up to maxn bytes. Returns length or 0. */
static int bpe_longest(const char *s, int maxn) {
    /* Cap the probe: the longest tokens in a SentencePiece vocab are short;
       a 32-byte window is generous and keeps this O(32) per position. */
    char buf[64];
    int limit = maxn < 32 ? maxn : 32;
    for (int n = limit; n >= 1; n--) {
        memcpy(buf, s, (size_t)n);
        buf[n] = '\0';
        if (bpe_lookup(buf) >= 0) return n;
    }
    return 0;
}

typedef struct { char *s; } BpeSym;

int64_t ionic_bpe_load(const char *path) {
    bpe_free_bucket(g_bpe.bucket);
    bpe_free_bucket(g_bpe.merges);
    if (g_bpe.id_to_str) {
        for (int i = 0; i < g_bpe.vocab_size; i++) free(g_bpe.id_to_str[i]);
        free(g_bpe.id_to_str);
        g_bpe.id_to_str = NULL;
    }
    memset(&g_bpe, 0, sizeof(g_bpe));

    JVal *root = jparse_file(path);
    if (!root) { fprintf(stderr, "[ionic/bpe] cannot read %s\n", path); return -1; }

    JVal *model = jget(root, "model");
    JVal *vocab = model ? jget(model, "vocab") : NULL;
    JVal *merges = model ? jget(model, "merges") : NULL;
    if (!vocab || vocab->type != JOBJ) {
        fprintf(stderr, "[ionic/bpe] no vocab in %s\n", path);
        jfree(root);
        return -1;
    }

    /* Determine the id space. Ids are dense but not necessarily ordered, so
       size from the maximum id seen. */
    int maxid = -1;
    for (int i = 0; i < vocab->n; i++) {
        int id = (int)vocab->items[i]->num;
        if (id > maxid) maxid = id;
    }
    g_bpe.vocab_size = maxid + 1;
    g_bpe.id_to_str = (char **)calloc((size_t)g_bpe.vocab_size, sizeof(char *));
    for (int i = 0; i < 256; i++) g_bpe.byte_ids[i] = -1;

    for (int i = 0; i < vocab->n; i++) {
        const char *k = vocab->keys[i];
        int id = (int)vocab->items[i]->num;
        bpe_bucket_put(g_bpe.bucket, k, id);
        g_bpe.id_to_str[id] = strdup(k);
        /* record <0xNN> byte tokens */
        if (k[0] == '<' && k[1] == '0' && k[2] == 'x' && k[5] == '>') {
            char hex[3] = { k[3], k[4], 0 };
            int b = (int)strtol(hex, NULL, 16);
            if (b >= 0 && b < 256) g_bpe.byte_ids[b] = id;
        }
    }

    if (merges && merges->type == JARR) {
        for (int i = 0; i < merges->n; i++) {
            JVal *m = merges->items[i];
            if (m->type != JSTR || !m->str) continue;
            char key[512];
            char a[256], b[256];
            const char *sp = strchr(m->str, ' ');
            if (!sp) continue;
            size_t la = (size_t)(sp - m->str);
            if (la >= sizeof(a)) la = sizeof(a) - 1;
            memcpy(a, m->str, la); a[la] = '\0';
            snprintf(b, sizeof(b), "%s", sp + 1);
            bpe_merge_key(key, sizeof(key), a, b);
            bpe_bucket_put(g_bpe.merges, key, i);
        }
        g_bpe.merge_count = merges->n;
    }

    /* special ids */
    g_bpe.bos_id = bpe_lookup("<s>");
    g_bpe.eos_id = bpe_lookup("</s>");
    g_bpe.unk_id = bpe_lookup("<unk>");
    if (g_bpe.bos_id < 0) g_bpe.bos_id = 1;
    if (g_bpe.eos_id < 0) g_bpe.eos_id = 2;
    if (g_bpe.unk_id < 0) g_bpe.unk_id = 0;

    JVal *bf = model ? jget(model, "byte_fallback") : NULL;
    g_bpe.byte_fallback = bf && bf->type == JBOOL ? bf->boolean : 1;

    /* Capture this *before* jfree(): `merges` dangles afterwards, and reading
       it in the log line below is a use-after-free that silently printed
       "(no merges)" for every vocabulary. */
    int n_merges = g_bpe.merge_count;
    g_bpe.loaded = 1;
    jfree(root);
    fprintf(stderr, "[ionic/bpe] loaded %s: %d tokens, %d merges\n",
            path, g_bpe.vocab_size, n_merges);
    return g_bpe.vocab_size;
}

static void bpe_push_str(IonicArray *out, int id) {
    ionic_array_push_i64(out, (int64_t)id);
}

/* Merge syms in place until no adjacent pair has a merge rank. */
static int bpe_merge(BpeSym *syms, int n) {
    char key[600];
    for (;;) {
        int best = -1, best_rank = -1;
        for (int i = 0; i + 1 < n; i++) {
            bpe_merge_key(key, sizeof(key), syms[i].s, syms[i + 1].s);
            int r = bpe_bucket_get(g_bpe.merges, key);
            if (r >= 0 && (best_rank < 0 || r < best_rank)) { best_rank = r; best = i; }
        }
        if (best < 0) break;
        /* merge syms[best] and syms[best+1] */
        size_t la = strlen(syms[best].s), lb = strlen(syms[best + 1].s);
        char *m = (char *)malloc(la + lb + 1);
        memcpy(m, syms[best].s, la);
        memcpy(m + la, syms[best + 1].s, lb);
        m[la + lb] = '\0';
        free(syms[best].s);
        free(syms[best + 1].s);
        syms[best].s = m;
        for (int j = best + 1; j + 1 < n; j++) syms[j] = syms[j + 1];
        n--;
    }
    return n;
}

/* Encode `text` into `out` (an [int64] array). Returns the token count. */
int64_t ionic_bpe_encode(const char *text, void *out_arr, int64_t add_bos) {
    if (!g_bpe.loaded) { fprintf(stderr, "[ionic/bpe] not loaded\n"); return -1; }
    IonicArray *out = (IonicArray *)out_arr;
    out->len = 0;
    if (!text) return 0;
    if (add_bos) bpe_push_str(out, g_bpe.bos_id);

    /* Step 1: normalizer — replace ' ' with the word marker, prefix one. */
    size_t tl = strlen(text);
    size_t ncap = tl * 4 + 8;
    char *norm = (char *)malloc(ncap);
    size_t np = 0;
    /* U+2581 = E2 96 81 */
    norm[np++] = (char)0xE2; norm[np++] = (char)0x96; norm[np++] = (char)0x81;
    for (size_t i = 0; i < tl; i++) {
        if (np + 4 > ncap) { ncap *= 2; norm = (char *)realloc(norm, ncap); }
        if (text[i] == ' ') {
            norm[np++] = (char)0xE2; norm[np++] = (char)0x96; norm[np++] = (char)0x81;
        } else {
            norm[np++] = text[i];
        }
    }
    norm[np] = '\0';

    /* Step 2: greedy longest-match into symbols, then BPE-merge each run. */
    BpeSym *syms = (BpeSym *)malloc(sizeof(BpeSym) * (np + 1));
    int nsym = 0;
    size_t pos = 0;
    while (pos < np) {
        int m = bpe_longest(norm + pos, (int)(np - pos));
        if (m > 0) {
            syms[nsym].s = (char *)malloc((size_t)m + 1);
            memcpy(syms[nsym].s, norm + pos, (size_t)m);
            syms[nsym].s[m] = '\0';
            nsym++;
            pos += (size_t)m;
        } else {
            /* single raw byte; may need byte-fallback below */
            syms[nsym].s = (char *)malloc(2);
            syms[nsym].s[0] = norm[pos];
            syms[nsym].s[1] = '\0';
            nsym++;
            pos++;
        }
    }

    nsym = bpe_merge(syms, nsym);

    for (int i = 0; i < nsym; i++) {
        int id = bpe_lookup(syms[i].s);
        if (id >= 0) {
            bpe_push_str(out, id);
        } else if (g_bpe.byte_fallback) {
            for (const unsigned char *c = (const unsigned char *)syms[i].s; *c; c++) {
                int bid = g_bpe.byte_ids[*c];
                bpe_push_str(out, bid >= 0 ? bid : g_bpe.unk_id);
            }
        } else {
            bpe_push_str(out, g_bpe.unk_id);
        }
        free(syms[i].s);
    }
    free(syms);
    free(norm);
    return out->len;
}

/* Decode `ids` (an [int64] array) into a freshly allocated string. */
char *ionic_bpe_decode(void *ids_arr) {
    IonicArray *a = (IonicArray *)ids_arr;
    size_t cap = (size_t)a->len * 4 + 16, len = 0;
    char *out = (char *)malloc(cap);
    int64_t *slots = (int64_t *)a->data;
    for (int64_t i = 0; i < a->len; i++) {
        int id = (int)slots[i];
        if (id < 0 || id >= g_bpe.vocab_size || !g_bpe.id_to_str[id]) continue;
        const char *tok = g_bpe.id_to_str[id];
        /* <0xNN> -> raw byte */
        if (tok[0] == '<' && tok[1] == '0' && tok[2] == 'x' && tok[5] == '>' && strlen(tok) == 6) {
            char hex[3] = { tok[3], tok[4], 0 };
            unsigned char b = (unsigned char)strtol(hex, NULL, 16);
            if (len + 2 > cap) { cap *= 2; out = (char *)realloc(out, cap); }
            out[len++] = (char)b;
            continue;
        }
        /* skip specials */
        if ((tok[0] == '<' && tok[1] == 's' && tok[2] == '>') ||
            (strcmp(tok, "</s>") == 0) || (strcmp(tok, "<unk>") == 0)) continue;
        for (const char *p = tok; *p; ) {
            if ((unsigned char)p[0] == 0xE2 && p[1] && (unsigned char)p[1] == 0x96 &&
                p[2] && (unsigned char)p[2] == 0x81) {
                if (len + 2 > cap) { cap *= 2; out = (char *)realloc(out, cap); }
                out[len++] = ' ';
                p += 3;
            } else {
                if (len + 2 > cap) { cap *= 2; out = (char *)realloc(out, cap); }
                out[len++] = *p++;
            }
        }
    }
    out[len] = '\0';
    return out;
}

int64_t ionic_bpe_bos(void) { return g_bpe.bos_id; }
int64_t ionic_bpe_eos(void) { return g_bpe.eos_id; }

/* Append one token id to an [int64] array (used by the generation loop). */
int64_t ionic_arr_push_i64(void *arr, int64_t v) {
    ionic_array_push_i64((IonicArray *)arr, v);
    return 0;
}

/* ════════════════════════════════════════════════════════════════════════════
 * Minimal HTTP/1.1 server
 *
 * The runtime only ever had a *client*. Serving needs bind/listen/accept plus
 * request parsing, so it lives here next to the client for symmetry. Scope is
 * deliberately one request at a time: this exists to put a model behind a
 * socket for a proof of concept, not to be a production web server.
 *
 *   ionic_http_serve(port)     -> listener fd (bound + listening), or -1
 *   ionic_http_accept(fd)      -> *request* handle, or -1
 *   ionic_http_method(req)     -> "GET" / "POST" / ...
 *   ionic_http_path(req)       -> "/v1/models"   (query string stripped)
 *   ionic_http_req_body(req)   -> request body, "" when there is none
 *   ionic_http_respond(req, status, ctype, body) -> bytes written, or -1
 *   ionic_http_req_done(req)   -> close the connection and free the request
 *   ionic_http_close(fd)       -> close the *listener*
 *
 * One handle serves the whole exchange, so a caller can never mix up a socket
 * fd with a request pointer: `accept` returns the request, and `respond` /
 * `req_done` take that same request back. The connection fd lives inside it.
 * ════════════════════════════════════════════════════════════════════════════ */

typedef struct {
    int   conn;         /* accepted connection, owned by this request */
    char  method[16];
    char  path[1024];
    char *body;
    int   body_len;
    char *raw;          /* headers + body, kept alive for debugging */
} HttpRequest;

static void http_req_free(HttpRequest *r) {
    if (!r) return;
    if (r->conn >= 0) { close(r->conn); r->conn = -1; }
    free(r->body);
    free(r->raw);
    free(r);
}

/* Read headers, then exactly Content-Length bytes. Returns NULL on error. */
static HttpRequest *http_read_request(int conn) {
    size_t cap = 8192, len = 0;
    char *buf = (char *)malloc(cap);
    char tmp[4096];
    long hdr_end = -1;

    /* Phase 1: headers. This function takes ownership of `conn`: every failure
     * path below closes it, so callers must not close it a second time. */
    while (hdr_end < 0) {
        ssize_t n = recv(conn, tmp, sizeof(tmp), 0);
        if (n <= 0) { free(buf); close(conn); return NULL; }
        if (len + (size_t)n + 1 > cap) {
            while (len + (size_t)n + 1 > cap) cap *= 2;
            buf = (char *)realloc(buf, cap);
        }
        memcpy(buf + len, tmp, (size_t)n);
        len += (size_t)n;
        buf[len] = '\0';
        char *p = strstr(buf, "\r\n\r\n");
        if (p) hdr_end = (long)(p - buf) + 4;
    }

    HttpRequest *r = (HttpRequest *)calloc(1, sizeof(HttpRequest));
    r->conn = conn;

    /* request line: METHOD SP PATH SP VERSION */
    {
        const char *sp1 = strchr(buf, ' ');
        if (!sp1) { free(buf); http_req_free(r); return NULL; }
        size_t ml = (size_t)(sp1 - buf);
        if (ml >= sizeof(r->method)) ml = sizeof(r->method) - 1;
        memcpy(r->method, buf, ml); r->method[ml] = '\0';
        const char *sp2 = strchr(sp1 + 1, ' ');
        const char *pe = sp2 ? sp2 : sp1 + 1 + strlen(sp1 + 1);
        size_t pl = (size_t)(pe - (sp1 + 1));
        if (pl >= sizeof(r->path)) pl = sizeof(r->path) - 1;
        memcpy(r->path, sp1 + 1, pl); r->path[pl] = '\0';
        /* strip query string */
        char *q = strchr(r->path, '?');
        if (q) *q = '\0';
    }

    /* Content-Length (case-insensitive) */
    long content_len = 0;
    {
        const char *cl = NULL;
        for (char *p = buf; p && *p; p++) {
            if (!strncasecmp(p, "content-length:", 15)) { cl = p + 15; break; }
        }
        if (cl) content_len = strtol(cl, NULL, 10);
    }

    size_t have_body = len - (size_t)hdr_end;
    while ((long)have_body < content_len) {
        ssize_t n = recv(conn, tmp, sizeof(tmp), 0);
        if (n <= 0) break;
        if (len + (size_t)n + 1 > cap) {
            while (len + (size_t)n + 1 > cap) cap *= 2;
            buf = (char *)realloc(buf, cap);
        }
        memcpy(buf + len, tmp, (size_t)n);
        len += (size_t)n;
        buf[len] = '\0';
        have_body += (size_t)n;
    }

    r->raw = buf;
    r->body_len = (int)content_len;
    r->body = (char *)malloc((size_t)content_len + 1);
    if (content_len > 0 && len >= (size_t)hdr_end) {
        size_t avail = len - (size_t)hdr_end;
        size_t take = avail < (size_t)content_len ? avail : (size_t)content_len;
        memcpy(r->body, buf + hdr_end, take);
        r->body[take] = '\0';
        r->body_len = (int)take;
    } else {
        r->body[0] = '\0';
        r->body_len = 0;
    }
    return r;
}

int64_t ionic_http_serve(int64_t port) {
#ifndef _WIN32
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) { fprintf(stderr, "[ionic/http] socket: %s\n", strerror(errno)); return -1; }
    int one = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    struct sockaddr_in sa;
    memset(&sa, 0, sizeof(sa));
    sa.sin_family      = AF_INET;
    sa.sin_addr.s_addr = htonl(INADDR_ANY);
    sa.sin_port        = htons((uint16_t)port);
    if (bind(fd, (struct sockaddr *)&sa, sizeof(sa)) < 0) {
        fprintf(stderr, "[ionic/http] bind %lld: %s\n", (long long)port, strerror(errno));
        close(fd); return -1;
    }
    if (listen(fd, 16) < 0) {
        fprintf(stderr, "[ionic/http] listen: %s\n", strerror(errno));
        close(fd); return -1;
    }
    return fd;
#else
    (void)port;
    fprintf(stderr, "[ionic/http] server not implemented on Windows\n");
    return -1;
#endif
}

/* Returns 0 rather than void: every other builtin in this runtime returns
 * int64 so generated code has a uniform "call -> i64" shape. A void return
 * would force the two compilers to agree on a special case for one call. */
int64_t ionic_http_close(int64_t fd) {
#ifndef _WIN32
    if (fd >= 0) close((int)fd);
#endif
    return 0;
}

/* Accept one connection and read its request. The request object is passed to
 * Ionic as an opaque int64; the accessors below read through it. */
int64_t ionic_http_accept(int64_t listen_fd) {
#ifndef _WIN32
    int conn = accept((int)listen_fd, NULL, NULL);
    if (conn < 0) { fprintf(stderr, "[ionic/http] accept: %s\n", strerror(errno)); return -1; }
    /* http_read_request owns `conn` from here on: on failure it has already
     * closed it, so do not close again here (that would close a recycled fd). */
    HttpRequest *r = http_read_request(conn);
    if (!r) return -1;
    return (int64_t)(intptr_t)r;
#else
    (void)listen_fd;
    return -1;
#endif
}

/* These three return into the request object: `method` and `path` are inline
 * char arrays, `body` a separately malloc'd buffer freed by
 * ionic_http_req_done(). Handing any of them back directly would break the
 * "a returned TY_STR is the caller's" contract in two ways — releasing the
 * result would free the middle of the HttpRequest, and the value would dangle
 * the moment the caller closed the request. Both compilers already classify
 * these as owned strings, so the copy makes their existing model true rather
 * than changing it. See docs/memory-model.md §11.2. */
char *ionic_http_method(int64_t req) {
    HttpRequest *r = (HttpRequest *)(intptr_t)req;
    return rt_dup_str(r ? r->method : NULL);
}

char *ionic_http_path(int64_t req) {
    HttpRequest *r = (HttpRequest *)(intptr_t)req;
    return rt_dup_str(r ? r->path : NULL);
}

char *ionic_http_req_body(int64_t req) {
    HttpRequest *r = (HttpRequest *)(intptr_t)req;
    return rt_dup_str(r ? r->body : NULL);
}

int64_t ionic_http_req_done(int64_t req) {
    HttpRequest *r = (HttpRequest *)(intptr_t)req;
    http_req_free(r);
    return 0;
}

/* Write a complete HTTP response on the request's connection. Every response
 * says Connection: close and `ionic_http_req_done` closes it, which keeps the
 * single-threaded loop simple and interoperable with curl. */
int64_t ionic_http_respond(int64_t req, int64_t status, const char *ctype, const char *body) {
#ifndef _WIN32
    HttpRequest *r = (HttpRequest *)(intptr_t)req;
    if (!r || r->conn < 0) return -1;
    int conn = r->conn;
    const char *reason = "OK";
    if (status == 201) reason = "Created";
    else if (status == 204) reason = "No Content";
    else if (status == 400) reason = "Bad Request";
    else if (status == 404) reason = "Not Found";
    else if (status == 405) reason = "Method Not Allowed";
    else if (status == 500) reason = "Internal Server Error";

    size_t blen = body ? strlen(body) : 0;
    char head[512];
    int hn = snprintf(head, sizeof(head),
        "HTTP/1.1 %lld %s\r\n"
        "Content-Type: %s\r\n"
        "Content-Length: %zu\r\n"
        "Connection: close\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "\r\n",
        (long long)status, reason, ctype ? ctype : "application/json", blen);
    if (hn < 0) return -1;

    size_t off = 0;
    while (off < (size_t)hn) {
        ssize_t w = send((int)conn, head + off, (size_t)hn - off, 0);
        if (w <= 0) return -1;
        off += (size_t)w;
    }
    size_t boff = 0;
    while (boff < blen) {
        ssize_t w = send((int)conn, body + boff, blen - boff, 0);
        if (w <= 0) return -1;
        boff += (size_t)w;
    }
    return (int64_t)(hn + (int)blen);
#else
    (void)req; (void)status; (void)ctype; (void)body;
    return -1;
#endif
}
