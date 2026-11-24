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

/* `ionic_array_alloc` and `ionic_array_push_i64` are defined below the
 * reference-counting primitives, because the header is a counted block and
 * `ionic_rc_alloc` has to exist first. */

/* ── Reference counting: primitives (memory-model §8 step 4, stage 1) ──────────
 *
 * A counted object carries an 8-byte count immediately *before* its payload, and
 * the payload pointer is what the program holds. That layout is the whole point:
 * a string stays one `i64` at the ABI, so no call site, no struct, and no array
 * slot changes shape. It costs 8 bytes per object and one non-atomic increment
 * per reference copy.
 *
 *     base: [ rc | payload ................ ]
 *            ^     ^
 *            |     `-- what Ionic sees
 *            `-------- what free() takes
 *
 * Release on a **pinned** object is a no-op. Pinned is how string literals and
 * foreign pointers participate uniformly instead of needing an exemption at
 * every use site: a literal's managed copy is pinned, so releasing it is safe
 * and does nothing (see ionic_str_lit).
 *
 * Nothing in this file *calls* these yet. Stage 1 is deliberately behaviour-free:
 * the producers allocate through rc_alloc, but no retain or release is emitted,
 * so the generated code is unchanged and `--verify-self-hosting` still produces
 * a byte-identical object. Emission is stage 2, and it is where the risk lives —
 * a misplaced release turns a benign leak into a use-after-free.
 *
 * Alignment: payload sits at base+8, so it is 8-byte aligned (malloc gives 16).
 * Fine for strings and pointers; not for anything needing 16-byte alignment.
 */
#define IONIC_RC_PINNED (-1)

/* Allocate `n` bytes of payload with a count of 1. Returns the payload, or NULL
 * on OOM (the one case where the "caller owns the result" contract cannot be
 * honoured, as elsewhere in this runtime). */
static void *ionic_rc_alloc(size_t n) {
    size_t total = n + 8;
    char  *base  = (char *)malloc(total);
    if (!base) return NULL;
    *((int64_t *)base) = 1;
    return base + 8;
}

/* Allocate a payload that is never freed. Used for the managed copies of string
 * literals: the copy is made once and outlives every reference to it. */
static void *ionic_rc_alloc_pinned(size_t n) {
    size_t total = n + 8;
    char  *base  = (char *)malloc(total);
    if (!base) return NULL;
    *((int64_t *)base) = IONIC_RC_PINNED;
    return base + 8;
}

/* Add a reference. Null-tolerant, and a no-op on a pinned object: the count of
 * a pinned object is not tracked because it can never reach zero. */
static void *ionic_rc_retain(void *p) {
    if (!p) return NULL;
    int64_t *rc = ((int64_t *)p) - 1;
    if (*rc != IONIC_RC_PINNED) { *rc += 1; }
    return p;
}

/* Drop a reference, freeing the block when the last one goes. Null-tolerant.
 *
 * Release is **not** idempotent and this cannot make it so: once the count hits
 * zero the block is freed, so a second release necessarily reads reclaimed
 * memory — the read is the use-after-free, before any guard could look at the
 * count. (Measured: an ASan probe of `release(p); release(p);` reports a
 * heap-use-after-free on the header load, not a caught error.) The discipline
 * that keeps this correct is the same one model RAII uses: exactly one release
 * per owning slot, from a single emit point. See docs/memory-model.md §8 step 4.
 *
 * The `<= 0` check is therefore defence against a *corrupt* header that already
 * reads zero, not against double release. It costs one compare and stops such a
 * block from being decremented to -1, which would look like PINNED and leak. */
static void ionic_rc_release(void *p) {
    if (!p) return;
    int64_t *rc = ((int64_t *)p) - 1;
    if (*rc == IONIC_RC_PINNED) return;
    if (*rc <= 0) return;
    *rc -= 1;
    if (*rc == 0) free(rc);
}

/* Grow a counted payload. The header moves with it, so this is the only way a
 * rc_alloc'd block may be resized — a plain realloc(payload) would free the
 * header and return a pointer with no count in front of it. */
static void *ionic_rc_realloc(void *p, size_t n) {
    if (!p) return ionic_rc_alloc(n);
    int64_t *rc = ((int64_t *)p) - 1;
    if (*rc == IONIC_RC_PINNED) {
        /* A pinned block cannot be grown in place, so copy it into a live one.
         *
         * The old length is knowable even though the header does not store it:
         * every pinned block comes from ionic_str_lit, which allocates exactly
         * strlen(s)+1. Copying that much (clamped to the new size, for a shrink)
         * is therefore safe, whereas memcpy(q, p, n) would read past the end
         * whenever the block grows. The interned original is left untouched, so
         * the literal table keeps pointing at a valid value. */
        size_t have = strlen((const char *)p) + 1;
        void *q = ionic_rc_alloc(n);
        if (!q) return NULL;
        memcpy(q, p, have < n ? have : n);
        return q;
    }
    char *base = (char *)realloc(rc, n + 8);
    if (!base) return NULL;
    return base + 8;
}

/* ── Dynamic arrays as counted objects (memory-model §11.4) ───────────────────
 *
 * `IonicArray` is a *two-allocation* object: the header `{len, cap, data}` and
 * the element buffer `data`. A count in front of the header covers only the
 * header, so releasing with `ionic_rc_release` would free the header and leak
 * the buffer. `ionic_array_release` is the release an array needs, and it is
 * the only correct one — mixing the two would free `data` on the wrong
 * reference or not at all.
 *
 * What release deliberately does **not** do is release the *elements*.
 * Elements are stored as int64 bit patterns with no per-slot type, so the
 * runtime cannot tell `[int64]` (whose elements are raw integers, and where
 * "releasing" one would mean freeing an address built from `1`, `2`, `3`) from
 * `[string]` (whose elements may be counted blocks). Releasing elements would
 * therefore corrupt every numeric array.
 *
 * Even where the element type *is* known at the release site, releasing is not
 * generally sound: a `[string]` element may be a `getenv`/`get_arg` borrow or a
 * function parameter, which an escape rule dropped the claim over precisely
 * because it is not ours to free. Strings stored in a `[string]` array are
 * therefore **leaked, not reclaimed** — the same accepted, documented direction
 * `docs/memory-model.md` §4.2 takes for any container escape. `[int64]`,
 * `[float64]` and `[T]`-of-struct arrays are fully reclaimed.
 */

static IonicArray *ionic_array_alloc(int64_t cap) {
    IonicArray *a = (IonicArray *)ionic_rc_alloc(sizeof(IonicArray));
    if (!a) return NULL;
    a->len  = 0;
    a->cap  = cap > 0 ? cap : 8;
    a->data = malloc((size_t)(a->cap * 8));
    if (!a->data) { ionic_rc_release(a); return NULL; }
    return a;
}

static void ionic_array_push_i64(IonicArray *a, int64_t v) {
    if (a->len >= a->cap) {
        a->cap *= 2;
        a->data = realloc(a->data, (size_t)(a->cap * 8));
    }
    ((int64_t *)a->data)[a->len++] = v;
}

/* Drop one reference to the array. The last reference frees the element buffer
 * as well as the header, because `data` is owned by the header and by nothing
 * else. Null-tolerant; a no-op on a pinned array, which nothing creates today
 * but which keeps the contract uniform with strings. */
static void ionic_array_release(IonicArray *a) {
    if (!a) return;
    int64_t *rc = ((int64_t *)a) - 1;
    if (*rc == IONIC_RC_PINNED) return;
    if (*rc <= 0) return;
    *rc -= 1;
    if (*rc == 0) {
        free(a->data);
        free(rc);
    }
}

/* ── Exported surface for emitted code (memory-model §8 step 4, stage 2) ──────
 *
 * The primitives above are `static` because nothing in this file's own code
 * should bypass them. Emitted code cannot call a static symbol, though, so
 * stage 2 needs a public entry point for each operation the codegen emits.
 *
 * These are deliberately thin wrappers, not re-exports: keeping the `static`
 * originals means the compiler cannot accidentally resolve a call to the shim
 * instead of the real thing, and it keeps `scripts/rcprobe.c` (which includes
 * this TU directly) able to test the actual implementation.
 *
 * `ionic_rc_retain` returns its argument so a retain can be used as a value
 * expression (`x = retain(y)`) rather than only as a statement.
 */

void *ionic_rc_retain_pub(void *p) { return ionic_rc_retain(p); }

/* Release is a statement in the emitted code: the value is dead afterwards, so
 * returning anything would invite a caller to use it. Stage 2 emits a `bl` and
 * ignores X0. */
void ionic_rc_release_pub(void *p) { ionic_rc_release(p); }

/* The array release, for emitted code. Separate from `ionic_rc_release_pub`
 * because an array owns its element buffer as well as its header: releasing an
 * `IonicArray *` with the plain one would free the header and leak `data`
 * (§11.4). Codegen must pick this symbol for an array-typed slot.
 *
 * Elements are not released — see the note on `ionic_array_release`. */
void ionic_array_release_pub(void *p) { ionic_array_release((IonicArray *)p); }

/* The array retain, for emitted code. A copy of an array value — `let b = a` —
 * is a second name for one header, so it must raise the count. Without it the
 * copy has no claim of its own while the original still has one, and the two
 * names can disagree about who owns the header: `let b = a; return b;` would
 * release `a` at the exit block and hand the caller a pointer into freed
 * memory. Retaining makes the count match the number of live names, which is
 * the whole invariant.
 *
 * Retain and release are deliberately asymmetric in *which* allocations they
 * touch: retain raises the count on the header only, and release frees the
 * element buffer when the count reaches zero. That is correct because `data`
 * is owned by the header and by nothing else — it is reached through the
 * header, never stored independently.
 *
 * Returns its argument so a retain can be used as a value expression
 * (`b = retain(a)`) rather than only as a statement. */
void *ionic_array_retain_pub(void *p) { return ionic_rc_retain(p); }

/* Deliberately absent: an element-release helper for `ionic_array_get`.
 * Releasing an element would be correct only for a `[string]` array, and the
 * element slots carry no type — see the note on `ionic_array_release`. A
 * release helper that could only be correct half the time is worse than no
 * helper at all. */

/* ── String literals as managed values ───────────────────────────────────────
 *
 * A literal lives in `__cstring` (read-only, no count in front of it), so it
 * cannot be released. Rather than teach every release site to recognise one,
 * literals are handed out as *pinned managed copies*, interned by their rodata
 * address: the first request for a given literal allocates one copy, and every
 * later request returns that same block. Releasing it does nothing.
 *
 * Keyed by address rather than by content because codegen already interns each
 * literal node to a fixed `__cstring` offset, so the address is a stable
 * identity for it and costs nothing to compare.
 *
 * Open-addressed table, growth by doubling. Literal counts are bounded by the
 * program (the compiler itself has a few thousand), so this is a small, flat
 * structure rather than a general-purpose map.
 */
static const char **g_lit_src  = NULL;   /* rodata address, or NULL = empty slot */
static char       **g_lit_copy = NULL;
static int64_t      g_lit_cap  = 0;
static int64_t      g_lit_used = 0;

static uint64_t lit_hash(const void *p) {
    uint64_t v = (uint64_t)(uintptr_t)p;
    return v * 11400714819323198485ull;   /* fibonacci mix; addresses are aligned */
}

static int lit_grow(void) {
    int64_t newcap = g_lit_cap ? g_lit_cap * 2 : 256;
    const char **ns = (const char **)calloc((size_t)newcap, sizeof(char *));
    char       **nc = (char **)calloc((size_t)newcap, sizeof(char *));
    if (!ns || !nc) { free((void *)ns); free((void *)nc); return -1; }
    /* Reinsert every live entry. The masks differ, so positions all move. */
    for (int64_t i = 0; i < g_lit_cap; i++) {
        if (!g_lit_src[i]) continue;
        uint64_t h = lit_hash(g_lit_src[i]) & (uint64_t)(newcap - 1);
        while (ns[h]) { h = (h + 1) & (uint64_t)(newcap - 1); }
        ns[h] = g_lit_src[i];
        nc[h] = g_lit_copy[i];
    }
    free((void *)g_lit_src);
    free((void *)g_lit_copy);
    g_lit_src  = ns;
    g_lit_copy = nc;
    g_lit_cap  = newcap;
    return 0;
}

/* The managed, pinned value for a string literal. `s` must be a rodata address
 * with static lifetime — that is what codegen passes, and why the address is a
 * usable key.
 *
 * On OOM it returns `s` unchanged. That is the *less* dangerous of two bad
 * options, not a correct one: the caller gets a raw rodata pointer with no
 * count in front of it, so a later release reads the byte before the literal
 * (in-bounds rodata, but garbage) and may try to free a static address. The
 * alternative — returning NULL — turns an out-of-memory condition into a
 * null-deref at the first use, and a string is more often printed than released.
 * Either way this is an OOM-only compromise, recorded so it is not mistaken for
 * a guarantee. Same class as the OOM note in docs/memory-model.md §11.1. */
const char *ionic_str_lit(const char *s) {
    if (!s) return s;
    /* Keep the table at least half empty so linear probing terminates quickly. */
    if (g_lit_used * 2 >= g_lit_cap && lit_grow() != 0) return s;

    uint64_t mask = (uint64_t)(g_lit_cap - 1);
    uint64_t h    = lit_hash(s) & mask;
    while (g_lit_src[h]) {
        if (g_lit_src[h] == s) { return g_lit_copy[h]; }
        h = (h + 1) & mask;
    }
    size_t n   = strlen(s) + 1;
    char  *cp  = (char *)ionic_rc_alloc_pinned(n);
    if (!cp) return s;
    memcpy(cp, s, n);
    g_lit_src[h]  = s;
    g_lit_copy[h] = cp;
    g_lit_used   += 1;
    return cp;
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

/* Physical host memory in bytes, or 0 when it cannot be determined.
 *
 * This is the capacity of the SRAM pool, and it is a fact about the machine
 * rather than about any backend, so it lives outside the MLX conditional: a
 * program can be told what its host pool really holds on every platform. */
static int64_t host_ram_bytes(void) {
#if defined(_WIN32)
    MEMORYSTATUSEX ms;
    ms.dwLength = sizeof(ms);
    if (!GlobalMemoryStatusEx(&ms)) return 0;
    return (int64_t)ms.ullTotalPhys;
#else
    long pages = sysconf(_SC_PHYS_PAGES);
    long psize = sysconf(_SC_PAGESIZE);
    if (pages <= 0 || psize <= 0) return 0;
    return (int64_t)pages * (int64_t)psize;
#endif
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

/* Reset an array to empty without freeing it. Drops every element slot by
 * setting `len` to 0; the slot *storage* stays allocated for reuse, which is
 * the point of the call.
 *
 * The dropped elements are **not** released. This is the same decision
 * `ionic_array_release` makes and for the same reason: element slots carry no
 * type, so the runtime cannot tell a `[int64]` element (a plain number, where
 * "releasing" means freeing an address built from it) from a `[string]`
 * element. A `[string]` reset therefore leaks its element blocks, which is the
 * documented safe direction rather than a use-after-free. See §11.4. */
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
    char *e = (char *)ionic_rc_alloc(1);
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
    char *d = (char *)ionic_rc_alloc(n);
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
    char *buf = (char *)ionic_rc_alloc((size_t)(sz + 1));
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

/* Does a regular file exist at `path`? 1 or 0.
 *
 * Defined *in terms of* ionic_file_size rather than carrying its own stat(2),
 * so the two builtins cannot drift: `file_exists(p)` and
 * `file_size(p) >= 0` agree by construction, and the empty-file and
 * directory cases are answered once. It is a separate name only because that
 * is what a reader of `if (file_exists(p))` expects to be able to write; the
 * call is the same stat. */
int64_t ionic_file_exists(const char *path) {
    return ionic_file_size(path) >= 0 ? 1 : 0;
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
    if (!fgets(buf, sizeof(buf), stdin)) { char *r = (char *)ionic_rc_alloc(1); r[0] = '\0'; return r; }
    size_t n = strlen(buf);
    if (n > 0 && buf[n-1] == '\n') buf[n-1] = '\0';
    char *r = (char *)ionic_rc_alloc(strlen(buf) + 1);
    strcpy(r, buf);
    return r;
}

char *ionic_str_slice(const char *s, int64_t start, int64_t end) {
    if (!s) { char *r = (char *)ionic_rc_alloc(1); r[0] = '\0'; return r; }
    int64_t n = (int64_t)strlen(s);
    if (start < 0) start = 0;
    if (end > n) end = n;
    if (start >= end) { char *r = (char *)ionic_rc_alloc(1); r[0] = '\0'; return r; }
    int64_t len = end - start;
    char *r = (char *)ionic_rc_alloc((size_t)(len + 1));
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
        char *r = (char *)ionic_rc_alloc(n + 1);
        memcpy(r, src, n + 1);
        return r;
    }
    size_t ls = strlen(s), lf = strlen(from), lt = strlen(to);
    if (lf == 0) { char *r = (char *)ionic_rc_alloc(ls + 1); memcpy(r, s, ls + 1); return r; }
    /* Count occurrences */
    int count = 0;
    const char *p = s;
    while ((p = strstr(p, from))) { count++; p += lf; }
    size_t need = ls + (size_t)count * (lt > lf ? lt - lf : 0) + 1;
    char *r = (char *)ionic_rc_alloc(need);
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
    char *r = (char *)ionic_rc_alloc(la + lb + 1);
    memcpy(r, a, la); memcpy(r + la, b, lb); r[la + lb] = '\0';
    return r;
}

/* ── Bulk string building ──────────────────────────────────────────────────────
 *
 * `str_concat` allocates a fresh buffer for every call and nothing frees the
 * old one, so building a result by repeated concatenation costs
 *     sum(k = 1..n) k*avg  ~=  n^2*avg/2
 * bytes of allocation and the same again in leaked memory. That is invisible at
 * n = 10 and fatal at n = 10,000: the compiler's own import expander joined a
 * ~170 KB source one line at a time and peaked at 269 MB doing it.
 *
 * `ionic_str_join` exists so the *bulk* case is linear: two passes and one
 * allocation, so the peak is the result rather than the churn.
 *
 * It takes an IonicArray of strings: the language has no string-builder type,
 * and every alternative (a byte array of [int64], a linked list of fragments)
 * either re-wraps the whole payload per append or re-walks the list per copy.
 * An [string] holds the fragments as pointers, so a pass over it to compute
 * length is a pass over n words — the payload is never recopied until the single
 * final fill.
 *
 * Callers still have to *collect* the fragments, which is why the two users in
 * this tree push into an array instead of accumulating a string. The array is
 * the builder; this is the one place the pieces get copied.
 */

/* Concatenate the run `arr[first .. first+count)`, putting `sep` *after* every
   element — including the last.
 *
 * Terminator, not separator, because that is what both callers actually want:
 * the import expander assembles line fragments and the driver assembles file
 * fragments, and each piece is followed by a newline. `parts[0] + "\n" +
 * parts[1] + "\n"` and `parts[0] + "\n" + parts[1]` differ by one byte, and a
 * primitive that silently drops the trailing one would be the kind of thing
 * that gets "fixed" by appending an empty element at every call site.
 * std.text's `join` remains the separator-between version.
 *
 * `first` is an index into the array (0-based, like `arr[i]`), not an offset
 * from 1: callers in this tree keep a sentinel at slot 0 and join from 1.
 *
 * Two passes and one allocation. Pass 1 reads one word per element; the
 * payload is touched once, during the fill. */
char *ionic_str_join(IonicArray *arr, int64_t first, int64_t count, const char *sep) {
    if (!sep) sep = "";
    if (!arr || count <= 0) { char *r = (char *)ionic_rc_alloc(1); r[0] = '\0'; return r; }
    if (first < 0) first = 0;
    if (first >= arr->len) { char *r = (char *)ionic_rc_alloc(1); r[0] = '\0'; return r; }
    /* Clamp rather than trust the caller: a count past the end would read
       whatever follows the data block, and this is reachable from ordinary
       Ionic, where `len` and the count are two separate values. */
    if (count > arr->len - first) { count = arr->len - first; }

    size_t sl  = strlen(sep);
    size_t put = sl * (size_t)count;      /* one sep per element, always */

    int64_t i = first;
    while (i < first + count) {
        const char *s = (const char *)((int64_t *)arr->data)[i];
        if (s) put += strlen(s);
        i = i + 1;
    }

    char *r = (char *)ionic_rc_alloc(put + 1);
    size_t at = 0;
    i = first;
    while (i < first + count) {
        const char *s = (const char *)((int64_t *)arr->data)[i];
        if (s) { size_t l = strlen(s); memcpy(r + at, s, l); at += l; }
        memcpy(r + at, sep, sl); at += sl;
        i = i + 1;
    }
    r[at] = '\0';
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
    char *buf = (char *)ionic_rc_alloc(32);
    snprintf(buf, 32, "%lld", (long long)v);
    return buf;
}

__attribute__((weak))
char *ionic_float64_to_str(int64_t b) {
    double v = i642d(b);
    char *buf = (char *)ionic_rc_alloc(32);
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
    char *out = (char *)ionic_rc_alloc(out_sz + 1);
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
    char *buf = (char *)ionic_rc_alloc(2);
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


/* ── Backend implementations ─────────────────────────────────────────────────
 * The two backends are #included rather than compiled as their own translation
 * units, and this is load-bearing rather than a packaging preference: the code
 * *below* this point (the public model API and Piper TTS) calls their `static`
 * helpers by name — `ort_load`, `g_ort`, `ort_check` — so those must be visible
 * at the point of use. Giving each backend its own .c file would require those
 * three to lose `static`, widening the exported symbol surface for no gain.
 *
 * The two .inc files do not reference each other; the order below is for
 * reading (model loading before tensor math), not for correctness.
 *
 * Each is self-guarded (`#ifdef IONIC_HAVE_ORT`, `#if IONIC_MLX_ACTIVE`), so it
 * compiles to nothing when its backend is absent. Note the asymmetry those
 * guards create: `IONIC_HAVE_ORT` arrives on the command line, whereas
 * `IONIC_MLX_ACTIVE` is derived near the top of this file from IONIC_HAVE_MLX
 * and __APPLE__. The includes therefore have to sit *below* that derivation,
 * which is why they are here rather than in the #include block at the top.
 * ------------------------------------------------------------------------- */
#include "ionic_runtime_ort.inc"
#include "ionic_runtime_mlx.inc"

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
        return rt_empty_str();
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
    char *buf = (char *)ionic_rc_alloc(cap);
    if (!buf) { close(sock); return NULL; }
    char tmp[4096];
    for (;;) {
        ssize_t r = recv(sock, tmp, sizeof(tmp), 0);
        if (r <= 0) break;
        if (len + (size_t)r + 1 > cap) {
            cap *= 2;
            char *nb = (char *)ionic_rc_realloc(buf, cap);
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
        return rt_dup_str(raw);
    }
    size_t n = strlen(sep + 4) + 1;
    char *b = (char *)ionic_rc_alloc(n);
    if (!b) return rt_empty_str();
    memcpy(b, sep + 4, n);
    return b;
}

/* URL-encode a string (percent-encoding). */
char *ionic_http_urlencode(const char *s) {
    if (!s) s = "";
    char *out = (char *)ionic_rc_alloc(strlen(s) * 3 + 1);
    if (!out) return rt_empty_str();
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
    char *out = (char *)ionic_rc_alloc(cap);
    if (!out) return rt_empty_str();
    int64_t *slots = (int64_t *)a->data;
    for (int64_t i = 0; i < a->len; i++) {
        int id = (int)slots[i];
        if (id < 0 || id >= g_bpe.vocab_size || !g_bpe.id_to_str[id]) continue;
        const char *tok = g_bpe.id_to_str[id];
        /* <0xNN> -> raw byte */
        if (tok[0] == '<' && tok[1] == '0' && tok[2] == 'x' && tok[5] == '>' && strlen(tok) == 6) {
            char hex[3] = { tok[3], tok[4], 0 };
            unsigned char b = (unsigned char)strtol(hex, NULL, 16);
            if (len + 2 > cap) { cap *= 2; out = (char *)ionic_rc_realloc(out, cap); }
            out[len++] = (char)b;
            continue;
        }
        /* skip specials */
        if ((tok[0] == '<' && tok[1] == 's' && tok[2] == '>') ||
            (strcmp(tok, "</s>") == 0) || (strcmp(tok, "<unk>") == 0)) continue;
        for (const char *p = tok; *p; ) {
            if ((unsigned char)p[0] == 0xE2 && p[1] && (unsigned char)p[1] == 0x96 &&
                p[2] && (unsigned char)p[2] == 0x81) {
                if (len + 2 > cap) { cap *= 2; out = (char *)ionic_rc_realloc(out, cap); }
                out[len++] = ' ';
                p += 3;
            } else {
                if (len + 2 > cap) { cap *= 2; out = (char *)ionic_rc_realloc(out, cap); }
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
