

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
    /* How many handles name this allocation. `let b = a` makes this 2, and
     * `ionic_mlx_free` releases only when the last one goes. This fits the
     * padding `owns_array` already left before the next 8-byte member, so the
     * struct stays 40 bytes and no call site moves. */
    uint32_t   rc;
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

/* How many bytes the device backing `pool` can actually hold, or 0 if unknown.
 *
 * For VRAM this is Metal's `max_recommended_working_set_size`, not the card's
 * nominal memory: on Apple silicon the GPU may address only a slice of the
 * unified RAM, and that slice is the number an allocator will honour. Asking
 * for `memory_size` instead would report 8 GiB on a machine that Metal caps at
 * 5.3 GiB, and the whole point of the comparison is to catch exactly that gap.
 *
 * For SRAM it is physical host memory — the pool has no other limit, because
 * `vram` names the accelerator pool (see g_pool_capacity above). */
static int64_t mlx_device_capacity(int64_t pool) {
    if (pool == IONIC_POOL_CPU) return host_ram_bytes();
    if (pool != IONIC_POOL_GPU) return 0;

    ionic_mlx_init();
    mlx_device      dev  = mlx_device_new_type(MLX_GPU, 0);
    mlx_device_info info = mlx_device_info_new();   /* empty; get() fills it */
    int64_t         out  = 0;

    if (mlx_ck(mlx_device_info_get(&info, dev)) == 0) {
        size_t v = 0;
        if (mlx_ck(mlx_device_info_get_size(
                &v, info, "max_recommended_working_set_size")) == 0) {
            out = (int64_t)v;
        }
    }
    mlx_device_info_free(info);
    mlx_device_free(dev);
    return out;
}

/* Ionic-facing: how many bytes pool `p` really holds, or 0 when unknown.
 *
 * `mlx_pool_capacity(p)` answers "how much did the program say it may use",
 * this answers "how much is there". A program (or a test) needs both to reason
 * about whether its declaration is honest on the machine it is running on. */
int64_t ionic_mlx_device_capacity(int64_t pool) { return mlx_device_capacity(pool); }

/* Pools whose declaration has already been compared against the hardware.
 *
 * The comparison is a warning, and a warning printed once per pool is a
 * diagnostic; one printed per allocation is noise the user will scroll past. */
static int g_pool_checked[IONIC_POOL_COUNT] = {0, 0};

/* Format a byte count as GiB with one decimal, for diagnostics.
 *
 * Byte counts are unreadable at ten digits and the question this answers is
 * "which of these two numbers is bigger", so the comparison is shown in the
 * units it is actually made in. Integer arithmetic on tenths avoids rounding
 * surprises; `bytes * 10` cannot overflow for any plausible capacity. */
static void fmt_gib(char *buf, size_t cap, int64_t bytes) {
    int64_t tenths = (bytes * 10) / (int64_t)(1024 * 1024 * 1024);
    snprintf(buf, cap, "%lld.%lld GiB",
             (long long)(tenths / 10), (long long)(tenths % 10));
}

/* Report, once, when a declared pool is larger than the device backing it.
 *
 * Deliberately a warning and not a refusal. The declaration is a budget the
 * program promises to stay under, and a budget larger than today's device is
 * a program that will run on a bigger machine — refusing that would make the
 * declaration a hardware requirement it was never documented to be. What is
 * not acceptable is silence: `let vram = 25769803776;` on an 8 GiB laptop is a
 * number that can never be honoured here, and an author who wrote it deserves
 * to be told which two numbers disagree.
 *
 * Called from pool_charge() rather than ionic_pool_set_capacity(). Setting the
 * capacity happens before any user code runs, and this repo's own `vram`
 * examples are compile-only references that declare 24 GiB and never allocate;
 * comparing at startup would warn on programs that are simply not running yet.
 * The declaration only becomes a claim about this machine when the program asks
 * this machine for accelerator bytes. */
static void pool_check_declaration(int64_t pool) {
    if (!pool_valid(pool) || g_pool_checked[pool]) return;
    g_pool_checked[pool] = 1;

    int64_t declared = g_pool_capacity[pool];
    int64_t device   = mlx_device_capacity(pool);
    if (declared <= 0 || device <= 0 || declared <= device) return;

    char want[32], have[32];
    fmt_gib(want, sizeof(want), declared);
    fmt_gib(have, sizeof(have), device);
    fprintf(stderr,
        "ionic: warning: %s is declared as %s but this device offers %s\n"
        "       the first allocation will be charged against the %s figure, so "
        "this program cannot use the budget it declares on this machine. The "
        "declaration is legal (it will fit a larger device); lower `let vram` to "
        "match if this program is meant to run here.\n",
        pool_name(pool), want, have, have);
}

/* Charge `bytes` to `pool`, refusing the allocation if it would not fit.
 *
 * Release is unconditional: giving bytes back can never over-fill. The
 * comparison avoids `used + bytes` so a huge request cannot overflow into a
 * negative sum and slip past the check. */
static void pool_charge(int64_t pool, int64_t bytes) {
    if (!pool_valid(pool)) return;
    if (bytes > 0) pool_check_declaration(pool);
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
    t->rc         = 1;
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

/* Take another reference to a handle. Emitted by codegen wherever a handle
 * value is *copied* rather than moved: `let b = a;`, passing a tensor to a
 * function, and `return x`. Returns the handle so the caller can use the result
 * as a value.
 *
 * This is what makes `let b = a` mean what the docs say it means. Before it,
 * both names held the same `MlxTensor*`, so `mlx_free(a)` freed the struct and
 * left `b` dangling: `mlx_tensor_of(b)` then read `magic` out of reclaimed
 * memory, printed "bad tensor handle", and returned -1. Because the handle is
 * 40 bytes — the same size class as much of the rest of the heap — the free
 * usually *succeeded* and the program exited 0 with a wrong answer. */
void *ionic_mlx_retain(void *h) {
    MlxTensor *t = mlx_tensor_of(h);
    if (!t) return h;
    t->rc += 1;
    return h;
}

/* Free a tensor reference. Null-tolerant.
 *
 * The handle is reference counted: `let b = a` and a tensor argument each add a
 * reference, so this releases the allocation only when the last name for it
 * goes. A second `mlx_free` on a name that is already dead is caught by the
 * `alive` flag (the checker rejects it statically; this is the backstop).
 *
 * A handle that no longer owns its array (it was moved away from) releases
 * nothing: the bytes belong to the handle the transfer produced. Releasing the
 * array twice would be a double free of the same refcount, which MLX turns
 * into a use-after-free the moment either handle is read. */
void ionic_mlx_free(void *h) {
    MlxTensor *t = mlx_tensor_of(h);
    if (!t) return;
    if (t->owns_array) {
        /* A live owner. Drop this reference; the last one releases the bytes. */
        if (t->rc > 1) { t->rc -= 1; return; }
        pool_charge(t->pool, -t->bytes);
        mlx_array_free(t->a);
    }
    /* Either the last owner, or a tombstone left by a move. Neither has bytes
     * left to give back, so both just retire the struct. */
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
/* No handles exist without a backend, so there is nothing to count. Returns the
 * argument unchanged so the emitted `retain(h)` call is still a no-op value
 * expression rather than a NULL that would poison a subsequent assignment. */
void   *ionic_mlx_retain(void *h) { return h; }
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
/* The host pool's real size is a fact about the machine, so it is answerable
 * here too; VRAM is 0 because without an accelerator there is no such device
 * to report, and 0 means "unknown", not "empty". */
int64_t ionic_mlx_device_capacity(int64_t p) {
    return p == IONIC_POOL_CPU ? host_ram_bytes() : 0;
}
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
