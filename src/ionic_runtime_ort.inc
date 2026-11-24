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
