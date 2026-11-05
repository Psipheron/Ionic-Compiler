/* Reference-counting probe. Answers one question: does every string producer
 * hand back a *counted* block — an 8-byte header immediately before the payload
 * that reads 1 — and does ionic_rc_release reclaim it exactly once?
 *
 * Why this exists
 * ---------------
 * Stage 1 of step 4 (docs/memory-model.md §8) routes every producer through
 * ionic_rc_alloc but emits no retain/release, so nothing in a running program
 * observes the header. That makes the header trivially correct-by-omission and
 * completely untested. Stage 2 emits releases, and a producer that forgot to
 * route through rc_alloc turns the first release into a free() of a pointer
 * eight bytes past a malloc block — a heap corruption that only shows up in the
 * programs that call that producer. This probe is the contract, exercised
 * before anything depends on it.
 *
 * Including the runtime TU directly is deliberate: the rc primitives are
 * `static`, and the question is about *this* runtime, not a copy of it. It also
 * means the probe cannot drift from the code it is checking.
 *
 * Run via scripts/rc_check.sh (it supplies the sanitizer flags).
 */
#include "../src/ionic_model_runtime.c"

static int fails = 0;

/* A producer's result must carry a header reading 1. `rc != 1` means either no
 * header at all (plain malloc — the bug this probe exists to catch) or a shared
 * count (a producer returning something it does not own). Both are wrong here:
 * every producer allocates a fresh block for the caller. */
static void check(const char *name, void *p) {
    if (!p) { printf("FAIL %-22s null\n", name); fails++; return; }
    int64_t *hdr = ((int64_t *)p) - 1;
    int64_t rc = *hdr;
    char *s = (char *)p;
    printf("%-24s ptr=%p rc=%lld len=%zu\n", name, p, (long long)rc, strlen(s));
    if (rc != 1) { printf("  FAIL: expected rc==1, got %lld\n", (long long)rc); fails++; }
}

int main(void) {
    IonicArray *a = ionic_array_alloc(4);

    /* Every path of every producer, including the failure paths. The failure
     * paths are the ones §11.1 was about — they used to return rodata "" and so
     * could not honour the ownership contract at all. */
    check("str_concat",         ionic_str_concat("a", "b"));
    check("int64_to_str",       ionic_int64_to_str(42));
    check("float64_to_str",     ionic_float64_to_str(int_to_fbits(2)));
    check("str_slice",          ionic_str_slice("abcdef", 1, 4));
    check("str_slice(empty)",   ionic_str_slice("abcdef", 4, 2));
    check("str_slice(null)",    ionic_str_slice(NULL, 0, 1));
    check("str_replace",        ionic_str_replace("aXa", "X", "b"));
    check("format",             ionic_format("x{}y", "Z", NULL, NULL, NULL, NULL, NULL, NULL));
    check("char_to_str",        ionic_char_to_str('q'));
    check("str_join(empty)",    ionic_str_join(a, 0, 0, "\n"));
    check("rt_empty_str",       rt_empty_str());
    check("rt_dup_str",         rt_dup_str("dup"));
    check("rt_dup_str(null)",   rt_dup_str(NULL));
    check("file_read(missing)", ionic_file_read("/nonexistent/xyz"));
    check("read_line(eof)",     ionic_read_line());   /* stdin is /dev/null */
    check("http_body(no sep)",  ionic_http_body("no-separator-here"));
    check("http_urlencode",     ionic_http_urlencode("a b"));
    check("bpe_decode(empty)",  ionic_bpe_decode(a));

    /* The three request accessors copy out of the HttpRequest (§11.2), so they
     * are producers now even though their source is request-owned storage. With
     * a null handle they take the rt_dup_str(NULL) -> rt_empty_str path, the
     * same one a missing request hits. These were the *interior* class: handing
     * back `r->method` put the release word inside a calloc'd block, so they
     * must be counted like anything else. */
    check("http_method(null)",  ionic_http_method(0));
    check("http_path(null)",    ionic_http_path(0));
    check("http_req_body(null)",ionic_http_req_body(0));

    /* str_join over real elements: one separator after every element, per
     * lib/std/text.ionic's contract. */
    char *e0 = (char *)ionic_rc_alloc(2); e0[0] = 'p'; e0[1] = '\0';
    char *e1 = (char *)ionic_rc_alloc(2); e1[0] = 'q'; e1[1] = '\0';
    ionic_array_push_i64(a, (int64_t)e0);
    ionic_array_push_i64(a, (int64_t)e1);
    char *j = ionic_str_join(a, 0, 2, "|");
    check("str_join(\"p|q|\")", j);
    if (strcmp(j, "p|q|") != 0) { printf("  FAIL: got \"%s\"\n", j); fails++; }
    /* A count past the end is clamped rather than read past. */
    char *jc = ionic_str_join(a, 0, 99, "|");
    if (strcmp(jc, "p|q|") != 0) { printf("  FAIL: clamp got \"%s\"\n", jc); fails++; }

    /* A pinned literal copy: release must be a no-op, and the same literal must
     * intern to the same address (codegen interns each literal node to a fixed
     * __cstring offset, so the rodata address is the key). */
    const char *l1 = ionic_str_lit("literal-x");
    const char *l2 = ionic_str_lit("literal-x");
    printf("%-24s ptr=%p\n", "str_lit", (const void *)l1);
    if (l1 != l2) { printf("  FAIL: literal not interned\n"); fails++; }
    if (((int64_t *)l1)[-1] != IONIC_RC_PINNED) { printf("  FAIL: not pinned\n"); fails++; }
    ionic_rc_release((void *)l1);
    if (((int64_t *)l1)[-1] != IONIC_RC_PINNED) { printf("  FAIL: pin changed\n"); fails++; }
    if (strcmp(l1, "literal-x") != 0) { printf("  FAIL: literal corrupted\n"); fails++; }

    /* retain/release balance: 1 -> 3 -> 1, still alive. */
    char *r = (char *)ionic_rc_alloc(16);
    strcpy(r, "balanced");
    ionic_rc_retain(r); ionic_rc_retain(r);
    ionic_rc_release(r); ionic_rc_release(r);
    if (((int64_t *)r)[-1] != 1) { printf("  FAIL: rc after balance\n"); fails++; }
    ionic_rc_release(r);                        /* freed */

    /* The underflow guard defends against a *corrupt* zero header, not against
     * double release: a second release reads freed memory (the header load is
     * the use-after-free) before any guard could look at the count. So there is
     * no `release(p); release(p);` case here — it would only demonstrate that
     * ASan detects a bug the runtime cannot prevent. The discipline that keeps
     * this correct is one release per owning slot from a single emit point; see
     * the ionic_rc_release comment and docs/memory-model.md §8 step 4. */

    /* rc_realloc grows a counted payload, preserving header and contents. */
    char *g = (char *)ionic_rc_alloc(8);
    strcpy(g, "grow");
    g = (char *)ionic_rc_realloc(g, 64);
    if (!g) { printf("FAIL rc_realloc null\n"); fails++; }
    else {
        printf("%-24s rc=%lld content=\"%s\"\n", "rc_realloc(8->64)",
               (long long)(((int64_t *)g)[-1]), g);
        if (((int64_t *)g)[-1] != 1) { printf("  FAIL: header lost\n"); fails++; }
        if (strcmp(g, "grow") != 0)  { printf("  FAIL: content lost\n"); fails++; }
        ionic_rc_release(g);
    }

    /* rc_realloc on a pinned block must copy into a live one, must not read past
       the interned copy's end, and must leave the interned original intact. */
    void *pg = ionic_rc_realloc((void *)l1, 64);
    printf("%-24s rc=%lld content=\"%s\"\n", "rc_realloc(pinned)",
           (long long)(((int64_t *)pg)[-1]), (char *)pg);
    if (((int64_t *)pg)[-1] != 1) { printf("  FAIL: pinned copy not live\n"); fails++; }
    if (strcmp((char *)pg, "literal-x") != 0) { printf("  FAIL: pinned copy content\n"); fails++; }
    if (((int64_t *)l1)[-1] != IONIC_RC_PINNED) { printf("  FAIL: interned literal mutated\n"); fails++; }
    if (strcmp(l1, "literal-x") != 0) { printf("  FAIL: interned literal corrupted\n"); fails++; }
    ionic_rc_release(pg);

    printf("\n%s (%d failures)\n", fails ? "FAIL" : "PASS", fails);
    return fails ? 1 : 0;
}
