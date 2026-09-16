/* Memory bandwidth probe for will-it-run-local.
 *
 * Two kernels, because MoE decode is a mixture of both access patterns:
 *
 *   stream  sequential read of one cache line per iteration. This is the
 *           optimistic bound and the one that best predicts reading a large
 *           contiguous expert tensor.
 *
 *   gather  dependent-free random 64-byte reads across the whole buffer. This
 *           is the pessimistic bound: it defeats the prefetcher and exposes
 *           real DRAM latency and channel parallelism.
 *
 * Threads are pinned to distinct cores so the result reflects the memory
 * subsystem rather than the scheduler.
 */
#define _GNU_SOURCE
#include <pthread.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <time.h>

static size_t NBYTES;
static int NTHREADS;
static char *buf;
static volatile uint64_t sink[512];
static unsigned GATHER_ITERS = 4u << 20;

typedef struct { long id; int core; } arg_t;

static void pin(int core) {
    if (core < 0) return;
    cpu_set_t s;
    CPU_ZERO(&s);
    CPU_SET(core, &s);
    pthread_setaffinity_np(pthread_self(), sizeof(s), &s);
}

static void *stream_worker(void *a) {
    arg_t *w = (arg_t *)a;
    pin(w->core);
    size_t chunk = NBYTES / NTHREADS;
    const volatile uint64_t *p = (const volatile uint64_t *)(buf + (size_t)w->id * chunk);
    size_t n = chunk / sizeof(uint64_t);
    uint64_t s = 0;
    /* stride 8 uint64 = 64 B = exactly one cache line */
    for (size_t i = 0; i < n; i += 8) s += p[i];
    sink[w->id] = s;
    return NULL;
}

static void *gather_worker(void *a) {
    arg_t *w = (arg_t *)a;
    pin(w->core);
    uint64_t x = 88172645463325252ULL ^ ((uint64_t)w->id * 2654435761u);
    size_t lines = NBYTES / 64;
    uint64_t s = 0;
    for (unsigned i = 0; i < GATHER_ITERS; i++) {
        x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        s += *(const volatile uint64_t *)(buf + (x % lines) * 64);
    }
    sink[w->id] = s;
    return NULL;
}

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec / 1e9;
}

/* argv: mode(stream|gather) threads gib reps [core0,core1,...] */
int main(int argc, char **argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: membw MODE THREADS GIB REPS [CORELIST]\n");
        return 2;
    }
    const char *mode = argv[1];
    NTHREADS = atoi(argv[2]);
    NBYTES = (size_t)atoll(argv[3]) << 30;
    int reps = atoi(argv[4]);
    if (NTHREADS < 1 || NTHREADS > 512 || NBYTES == 0) return 2;

    int cores[512];
    for (int i = 0; i < NTHREADS; i++) cores[i] = -1;
    if (argc > 5) {
        char *s = strdup(argv[5]), *tok, *save = NULL;
        int i = 0;
        for (tok = strtok_r(s, ",", &save); tok && i < NTHREADS; tok = strtok_r(NULL, ",", &save))
            cores[i++] = atoi(tok);
        free(s);
    }

    buf = aligned_alloc(4096, NBYTES);
    if (!buf) { fprintf(stderr, "alloc %zu failed\n", NBYTES); return 1; }
    /* Fault every page in now so the timed section never takes a page fault. */
    memset(buf, 1, NBYTES);

    int is_stream = strcmp(mode, "stream") == 0;
    /* Keep the gather run near a second regardless of thread count. */
    if (!is_stream && NTHREADS > 0) {
        unsigned target = (unsigned)(48u << 20) / (unsigned)NTHREADS;
        if (target > (1u << 18)) GATHER_ITERS = target;
    }

    pthread_t th[512];
    arg_t args[512];
    for (int rep = 0; rep < reps; rep++) {
        for (long i = 0; i < NTHREADS; i++) { args[i].id = i; args[i].core = cores[i]; }
        /* One unrelated system-wide page-in aborted an otherwise quiet run.
         * Check disk faults in our timed workload; keep diagnostics outside
         * the clock interval so the measured kernels stay the same. */
        struct rusage before, after;
        if (getrusage(RUSAGE_SELF, &before) != 0) {
            perror("getrusage before bandwidth measurement");
            return 1;
        }
        double t0 = now();
        for (long i = 0; i < NTHREADS; i++)
            pthread_create(&th[i], NULL, is_stream ? stream_worker : gather_worker, &args[i]);
        for (long i = 0; i < NTHREADS; i++) pthread_join(th[i], NULL);
        double el = now() - t0;
        if (getrusage(RUSAGE_SELF, &after) != 0) {
            perror("getrusage after bandwidth measurement");
            return 1;
        }
        if (after.ru_majflt > before.ru_majflt) {
            fprintf(stderr, "major page faults during timed bandwidth measurement; result invalid\n");
            return 1;
        }
        double bytes = is_stream ? (double)NBYTES
                                 : (double)NTHREADS * GATHER_ITERS * 64.0;
        printf("%s %d %.6f %.0f\n", mode, rep, el, bytes);
        fflush(stdout);
    }
    return 0;
}
