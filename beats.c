/* beats: write a Kdenlive marker file for a song.
 *
 * One marker per beat, labeled bar.beat. The output sits next to the
 * audio and is named <songname>beats.txt. Kdenlive imports it from the
 * Markers panel: the bottom-right menu, then Import.
 *
 * Steady tempo is assumed. The beat is the gap between kick hits whose
 * phase in the first part of the groove still lands on the kicks later.
 *
 *   gcc -std=c11 -Wall -Wextra -O2 -o beats beats.c -lm
 *   ./beats "song.mp3" "another.flac"
 *   ./beats -m 3 waltz.mp3
 *   ./beats -o /tmp/markers.txt song.mp3
 *
 * Needs ffmpeg on PATH. Decodes through it, so mp3, flac, ogg, m4a, and
 * wav all work.
 */

#define _POSIX_C_SOURCE 200809L
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

enum {
    SR = 22050,
    NFFT = 2048,
    HOP = 256,
    MAX_SECONDS = 60 * 60
};

static const double BPM_MIN = 55.0;
static const double BPM_MAX = 185.0;

typedef struct {
    float re;
    float im;
} C32;

typedef struct {
    float *v;
    size_t n;
    double hop_sec;
} Env;

static void die(const char *msg)
{
    fprintf(stderr, "beats: %s\n", msg);
    exit(1);
}

static void *xmalloc(size_t n)
{
    void *p = malloc(n ? n : 1);
    if (!p) {
        die("out of memory");
    }
    return p;
}

static void *xrealloc(void *p, size_t n)
{
    void *q = realloc(p, n);
    if (!q) {
        die("out of memory");
    }
    return q;
}

static int cmp_float(const void *a, const void *b)
{
    float x = *(const float *)a;
    float y = *(const float *)b;
    return (x > y) - (x < y);
}

static int cmp_double(const void *a, const void *b)
{
    double x = *(const double *)a;
    double y = *(const double *)b;
    return (x > y) - (x < y);
}

static void fft(C32 *a, int n, int inverse)
{
    int i, j, len;

    for (i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) {
            j ^= bit;
        }
        j ^= bit;
        if (i < j) {
            C32 tmp = a[i];
            a[i] = a[j];
            a[j] = tmp;
        }
    }
    for (len = 2; len <= n; len <<= 1) {
        float ang = (inverse ? 2.0f : -2.0f) * (float)M_PI / (float)len;
        float wlen_re = cosf(ang);
        float wlen_im = sinf(ang);
        for (i = 0; i < n; i += len) {
            float wr = 1.0f;
            float wi = 0.0f;
            int half = len / 2;
            for (j = 0; j < half; j++) {
                C32 u = a[i + j];
                C32 v = a[i + j + half];
                float vr = v.re * wr - v.im * wi;
                float vi = v.re * wi + v.im * wr;
                a[i + j].re = u.re + vr;
                a[i + j].im = u.im + vi;
                a[i + j + half].re = u.re - vr;
                a[i + j + half].im = u.im - vi;
                float next_re = wr * wlen_re - wi * wlen_im;
                wi = wr * wlen_im + wi * wlen_re;
                wr = next_re;
            }
        }
    }
    if (inverse) {
        float scale = 1.0f / (float)n;
        for (i = 0; i < n; i++) {
            a[i].re *= scale;
            a[i].im *= scale;
        }
    }
}

/* ffmpeg writes mono float32 little-endian on its stdout. */
static int load_audio(const char *path, float **samples_out, size_t *n_out)
{
    int fds[2];
    pid_t pid;
    size_t n = 0;
    size_t cap = (size_t)SR * 30;
    float *samples;
    FILE *in;
    int status;
    const size_t limit = (size_t)MAX_SECONDS * SR;

    if (pipe(fds) != 0) {
        perror("beats: pipe");
        return -1;
    }
    pid = fork();
    if (pid < 0) {
        perror("beats: fork");
        close(fds[0]);
        close(fds[1]);
        return -1;
    }
    if (pid == 0) {
        char rate[16];
        snprintf(rate, sizeof rate, "%d", SR);
        if (dup2(fds[1], STDOUT_FILENO) < 0) {
            _exit(127);
        }
        close(fds[0]);
        close(fds[1]);
        execlp("ffmpeg", "ffmpeg", "-nostdin", "-v", "error", "-i", path, "-map", "0:a:0", "-ac", "1", "-ar", rate,
               "-f", "f32le", "pipe:1", (char *)NULL);
        fprintf(stderr, "beats: cannot run ffmpeg (%s)\n", strerror(errno));
        _exit(127);
    }
    close(fds[1]);
    in = fdopen(fds[0], "rb");
    if (!in) {
        perror("beats: fdopen");
        close(fds[0]);
        waitpid(pid, &status, 0);
        return -1;
    }
    samples = xmalloc(cap * sizeof(float));
    for (;;) {
        size_t got;
        if (n == cap) {
            if (cap > limit / 2) {
                fprintf(stderr, "beats: %s is longer than %d minutes\n", path, MAX_SECONDS / 60);
                free(samples);
                fclose(in);
                waitpid(pid, &status, 0);
                return -1;
            }
            cap *= 2;
            samples = xrealloc(samples, cap * sizeof(float));
        }
        got = fread(samples + n, sizeof(float), cap - n, in);
        n += got;
        if (got == 0) {
            break;
        }
        if (n > limit) {
            fprintf(stderr, "beats: %s is longer than %d minutes\n", path, MAX_SECONDS / 60);
            free(samples);
            fclose(in);
            waitpid(pid, &status, 0);
            return -1;
        }
    }
    fclose(in);
    if (waitpid(pid, &status, 0) < 0 || !WIFEXITED(status) || WEXITSTATUS(status) != 0) {
        fprintf(stderr, "beats: ffmpeg could not read %s\n", path);
        free(samples);
        return -1;
    }
    if (n < (size_t)SR) {
        fprintf(stderr, "beats: %s has less than a second of audio\n", path);
        free(samples);
        return -1;
    }
    *samples_out = samples;
    *n_out = n;
    return 0;
}

static void onset_envelope(const float *x, size_t n, Env *out)
{
    size_t nframes;
    size_t i;
    float *flux;
    float *prev;
    float *window;
    C32 *frame;
    double mean_w = 0.0;
    int k;
    int smooth;
    float *copy;
    float peak;

    if (n <= (size_t)NFFT) {
        die("audio is too short");
    }
    nframes = 1 + (n - (size_t)NFFT) / (size_t)HOP;
    flux = xmalloc(nframes * sizeof(float));
    prev = xmalloc(((size_t)NFFT / 2 + 1) * sizeof(float));
    window = xmalloc((size_t)NFFT * sizeof(float));
    frame = xmalloc((size_t)NFFT * sizeof(C32));
    memset(prev, 0, ((size_t)NFFT / 2 + 1) * sizeof(float));
    for (k = 0; k < NFFT; k++) {
        window[k] = 0.5f * (1.0f - cosf(2.0f * (float)M_PI * (float)k / (float)(NFFT - 1)));
    }
    for (k = 1; k <= NFFT / 2; k++) {
        double freq = (double)k * (double)SR / (double)NFFT;
        mean_w += 1.0 / sqrt(freq);
    }
    mean_w /= (double)(NFFT / 2);

    for (i = 0; i < nframes; i++) {
        const float *src = x + i * (size_t)HOP;
        double sum = 0.0;
        float first = (i == 0) ? src[0] : src[0] - 0.97f * x[i * (size_t)HOP - 1];
        frame[0].re = first * window[0];
        frame[0].im = 0.0f;
        for (k = 1; k < NFFT; k++) {
            float s = src[k] - 0.97f * src[k - 1];
            frame[k].re = s * window[k];
            frame[k].im = 0.0f;
        }
        fft(frame, NFFT, 0);
        for (k = 1; k <= NFFT / 2; k++) {
            double freq = (double)k * (double)SR / (double)NFFT;
            float mag = hypotf(frame[k].re, frame[k].im);
            float logm = log1pf(80.0f * mag);
            float diff = logm - prev[k];
            prev[k] = logm;
            if (diff > 0.0f) {
                sum += (double)diff / sqrt(freq);
            }
        }
        flux[i] = (float)(sum / mean_w);
    }
    free(prev);
    free(window);
    free(frame);

    smooth = (int)(0.40 * (double)SR / (double)HOP);
    if (smooth < 3) {
        smooth = 3;
    }
    if ((smooth & 1) == 0) {
        smooth++;
    }
    {
        float *floored = xmalloc(nframes * sizeof(float));
        int radius = smooth / 2;
        for (i = 0; i < nframes; i++) {
            size_t a = (i > (size_t)radius) ? i - (size_t)radius : 0;
            size_t b = i + (size_t)radius;
            double acc = 0.0;
            size_t t;
            if (b >= nframes) {
                b = nframes - 1;
            }
            for (t = a; t <= b; t++) {
                acc += flux[t];
            }
            floored[i] = flux[i] - (float)(acc / (double)(b - a + 1));
            if (floored[i] < 0.0f) {
                floored[i] = 0.0f;
            }
        }
        free(flux);
        flux = floored;
    }

    copy = xmalloc(nframes * sizeof(float));
    memcpy(copy, flux, nframes * sizeof(float));
    qsort(copy, nframes, sizeof(float), cmp_float);
    peak = copy[(size_t)((nframes - 1) * 0.99)];
    free(copy);
    if (peak > 1e-8f) {
        for (i = 0; i < nframes; i++) {
            flux[i] /= peak;
        }
    }
    out->v = flux;
    out->n = nframes;
    out->hop_sec = (double)HOP / (double)SR;
}

/* Wide kick band, then the rising edge of its envelope, in milliseconds. */
typedef struct {
    double b0, b1, b2, a1, a2;
} Sos;

/* butterworth bandpass, order 4, 45–160 Hz at 22050 Hz. scipy.signal.butter. */
static const Sos KICK_SOS[4] = {
    {6.90743007424191538e-08, 1.38148601484838308e-07, 6.90743007424191538e-08, -1.96010854158438796e+00,
     9.61190414852714281e-01},
    {1.0, 2.0, 1.0, -1.97893690659726618e+00, 9.79243485053564400e-01},
    {1.0, -2.0, 1.0, -1.97893907923107326e+00, 9.80877117406374510e-01},
    {1.0, -2.0, 1.0, -1.99407308743974143e+00, 9.94247226107280113e-01},
};

static void sos_run(const Sos *sos, int nsec, double *x, size_t n)
{
    int s;
    for (s = 0; s < nsec; s++) {
        double b0 = sos[s].b0, b1 = sos[s].b1, b2 = sos[s].b2;
        double a1 = sos[s].a1, a2 = sos[s].a2;
        double x1 = 0.0, x2 = 0.0, y1 = 0.0, y2 = 0.0;
        size_t i;
        for (i = 0; i < n; i++) {
            double xn = x[i];
            double yn = b0 * xn + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
            x2 = x1;
            x1 = xn;
            y2 = y1;
            y1 = yn;
            x[i] = yn;
        }
    }
}

static void reverse_d(double *x, size_t n)
{
    size_t i;
    for (i = 0; i < n / 2; i++) {
        double tmp = x[i];
        x[i] = x[n - 1 - i];
        x[n - 1 - i] = tmp;
    }
}

static int next_pow2(int n)
{
    int p = 1;
    while (p < n) {
        if (p > 0x3fffffff) {
            die("audio is too long to analyze");
        }
        p <<= 1;
    }
    return p;
}

/* Magnitude of the analytic signal. Replaces y[0..n). */
static void hilbert_envelope(double *y, size_t n)
{
    int nfft = next_pow2((int)n);
    int half;
    int k;
    size_t i;
    C32 *a = xmalloc((size_t)nfft * sizeof(C32));

    memset(a, 0, (size_t)nfft * sizeof(C32));
    for (i = 0; i < n; i++) {
        a[i].re = (float)y[i];
    }
    fft(a, nfft, 0);
    half = nfft / 2;
    for (k = 1; k < half; k++) {
        a[k].re *= 2.0f;
        a[k].im *= 2.0f;
    }
    for (k = half + 1; k < nfft; k++) {
        a[k].re = 0.0f;
        a[k].im = 0.0f;
    }
    fft(a, nfft, 1);
    for (i = 0; i < n; i++) {
        y[i] = hypot(a[i].re, a[i].im);
    }
    free(a);
}

static void kick_attack(const float *x, size_t n, Env *out)
{
    const size_t pad = 2048;
    size_t m = n + 2 * pad;
    size_t ms_n = n / (size_t)(SR / 1000);
    size_t i;
    int win = (int)(0.004 * (double)SR);
    double *band;
    double *smooth;
    double prev;
    float *ms;
    size_t bin = 0;
    float hold = 0.0f;

    if (ms_n == 0 || n <= pad) {
        die("audio is too short");
    }
    band = xmalloc(m * sizeof(double));
    /* Odd reflection, so the forward-backward filter does not ring at the ends. */
    for (i = 0; i < pad; i++) {
        double edge = x[0];
        size_t src = i + 1;
        if (src >= n) {
            src = n - 1;
        }
        band[pad - 1 - i] = 2.0 * edge - x[src];
    }
    for (i = 0; i < n; i++) {
        band[pad + i] = x[i];
    }
    for (i = 0; i < pad; i++) {
        double edge = x[n - 1];
        size_t src = (i + 1 < n) ? n - 2 - i : 0;
        band[pad + n + i] = 2.0 * edge - x[src];
    }
    sos_run(KICK_SOS, 4, band, m);
    reverse_d(band, m);
    sos_run(KICK_SOS, 4, band, m);
    reverse_d(band, m);
    memmove(band, band + pad, n * sizeof(double));

    hilbert_envelope(band, n);
    if (win < 3) {
        win = 3;
    }
    smooth = xmalloc(n * sizeof(double));
    {
        double acc = 0.0;
        for (i = 0; i < n; i++) {
            acc += band[i];
            if (i >= (size_t)win) {
                acc -= band[i - (size_t)win];
            }
            {
                size_t count = (i + 1 < (size_t)win) ? i + 1 : (size_t)win;
                size_t center = (i + 1 > (size_t)(win / 2)) ? i - (size_t)(win / 2) : 0;
                smooth[center] = acc / (double)count;
            }
        }
        for (i = n - (size_t)(win / 2); i < n; i++) {
            smooth[i] = smooth[n - (size_t)(win / 2) - 1];
        }
    }
    free(band);
    prev = smooth[0];
    for (i = 0; i < n; i++) {
        double attack = smooth[i] - prev;
        prev = smooth[i];
        if (attack < 0.0) {
            attack = 0.0;
        }
        smooth[i] = attack;
    }

    ms = xmalloc(ms_n * sizeof(float));
    memset(ms, 0, ms_n * sizeof(float));
    for (i = 0; i < n; i++) {
        size_t slot = i * 1000 / (size_t)SR;
        if (slot >= ms_n) {
            break;
        }
        if (slot != bin) {
            ms[bin] = hold;
            hold = 0.0f;
            bin = slot;
        }
        if ((float)smooth[i] > hold) {
            hold = (float)smooth[i];
        }
    }
    free(smooth);
    if (bin < ms_n) {
        ms[bin] = hold;
    }
    {
        float *copy = xmalloc(ms_n * sizeof(float));
        float peak;
        memcpy(copy, ms, ms_n * sizeof(float));
        qsort(copy, ms_n, sizeof(float), cmp_float);
        peak = copy[(size_t)((ms_n - 1) * 0.99)];
        free(copy);
        if (peak > 1e-8f) {
            for (i = 0; i < ms_n; i++) {
                ms[i] /= peak;
            }
        }
    }
    out->v = ms;
    out->n = ms_n;
    out->hop_sec = 0.001;
}

typedef struct {
    double phase;
    double on_mean;
    double off_mean;
} PhaseScore;

static double sample_at(const Env *e, double t)
{
    size_t i;
    if (t < 0.0) {
        return 0.0;
    }
    i = (size_t)(t / e->hop_sec + 0.5);
    if (i >= e->n) {
        return 0.0;
    }
    return e->v[i];
}

static double sample_peak(const Env *e, double t, double radius)
{
    int c, r, i, a, b;
    float best;
    if (t < 0.0 || e->n == 0) {
        return 0.0;
    }
    c = (int)(t / e->hop_sec + 0.5);
    r = (int)(radius / e->hop_sec + 0.5);
    a = c - r;
    b = c + r;
    if (a < 0) {
        a = 0;
    }
    if (b >= (int)e->n) {
        b = (int)e->n - 1;
    }
    if (a > b) {
        return 0.0;
    }
    best = e->v[a];
    for (i = a + 1; i <= b; i++) {
        if (e->v[i] > best) {
            best = e->v[i];
        }
    }
    return best;
}

static PhaseScore score_phase(const Env *e, double bpm, double phase, double t0, double t1)
{
    double period = 60.0 / bpm;
    double on = 0.0;
    double off = 0.0;
    int on_n = 0;
    int off_n = 0;
    double t;
    PhaseScore result;

    for (t = phase; t < t1; t += period) {
        if (t < t0) {
            continue;
        }
        on += sample_at(e, t);
        on_n++;
        off += sample_at(e, t + period * 0.5);
        off_n++;
    }
    result.phase = phase;
    result.on_mean = on_n ? on / on_n : 0.0;
    result.off_mean = off_n ? off / off_n : 0.0;
    return result;
}

static PhaseScore best_phase(const Env *e, double bpm, double t0, double t1)
{
    double period = 60.0 / bpm;
    int steps = (int)(period / e->hop_sec + 0.5);
    int s;
    PhaseScore best;
    best.phase = 0.0;
    best.on_mean = -1.0;
    best.off_mean = 0.0;
    if (steps < 16) {
        steps = 16;
    }
    /* 1 ms on the kick envelope. Coarser than that walks onto the wrong hit. */
    if (steps > 500) {
        steps = 500;
    }
    for (s = 0; s < steps; s++) {
        PhaseScore cur = score_phase(e, bpm, period * (double)s / (double)steps, t0, t1);
        if (cur.on_mean > best.on_mean) {
            best = cur;
        }
    }
    return best;
}

/* Widen each sample to the loudest hit within radius, so a grid point counts
 * when the kick is that close. The first maximum of this sits on the edge of
 * the plateau; use it to compare tempos, not to place the marker. */
static void envelope_hold(Env *e, double radius)
{
    int r = (int)(radius / e->hop_sec + 0.5);
    float *held;
    size_t i;
    if (r < 1) {
        return;
    }
    held = xmalloc(e->n * sizeof(float));
    for (i = 0; i < e->n; i++) {
        size_t a = (i > (size_t)r) ? i - (size_t)r : 0;
        size_t b = i + (size_t)r;
        size_t k;
        float best;
        if (b >= e->n) {
            b = e->n - 1;
        }
        best = e->v[a];
        for (k = a + 1; k <= b; k++) {
            if (e->v[k] > best) {
                best = e->v[k];
            }
        }
        held[i] = best;
    }
    free(e->v);
    e->v = held;
}

static double tempo_prior(double bpm)
{
    /* Mild pull toward a tapping tempo. Still allows slow and fast songs. */
    double x = log(bpm / 130.0);
    const double s = 0.55;
    return exp(-0.5 * x * x / (s * s));
}

typedef struct {
    double bpm;
    double phase;
    double score;
    double level;
    double err;
    double agree;
} TempoChoice;

static double wrap_err(double err, double period)
{
    err = fmod(err, period);
    if (err < 0.0) {
        err += period;
    }
    if (err > period * 0.5) {
        err -= period;
    }
    return err;
}

static void score_tempo(const Env *env, double bpm, double early0, double early1, double late0, double late1,
                        TempoChoice *out)
{
    PhaseScore early = best_phase(env, bpm, early0, early1);
    PhaseScore late = best_phase(env, bpm, late0, late1);
    double period = 60.0 / bpm;
    double err = wrap_err(late.phase - early.phase, period);
    double agree = exp(-0.5 * (err / 0.010) * (err / 0.010));
    double level = early.on_mean < late.on_mean ? early.on_mean : late.on_mean;
    out->bpm = bpm;
    out->phase = early.phase;
    out->level = level;
    out->err = err;
    out->agree = agree;
    out->score = level * agree * tempo_prior(bpm);
}

static int collect_peaks(const Env *e, double **times_out)
{
    int min_dist = (int)(0.22 / e->hop_sec);
    int last = -100000;
    size_t i;
    size_t cap = 256;
    size_t n = 0;
    double *times = xmalloc(cap * sizeof(double));
    if (min_dist < 2) {
        min_dist = 2;
    }
    for (i = 1; i + 1 < e->n; i++) {
        if (e->v[i] < 0.30f || e->v[i] < e->v[i - 1] || e->v[i] < e->v[i + 1]) {
            continue;
        }
        if ((int)i - last < min_dist) {
            if (e->v[i] > e->v[(size_t)last]) {
                times[n - 1] = (double)i * e->hop_sec;
                last = (int)i;
            }
            continue;
        }
        if (n == cap) {
            cap *= 2;
            times = xrealloc(times, cap * sizeof(double));
        }
        times[n++] = (double)i * e->hop_sec;
        last = (int)i;
    }
    *times_out = times;
    return (int)n;
}

/* Tempos suggested by the gaps between hits, plus half and double time. */
static int tempo_candidates(const double *peaks, int npeaks, double *bpm, int *weight, int max_bpm)
{
    const double ioi_min = 60.0 / BPM_MAX;
    const double ioi_max = 60.0 / BPM_MIN;
    const double bin_w = 0.008;
    int nbins = (int)((ioi_max - ioi_min) / bin_w) + 1;
    int *hist = xmalloc((size_t)nbins * sizeof(int));
    int i, b, filled = 0;
    memset(hist, 0, (size_t)nbins * sizeof(int));
    for (i = 1; i < npeaks; i++) {
        double gap = peaks[i] - peaks[i - 1];
        int bin;
        if (gap < ioi_min || gap >= ioi_max) {
            continue;
        }
        bin = (int)((gap - ioi_min) / bin_w);
        hist[bin] += 2;
        if (i + 1 < npeaks) {
            gap = peaks[i + 1] - peaks[i - 1];
            if (gap >= ioi_min && gap < ioi_max) {
                hist[(int)((gap - ioi_min) / bin_w)] += 1;
            }
        }
    }
    for (b = 1; b < nbins - 1 && filled < max_bpm; b++) {
        if (hist[b] < 4 || hist[b] < hist[b - 1] || hist[b] < hist[b + 1]) {
            continue;
        }
        {
            double center = ioi_min + ((double)b + 0.5) * bin_w;
            double base = 60.0 / center;
            double factors[3] = {1.0, 2.0, 0.5};
            int f;
            for (f = 0; f < 3 && filled < max_bpm; f++) {
                double candidate = base * factors[f];
                int seen = 0;
                int k;
                if (candidate < BPM_MIN || candidate > BPM_MAX) {
                    continue;
                }
                for (k = 0; k < filled; k++) {
                    if (fabs(bpm[k] - candidate) < 1.5) {
                        if (hist[b] > weight[k]) {
                            weight[k] = hist[b];
                        }
                        seen = 1;
                        break;
                    }
                }
                if (!seen) {
                    weight[filled] = hist[b];
                    bpm[filled++] = candidate;
                }
            }
        }
    }
    free(hist);
    return filled;
}

/* Loud stretch of the envelope. The intro and the fade are left out of the
 * tempo test, because a silent bar has no phase to lock. */
static void active_span(const Env *e, double duration, double *t0, double *t1)
{
    double block = 0.5;
    int w = (int)(block / e->hop_sec);
    int nblocks;
    int i;
    double *mean;
    double *copy;
    double gate;
    int first = -1;
    int last = -1;

    *t0 = 0.0;
    *t1 = duration;
    if (w < 1) {
        return;
    }
    nblocks = (int)(e->n / (size_t)w);
    if (nblocks < 8) {
        return;
    }
    mean = xmalloc((size_t)nblocks * sizeof(double));
    for (i = 0; i < nblocks; i++) {
        double acc = 0.0;
        int k;
        for (k = 0; k < w; k++) {
            acc += e->v[(size_t)i * (size_t)w + (size_t)k];
        }
        mean[i] = acc / (double)w;
    }
    copy = xmalloc((size_t)nblocks * sizeof(double));
    memcpy(copy, mean, (size_t)nblocks * sizeof(double));
    qsort(copy, (size_t)nblocks, sizeof(double), cmp_double);
    /* The loud fifth of the song sets the gate. A quiet intro stays outside it. */
    gate = 0.55 * copy[(int)(nblocks * 0.85)];
    free(copy);
    for (i = 0; i < nblocks; i++) {
        if (mean[i] >= gate) {
            if (first < 0) {
                first = i;
            }
            last = i;
        }
    }
    free(mean);
    if (first < 0 || (last - first + 1) * block < 8.0) {
        return;
    }
    *t0 = first * block;
    *t1 = (last + 1) * block;
    if (*t1 > duration) {
        *t1 = duration;
    }
}

static TempoChoice choose_tempo(const Env *env, double region0, double region1)
{
    double span = region1 - region0;
    double early0 = region0;
    double early1 = region0 + span * 0.34;
    double late0 = region1 - span * 0.28;
    double late1 = region1;
    double *peaks = NULL;
    int npeaks;
    double seeds[24];
    int weights[24];
    int nseeds;
    int keep;
    int s, k;
    TempoChoice refined[24];
    int refined_weight[24];
    int nrefined = 0;
    TempoChoice fallback;
    TempoChoice best;
    const double step = 0.02;
    const double reach = 2.5;
    int max_weight = 0;

    best.bpm = 120.0;
    best.phase = 0.0;
    best.score = -1.0;
    best.level = 0.0;
    best.err = 1.0;
    best.agree = 0.0;
    fallback = best;
    if (span < 8.0) {
        early0 = region0;
        early1 = region1;
        late0 = region0;
        late1 = region1;
    }
    npeaks = collect_peaks(env, &peaks);
    nseeds = tempo_candidates(peaks, npeaks, seeds, weights, 24);
    free(peaks);
    if (nseeds == 0) {
        seeds[0] = 120.0;
        weights[0] = 1;
        nseeds = 1;
    }
    for (s = 0; s < nseeds; s++) {
        if (weights[s] > max_weight) {
            max_weight = weights[s];
        }
    }
    /* Drop gaps that only happened a couple of times. Half-time and dotted
     * gaps stay in the list; the lock test below is what rejects them. */
    keep = 0;
    for (s = 0; s < nseeds; s++) {
        if (weights[s] * 2 >= max_weight) {
            seeds[keep] = seeds[s];
            weights[keep] = weights[s];
            keep++;
        }
    }
    nseeds = keep;

    {
        Env held = *env;
        held.v = xmalloc(env->n * sizeof(float));
        memcpy(held.v, env->v, env->n * sizeof(float));
        envelope_hold(&held, 0.025);
        for (s = 0; s < nseeds; s++) {
            TempoChoice trials[251];
            int ntrials = 0;
            double local_max = 0.0;
            TempoChoice local;
            int have = 0;
            int t;
            for (k = -125; k <= 125; k++) {
                double bpm = seeds[s] + (double)k * step;
                if (bpm < BPM_MIN || bpm > BPM_MAX || fabs(bpm - seeds[s]) > reach) {
                    continue;
                }
                score_tempo(&held, bpm, early0, early1, late0, late1, &trials[ntrials]);
                if (trials[ntrials].level > local_max) {
                    local_max = trials[ntrials].level;
                }
                ntrials++;
            }
            local.score = -1.0;
            for (t = 0; t < ntrials; t++) {
                TempoChoice *cur = &trials[t];
                int better;
                /* Ignore a tempo that only agrees by landing off the hits. */
                if (cur->level < local_max * 0.80) {
                    continue;
                }
                better = !have || fabs(cur->err) < fabs(local.err) - 0.002 ||
                         (fabs(cur->err) <= fabs(local.err) + 0.002 && cur->level > local.level);
                if (better) {
                    local = *cur;
                    have = 1;
                }
            }
            /* A real beat keeps its hits. A tempo that only lines up by
             * abandoning them is a subdivision, not the pulse. */
            if (have && (fallback.score < 0.0 || fabs(local.err) < fabs(fallback.err))) {
                fallback = local;
            }
            if (have && fabs(local.err) <= 0.008 && local.level >= local_max * 0.80) {
                refined_weight[nrefined] = weights[s];
                refined[nrefined++] = local;
            }
        }
        free(held.v);
    }

    /* The common gap between hits is the beat. Half-time is that same gap
     * counted twice, so when both lock, keep the faster one. */
    {
        int i;
        int best_i = -1;
        int best_w = -1;
        for (i = 0; i < nrefined; i++) {
            if (refined_weight[i] > best_w) {
                best_w = refined_weight[i];
                best_i = i;
            }
        }
        for (i = 0; i < nrefined; i++) {
            double faster;
            double slower;
            if (best_i < 0 || refined_weight[i] * 5 < best_w * 4) {
                continue;
            }
            faster = refined[i].bpm > refined[best_i].bpm ? refined[i].bpm : refined[best_i].bpm;
            slower = refined[i].bpm > refined[best_i].bpm ? refined[best_i].bpm : refined[i].bpm;
            if (faster > slower * 1.85 && faster < slower * 2.15 && refined[i].bpm > refined[best_i].bpm &&
                refined[i].level >= refined[best_i].level * 0.50) {
                best_i = i;
                best_w = refined_weight[i];
            }
        }
        if (best_i >= 0) {
            best = refined[best_i];
        } else if (fallback.score >= 0.0) {
            best = fallback;
        }
    }
    return best;
}

static int best_rotation(const Env *e, double bpm, double phase, int meter, double t0, double t1)
{
    double period = 60.0 / bpm;
    int rot;
    int best = 0;
    double best_mean = -1.0;
    for (rot = 0; rot < meter; rot++) {
        double sum = 0.0;
        int count = 0;
        double t;
        for (t = phase + (double)rot * period; t < t1; t += period * (double)meter) {
            if (t < t0) {
                continue;
            }
            sum += sample_peak(e, t, 0.015);
            count++;
        }
        if (count && sum / count > best_mean) {
            best_mean = sum / count;
            best = rot;
        }
    }
    return best;
}

/* Move the whole grid onto the kick. The median of the downbeats is used,
 * and only when those downbeats agree, so one flam cannot drag a marker. */
static double attack_shift(const Env *attack, const double *times, int ntimes, int meter, int downbeat, double t0,
                           double t1)
{
    int window = (int)(0.040 / attack->hop_sec);
    double *lags;
    int n = 0;
    int i;
    double shift;
    double spread;
    if (window < 2) {
        return 0.0;
    }
    lags = xmalloc((size_t)ntimes * sizeof(double));
    for (i = 0; i < ntimes; i++) {
        int pos = (i - downbeat) % meter;
        int center;
        int k;
        int best_i;
        float best;
        double t = times[i];
        if (pos < 0) {
            pos += meter;
        }
        if (pos != 0 || t < t0 || t >= t1) {
            continue;
        }
        center = (int)(t / attack->hop_sec + 0.5);
        if (center < window || center + window >= (int)attack->n) {
            continue;
        }
        best = attack->v[center - window];
        best_i = -window;
        for (k = -window + 1; k <= window; k++) {
            float s = attack->v[center + k];
            if (s > best) {
                best = s;
                best_i = k;
            }
        }
        lags[n++] = (double)best_i * attack->hop_sec;
    }
    if (n < 8) {
        free(lags);
        return 0.0;
    }
    qsort(lags, (size_t)n, sizeof(double), cmp_double);
    shift = lags[n / 2];
    spread = lags[(n * 3) / 4] - lags[n / 4];
    free(lags);
    if (spread > 0.025) {
        return 0.0;
    }
    return shift;
}

/* Loudness over about a beat, so one kick in the fade does not look like
 * the groove. */
static void loudness(const float *x, size_t n, Env *out)
{
    int win = (int)(0.30 * (double)SR);
    int hop = SR / 100;
    size_t nb;
    size_t i;
    int k;
    double acc = 0.0;
    float *v;
    if (win < hop) {
        win = hop;
    }
    if (n < (size_t)win) {
        die("audio is too short");
    }
    nb = 1 + (n - (size_t)win) / (size_t)hop;
    v = xmalloc(nb * sizeof(float));
    for (k = 0; k < win; k++) {
        acc += (double)x[k] * (double)x[k];
    }
    v[0] = (float)sqrt(acc / (double)win);
    for (i = 1; i < nb; i++) {
        size_t enter = (size_t)win + (i - 1) * (size_t)hop;
        size_t leave = enter - (size_t)win;
        size_t j;
        for (j = 0; j < (size_t)hop; j++) {
            double a = x[leave + j];
            double b = x[enter + j];
            acc += b * b - a * a;
        }
        if (acc < 0.0) {
            acc = 0.0;
        }
        v[i] = (float)sqrt(acc / (double)win);
    }
    out->v = v;
    out->n = nb;
    out->hop_sec = (double)hop / (double)SR;
}

static void trim_grid(const Env *e, const Env *loud, double *times, int n, int meter, int downbeat, double region0,
                      double region1, int *first, int *last)
{
    double *level;
    double *loudness_at;
    double *downs;
    double *loud_downs;
    double gate;
    double loud_gate;
    int i;
    int ndown = 0;
    int start = -1;
    int end = -1;
    int end_limit;

    level = xmalloc((size_t)n * sizeof(double));
    loudness_at = xmalloc((size_t)n * sizeof(double));
    downs = xmalloc((size_t)n * sizeof(double));
    loud_downs = xmalloc((size_t)n * sizeof(double));
    for (i = 0; i < n; i++) {
        level[i] = sample_peak(e, times[i], 0.030);
        loudness_at[i] = sample_at(loud, times[i]);
        if (times[i] < region0 || times[i] > region1) {
            continue;
        }
        {
            int pos = (i - downbeat) % meter;
            if (pos < 0) {
                pos += meter;
            }
            if (pos == 0) {
                downs[ndown] = level[i];
                loud_downs[ndown] = loudness_at[i];
                ndown++;
            }
        }
    }
    if (ndown == 0) {
        free(level);
        free(loudness_at);
        free(downs);
        free(loud_downs);
        *first = 0;
        *last = n - 1;
        return;
    }
    qsort(downs, (size_t)ndown, sizeof(double), cmp_double);
    qsort(loud_downs, (size_t)ndown, sizeof(double), cmp_double);
    /* Gates from the groove, so a long intro does not lower them. */
    gate = 0.45 * downs[ndown / 2];
    loud_gate = 0.35 * loud_downs[ndown / 2];
    free(downs);
    free(loud_downs);

    for (i = 0; i < n; i++) {
        int pos = (i - downbeat) % meter;
        if (pos < 0) {
            pos += meter;
        }
        if (pos == 0 && level[i] >= gate && loudness_at[i] >= loud_gate) {
            if (start < 0) {
                start = i;
            }
            end = i;
        }
    }
    if (start < 0 || end < start) {
        free(level);
        free(loudness_at);
        *first = 0;
        *last = n - 1;
        return;
    }
    /* Keep later beats in the last bar while the song is still loud. */
    end_limit = end + meter;
    if (end_limit > n) {
        end_limit = n;
    }
    for (i = end + 1; i < end_limit; i++) {
        if (loudness_at[i] < loud_gate) {
            break;
        }
        end = i;
    }
    free(level);
    free(loudness_at);
    *first = start;
    *last = end;
}

static int build_output_path(const char *input, char *dest, size_t dest_n)
{
    const char *base = strrchr(input, '/');
    const char *dot;
    size_t stem;

    base = base ? base + 1 : input;
    dot = strrchr(base, '.');
    if (dot && dot != base) {
        stem = (size_t)(dot - input);
    } else {
        stem = strlen(input);
    }
    if (stem + strlen("beats.txt") + 1 > dest_n) {
        fprintf(stderr, "beats: path is too long: %s\n", input);
        return -1;
    }
    memcpy(dest, input, stem);
    memcpy(dest + stem, "beats.txt", strlen("beats.txt") + 1);
    return 0;
}

static int write_markers(const char *path, double bpm, int meter, const double *times, int first, int last, int downbeat)
{
    FILE *fp = fopen(path, "w");
    int i;
    int bar = 1;
    if (!fp) {
        fprintf(stderr, "beats: cannot write %s (%s)\n", path, strerror(errno));
        return -1;
    }
    fprintf(fp, "# %.2f BPM, %d beats per bar. Comment is bar.beat. Kdenlive: Markers panel, bottom-right menu, Import.\n",
            bpm, meter);
    for (i = first; i <= last; i++) {
        int pos = (i - downbeat) % meter;
        if (pos < 0) {
            pos += meter;
        }
        if (i != first && pos == 0) {
            bar++;
        }
        fprintf(fp, "%.3f %d.%d\n", times[i], bar, pos + 1);
    }
    if (fclose(fp) != 0) {
        fprintf(stderr, "beats: cannot finish %s (%s)\n", path, strerror(errno));
        return -1;
    }
    return 0;
}

static int process_file(const char *input, const char *output, int meter)
{
    float *samples = NULL;
    size_t n = 0;
    Env onset;
    Env attack;
    Env loud;
    const Env *pulse;
    TempoChoice tempo;
    PhaseScore locked;
    double duration;
    double period;
    double *times;
    int nbeats;
    int i;
    int first, last;
    int downbeat;
    int attack_rot;
    int onset_rot;
    double t0, t1;
    char generated[4096];
    const char *out_path = output;

    memset(&onset, 0, sizeof onset);
    memset(&attack, 0, sizeof attack);
    memset(&loud, 0, sizeof loud);
    if (load_audio(input, &samples, &n) != 0) {
        return -1;
    }
    duration = (double)n / (double)SR;
    onset_envelope(samples, n, &onset);
    kick_attack(samples, n, &attack);
    loudness(samples, n, &loud);
    free(samples);

    {
        double kick0, kick1, onset0, onset1;
        TempoChoice from_kick, from_onset;
        active_span(&attack, duration, &kick0, &kick1);
        active_span(&onset, duration, &onset0, &onset1);
        from_kick = choose_tempo(&attack, kick0, kick1);
        from_onset = choose_tempo(&onset, onset0, onset1);
        /* A kick that holds its phase is the thing a cut should land on.
         * The full mix is the fallback when the low end never locks. */
        if (from_kick.agree >= 0.50 && (from_onset.agree < 0.50 || from_kick.level >= 0.15)) {
            tempo = from_kick;
            pulse = &attack;
            t0 = kick0;
            t1 = kick1;
        } else if (from_onset.score > from_kick.score) {
            tempo = from_onset;
            pulse = &onset;
            t0 = onset0;
            t1 = onset1;
        } else {
            tempo = from_kick;
            pulse = &attack;
            t0 = kick0;
            t1 = kick1;
        }
    }
    period = 60.0 / tempo.bpm;
    /* Place the grid on the attack itself. The tempo test used a widened
     * envelope, whose first maximum sits before the hit. */
    locked = best_phase(pulse, tempo.bpm, t0, t1);
    tempo.phase = locked.phase;

    nbeats = (int)((duration - tempo.phase) / period) + 1;
    if (nbeats < meter * 2) {
        fprintf(stderr, "beats: %s is too short for a beat grid\n", input);
        free(onset.v);
        free(attack.v);
        free(loud.v);
        return -1;
    }
    times = xmalloc((size_t)nbeats * sizeof(double));
    for (i = 0; i < nbeats; i++) {
        times[i] = tempo.phase + (double)i * period;
    }

    onset_rot = best_rotation(&onset, tempo.bpm, tempo.phase, meter, t0, t1);
    attack_rot = best_rotation(&attack, tempo.bpm, tempo.phase, meter, t0, t1);
    /* Prefer the kick when it actually picks a beat. A flat low end means
     * every rotation scores about the same, so keep the full-band choice. */
    {
        double kick_best = 0.0;
        double kick_second = 0.0;
        int rot;
        for (rot = 0; rot < meter; rot++) {
            double sum = 0.0;
            int count = 0;
            double t;
            for (t = tempo.phase + (double)rot * period; t < t1; t += period * meter) {
                if (t < t0) {
                    continue;
                }
                sum += sample_peak(&attack, t, 0.015);
                count++;
            }
            if (count) {
                double mean = sum / count;
                if (mean > kick_best) {
                    kick_second = kick_best;
                    kick_best = mean;
                } else if (mean > kick_second) {
                    kick_second = mean;
                }
            }
        }
        downbeat = (kick_second > 0.0 && kick_best > kick_second * 1.25) ? attack_rot : onset_rot;
    }

    {
        double shift = attack_shift(&attack, times, nbeats, meter, downbeat, t0, t1);
        tempo.phase += shift;
        if (tempo.phase >= period) {
            tempo.phase -= period;
        }
        if (tempo.phase < 0.0) {
            tempo.phase += period;
        }
        nbeats = (int)((duration - tempo.phase) / period) + 1;
        times = xrealloc(times, (size_t)nbeats * sizeof(double));
        for (i = 0; i < nbeats; i++) {
            times[i] = tempo.phase + (double)i * period;
        }
    }

    trim_grid(pulse, &loud, times, nbeats, meter, downbeat, t0, t1, &first, &last);

    if (!out_path) {
        if (build_output_path(input, generated, sizeof generated) != 0) {
            free(times);
            free(onset.v);
            free(attack.v);
            free(loud.v);
            return -1;
        }
        out_path = generated;
    }
    if (write_markers(out_path, tempo.bpm, meter, times, first, last, downbeat) != 0) {
        free(times);
        free(onset.v);
        free(attack.v);
        free(loud.v);
        return -1;
    }
    fprintf(stderr, "%s: %.2f BPM, %d beats, %.3f–%.3f -> %s\n", input, tempo.bpm, last - first + 1, times[first],
            times[last], out_path);
    if (tempo.agree < 0.30) {
        fprintf(stderr, "beats: beat is weak in %s; check the markers\n", input);
    }
    free(times);
    free(onset.v);
    free(attack.v);
    free(loud.v);
    return 0;
}

static void usage(void)
{
    fprintf(stderr,
            "usage: beats [-m beats-per-bar] [-o outfile] song [song ...]\n"
            "  writes <songname>beats.txt beside each song\n"
            "  -m 4   beats in a bar (default 4, use 3 for a waltz)\n"
            "  -o     output path, only when one song is given\n");
}

int main(int argc, char **argv)
{
    int meter = 4;
    const char *output = NULL;
    int i;
    int files = 0;
    int failed = 0;

    for (i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            usage();
            return 0;
        }
        if (strcmp(argv[i], "-m") == 0) {
            if (i + 1 >= argc) {
                usage();
                return 1;
            }
            meter = atoi(argv[++i]);
            if (meter < 2 || meter > 12) {
                die("-m must be from 2 to 12");
            }
            continue;
        }
        if (strcmp(argv[i], "-o") == 0) {
            if (i + 1 >= argc) {
                usage();
                return 1;
            }
            output = argv[++i];
            continue;
        }
        if (argv[i][0] == '-' && argv[i][1] != '\0') {
            fprintf(stderr, "beats: unknown option %s\n", argv[i]);
            usage();
            return 1;
        }
        files++;
    }
    if (files == 0) {
        usage();
        return 1;
    }
    if (output && files != 1) {
        die("-o can only be used with one song");
    }
    for (i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0 || strcmp(argv[i], "-o") == 0) {
            i++;
            continue;
        }
        if (argv[i][0] == '-' && argv[i][1] != '\0') {
            continue;
        }
        if (process_file(argv[i], output, meter) != 0) {
            failed++;
        }
    }
    return failed ? 1 : 0;
}
