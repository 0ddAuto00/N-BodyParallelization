/* Implemented by Graceson and Larry
 *
 *  All-tiled pairs implementation
 *
 *  Same physics as sequential.c (direct O(N^2) summation, Plummer softening,
 *  kick-drift-kick leapfrog), but the N x N interaction matrix is processed
 *  in p x p tiles, and only tiles on or above the diagonal are evaluated:
 *  F_ji = -F_ij, so every pair is computed once, N(N-1)/2 evaluations.
 *
 *  One step = one OpenMP parallel region with three phases:
 *    1. kick + drift on the Body array, then copy positions and G*m into
 *       structure-of-arrays (x[], y[], z[], m[]) so the kernels can do
 *       contiguous SIMD loads.
 *    2. tiles (I, J), J >= I, handed out with schedule(dynamic, 1).
 *       Tile (I, J) adds to a_i for i in block I AND to a_j for j in
 *       block J, so two threads could hit the same body at the same time.
 *       Each thread therefore accumulates into its own private
 *       acceleration arrays: no atomics, no locks.
 *    3. per body, sum the private arrays (zeroing them for the next step),
 *       store b[i].acc and apply the second half-kick.
 *
 *  Kernels, picked at run time from the CPU (override: NBODY_SIMD):
 *    avx512   8 doubles / 16 floats per vector, 4 target bodies per pass
 *    avx2     4 doubles /  8 floats per vector, 2 target bodies per pass
 *    generic  plain C + #pragma omp simd, any CPU
 *  sqrt and divide are the slowest FP instructions and run on a single
 *  divider unit. The AVX-512 kernel replaces them with the hardware 1/sqrt
 *  estimate refined to full precision by a few FMAs. The AVX2 kernel does
 *  that for one of its two target bodies and uses sqrt + divide for the
 *  other, so the divider and the FMA units work at the same time.
 *  No special compiler flags are needed.
 *
 *  Knobs:
 *    -DUSE_FLOAT                        forces in float (state stays double)
 *    -DTILE_SIZE=p                      default tile edge (multiple of 16)
 *    NBODY_TILE=p                       env: fixed tile edge, no rebuild
 *    NBODY_SIMD=avx512|avx2|generic     env: force a kernel
 *    -DTILED_NO_SIMD                    build only the generic kernel
 */

#include "headers/tiled.h"

#include <stdint.h>

/* Windows + GCC: GCC can't align the stack past 16 bytes there (GCC bug 54412)
 * yet may spill AVX registers with aligned moves, which crash when the slot is
 * misaligned. MSYS2's GCC (what nbody.exe was built with) emits unaligned moves
 * by default, so it is safe. With another MinGW GCC (e.g. WinLibs) build with
 * CFLAGS_EXTRA=-Wa,-muse-unaligned-vector-move, or with -DTILED_NO_SIMD. */
#if (defined(__x86_64__) || defined(__i386__)) && defined(__GNUC__) && !defined(TILED_NO_SIMD)
#define TILED_X86 1
#include <immintrin.h>
#else
#define TILED_X86 0
#endif

#ifdef USE_FLOAT
typedef float real;
#define REAL_NAME "float"
#else
typedef double real;
#define REAL_NAME "double"
#endif

#ifndef TILE_SIZE
#define TILE_SIZE 256          /* default p; shrunk automatically for small N */
#endif
#define TILE_ALIGN       16    /* p is a multiple of this (widest vector, 4-row unroll) */
#define TILE_MIN         32    /* auto p never goes below this */
#define TILES_PER_THREAD 4     /* auto p: shrink until there are this many tiles per thread */
#define RED_CHUNK        64    /* bodies per work item in the reduction phase */
#define PAD_POS          1e18  /* where padding bodies (mass 0) are parked */

extern double* tiledTimes;

typedef void (*offdiag_fn)(const real*, const real*, const real*, const real*,
                           real*, real*, real*, size_t, size_t, size_t, real);
typedef void (*diag_fn)(const real*, const real*, const real*, const real*,
                        real*, real*, real*, size_t, size_t, real);

typedef struct {
    const char* name;
    offdiag_fn  offdiag;   /* tile (I, J), I < J */
    diag_fn     diag;      /* tile (I, I), pairs i < j only */
} Kernels;

/* Scratch space kept between calls; rebuilt only when n, p or the thread
 * count changes (outside the timed loop in implement_tiled). */
static struct {
    size_t    n, npad, p, ntiles, stride;
    int       nthreads;
    void*     raw;
    char*     base;
    real     *x, *y, *z, *m;
    uint32_t *tile_i, *tile_j;
    Kernels   k;
} ws;

/* round_up()
 * Rounds v up to the next multiple of a.
 * Used to pad N to a whole number of tiles and to keep the tile edge and
 * array sizes multiples of the SIMD width and cache-line size.
 */
static size_t round_up(size_t v, size_t a) { return (v + a - 1) / a * a; }

/* acc_of()
 * Returns thread t's private accumulator array for component c
 * (0 = x, 1 = y, 2 = z). Each array holds one value per (padded) body and
 * lives at its own offset inside the workspace block.
 */
static inline real* acc_of(size_t t, size_t c) { return (real*)(ws.base + (4 + 3 * t + c) * ws.stride); }

/* =========================================================================
 *  Kernels
 *
 *  Arguments: SoA positions x/y/z, masses m (already multiplied by G),
 *  this thread's accumulators ax/ay/az, tile origin(s), tile edge p,
 *  softening^2. Every array is 64-byte aligned and p is a multiple of 16,
 *  so all vector loads/stores are aligned and there are no remainder loops.
 *
 *  For a pair (i, j) with d = r_j - r_i and w = (|d|^2 + eps^2)^(-3/2):
 *  a_i += m_j w d  and  a_j -= m_i w d.
 * ========================================================================= */

/* ---- generic: plain C, vectorized by the compiler ---------------------- */

/* rsqrt_generic()
 * Computes 1/sqrt(x) without calling sqrt() or dividing.
 * sqrt() may set errno, which stops GCC from vectorizing the loop it is in,
 * so instead: start from the bit-level guess made famous by Quake III
 * (~5 correct bits), then apply Newton steps y <- y (3/2 - x/2 y^2), each
 * doubling the correct bits: 3 steps for float, 4 for double.
 * Used only by the generic (non-AVX) kernels.
 */
static inline real rsqrt_generic(real x)
{
    const real h = (real)0.5 * x;
#ifdef USE_FLOAT
    uint32_t bits;
    float y;
    memcpy(&bits, &x, sizeof bits);
    bits = UINT32_C(0x5f375a86) - (bits >> 1);
    memcpy(&y, &bits, sizeof y);
    y = y * (1.5f - h * y * y);
    y = y * (1.5f - h * y * y);
    y = y * (1.5f - h * y * y);
#else
    uint64_t bits;
    double y;
    memcpy(&bits, &x, sizeof bits);
    bits = UINT64_C(0x5fe6eb50c7b537a9) - (bits >> 1);
    memcpy(&y, &bits, sizeof y);
    y = y * (1.5 - h * y * y);
    y = y * (1.5 - h * y * y);
    y = y * (1.5 - h * y * y);
    y = y * (1.5 - h * y * y);
#endif
    return y;
}

/* offdiag_generic()
 * Portable kernel for an off-diagonal tile (I, J), I < J.
 * For every pair (i in block I, j in block J) it computes the force once and
 * applies it to both bodies: a_i += m_j w d, a_j -= m_i w d, with
 * d = r_j - r_i and w = (|d|^2 + eps^2)^(-3/2).
 *   x, y, z, m   SoA positions and G*mass (read only)
 *   ax, ay, az   this thread's private accumulators (updated)
 *   i0, j0       first body of block I and of block J
 *   p            tile edge (bodies per block)
 *   eps2         softening squared
 * The inner loop over j is vectorized by the compiler (#pragma omp simd).
 */
static void offdiag_generic(const real* restrict x, const real* restrict y, const real* restrict z,
                            const real* restrict m, real* restrict ax, real* restrict ay,
                            real* restrict az, size_t i0, size_t j0, size_t p, real eps2)
{
    for (size_t i = i0; i < i0 + p; i++) {
        const real xi = x[i], yi = y[i], zi = z[i], mi = m[i];
        real sx = 0, sy = 0, sz = 0;
#pragma omp simd reduction(+ : sx, sy, sz)
        for (size_t j = j0; j < j0 + p; j++) {
            const real dx = x[j] - xi, dy = y[j] - yi, dz = z[j] - zi;
            const real ri = rsqrt_generic(dx * dx + dy * dy + dz * dz + eps2);
            const real w = ri * ri * ri, sj = m[j] * w, si = mi * w;
            sx += sj * dx;    sy += sj * dy;    sz += sj * dz;
            ax[j] -= si * dx; ay[j] -= si * dy; az[j] -= si * dz;
        }
        ax[i] += sx; ay[i] += sy; az[i] += sz;
    }
}

/* diag_generic()
 * Portable kernel for a diagonal tile (I, I): only pairs i < j inside the
 * block, so every pair is counted once and self-interaction is skipped.
 * Arguments as offdiag_generic, with i0 the first body of the block.
 */
static void diag_generic(const real* restrict x, const real* restrict y, const real* restrict z,
                         const real* restrict m, real* restrict ax, real* restrict ay,
                         real* restrict az, size_t i0, size_t p, real eps2)
{
    for (size_t i = i0; i < i0 + p; i++) {
        const real xi = x[i], yi = y[i], zi = z[i], mi = m[i];
        real sx = 0, sy = 0, sz = 0;
#pragma omp simd reduction(+ : sx, sy, sz)
        for (size_t j = i + 1; j < i0 + p; j++) {
            const real dx = x[j] - xi, dy = y[j] - yi, dz = z[j] - zi;
            const real ri = rsqrt_generic(dx * dx + dy * dy + dz * dz + eps2);
            const real w = ri * ri * ri, sj = m[j] * w, si = mi * w;
            sx += sj * dx;    sy += sj * dy;    sz += sj * dz;
            ax[j] -= si * dx; ay[j] -= si * dy; az[j] -= si * dz;
        }
        ax[i] += sx; ay[i] += sy; az[i] += sz;
    }
}

#if TILED_X86

#define TARGET_AVX512 __attribute__((target("avx512f,avx2,fma")))
#define TARGET_AVX2   __attribute__((target("avx2,fma")))

/* r^-3 from r^2 and an estimate y0 of 1/r:
 *   e = 1 - r2*y0^2,   r^-3 = y0^3 (1-e)^(-3/2)
 *                           = y0^3 (1 + 3/2 e + 15/8 e^2 + 35/16 e^3 + 315/128 e^4 + 693/256 e^5 ...)
 * The series is cut once the next term is below the precision of `real`. */

/* ---- AVX-512 ------------------------------------------------------------ */

#ifdef USE_FLOAT
typedef __m512    zv;
typedef __mmask16 zmask;
#define ZW       16
#define z_set1   _mm512_set1_ps
#define z_zero   _mm512_setzero_ps
#define z_load   _mm512_load_ps
#define z_store  _mm512_store_ps
#define z_add    _mm512_add_ps
#define z_sub    _mm512_sub_ps
#define z_mul    _mm512_mul_ps
#define z_fmadd  _mm512_fmadd_ps
#define z_fnmadd _mm512_fnmadd_ps
#define z_rsqrt  _mm512_rsqrt14_ps
#define z_rsqrtz _mm512_maskz_rsqrt14_ps
#define z_hsum   _mm512_reduce_add_ps
#else
typedef __m512d  zv;
typedef __mmask8 zmask;
#define ZW       8
#define z_set1   _mm512_set1_pd
#define z_zero   _mm512_setzero_pd
#define z_load   _mm512_load_pd
#define z_store  _mm512_store_pd
#define z_add    _mm512_add_pd
#define z_sub    _mm512_sub_pd
#define z_mul    _mm512_mul_pd
#define z_fmadd  _mm512_fmadd_pd
#define z_fnmadd _mm512_fnmadd_pd
#define z_rsqrt  _mm512_rsqrt14_pd
#define z_rsqrtz _mm512_maskz_rsqrt14_pd
#define z_hsum   _mm512_reduce_add_pd
#endif

/* z_rinv3()
 * AVX-512: returns r^-3 for each lane, given r2 = |d|^2 + eps^2 and
 * y0, the hardware estimate of 1/r (rsqrt14, relative error < 2^-14).
 * With e = 1 - r2*y0^2, r^-3 = y0^3 (1-e)^(-3/2), and the series in e is cut
 * once the next term is below the precision of `real` (1 term for float,
 * 4 for double). This replaces a sqrt and a divide with a few FMAs.
 */
static inline TARGET_AVX512 zv z_rinv3(zv r2, zv y0)
{
    const zv one = z_set1(1);
    const zv y2  = z_mul(y0, y0);
    const zv e   = z_fnmadd(r2, y2, one);
    const zv y3  = z_mul(y2, y0);
#ifdef USE_FLOAT
    const zv s = z_fmadd(e, z_set1(1.5f), one);
#else
    zv s = z_fmadd(e, z_set1(315.0 / 128), z_set1(35.0 / 16));
    s    = z_fmadd(e, s, z_set1(15.0 / 8));
    s    = z_fmadd(e, s, z_set1(1.5));
    s    = z_fmadd(e, s, one);
#endif
    return z_mul(y3, s);
}

/* z_add_hsum4()
 * AVX-512: adds the horizontal sums of four vectors a, b, c, d to
 * arr[0..3]. One shuffle tree reduces all four rows at once, which is
 * cheaper than four separate horizontal reductions. arr must be aligned.
 */
static inline TARGET_AVX512 void z_add_hsum4(real* arr, zv a, zv b, zv c, zv d)
{
#ifdef USE_FLOAT
    const zv ab = z_add(_mm512_shuffle_f32x4(a, b, 0x44), _mm512_shuffle_f32x4(a, b, 0xEE));
    const zv cd = z_add(_mm512_shuffle_f32x4(c, d, 0x44), _mm512_shuffle_f32x4(c, d, 0xEE));
    zv s = z_add(_mm512_shuffle_f32x4(ab, cd, 0x88), _mm512_shuffle_f32x4(ab, cd, 0xDD));
    s = z_add(s, _mm512_permute_ps(s, 0x4E));
    s = z_add(s, _mm512_permute_ps(s, 0xB1));
    s = _mm512_permutexvar_ps(_mm512_setr_epi32(0, 4, 8, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), s);
    _mm_store_ps(arr, _mm_add_ps(_mm_load_ps(arr), _mm512_castps512_ps128(s)));
#else
    const zv ab = z_add(_mm512_shuffle_f64x2(a, b, 0x44), _mm512_shuffle_f64x2(a, b, 0xEE));
    const zv cd = z_add(_mm512_shuffle_f64x2(c, d, 0x44), _mm512_shuffle_f64x2(c, d, 0xEE));
    zv s = z_add(_mm512_shuffle_f64x2(ab, cd, 0x88), _mm512_shuffle_f64x2(ab, cd, 0xDD));
    s = z_add(s, _mm512_permute_pd(s, 0x55));
    s = _mm512_permutexvar_pd(_mm512_setr_epi64(0, 2, 4, 6, 0, 0, 0, 0), s);
    _mm256_store_pd(arr, _mm256_add_pd(_mm256_load_pd(arr), _mm512_castpd512_pd256(s)));
#endif
}

/* Target body (XI,YI,ZI,MI) against the source vector (xj,yj,zj,mj):
 * a_i accumulates in (SX,SY,SZ), the reaction is subtracted from the source
 * accumulators (axj,ayj,azj). Y0 is the 1/r estimate, an expression of r2_. */
#define Z_PAIR(XI, YI, ZI, MI, SX, SY, SZ, Y0)                                          \
    do {                                                                                \
        const zv dx_ = z_sub(xj, XI), dy_ = z_sub(yj, YI), dz_ = z_sub(zj, ZI);         \
        const zv r2_ = z_fmadd(dz_, dz_, z_fmadd(dy_, dy_, z_fmadd(dx_, dx_, veps)));   \
        const zv w_  = z_rinv3(r2_, Y0);                                                \
        const zv sj_ = z_mul(mj, w_), si_ = z_mul(MI, w_);                              \
        SX  = z_fmadd(sj_, dx_, SX);   SY  = z_fmadd(sj_, dy_, SY);   SZ  = z_fmadd(sj_, dz_, SZ);   \
        axj = z_fnmadd(si_, dx_, axj); ayj = z_fnmadd(si_, dy_, ayj); azj = z_fnmadd(si_, dz_, azj); \
    } while (0)

#define Z_LOAD_J(J)                                                                     \
    const zv xj = z_load(x + (J)), yj = z_load(y + (J)), zj = z_load(z + (J)), mj = z_load(m + (J)); \
    zv axj = z_load(ax + (J)), ayj = z_load(ay + (J)), azj = z_load(az + (J))

#define Z_STORE_J(J) \
    z_store(ax + (J), axj); z_store(ay + (J), ayj); z_store(az + (J), azj)

#define Z_ROW(R)                                                                        \
    const zv xi##R = z_set1(x[i + R]), yi##R = z_set1(y[i + R]),                        \
             zi##R = z_set1(z[i + R]), mi##R = z_set1(m[i + R]);                        \
    zv sx##R = z_zero(), sy##R = z_zero(), sz##R = z_zero()

/* offdiag_avx512()
 * AVX-512 kernel for an off-diagonal tile (I, J), I < J.
 * Each pass takes 4 target bodies from block I (broadcast into registers)
 * and sweeps block J one vector at a time (8 doubles / 16 floats), so each
 * source vector is loaded once and used 4 times. a_i is summed in
 * registers and added once per pass; the reaction on block J is written
 * back to the source accumulators every vector.
 * Arguments as offdiag_generic.
 */
static TARGET_AVX512 void offdiag_avx512(const real* restrict x, const real* restrict y,
                                         const real* restrict z, const real* restrict m,
                                         real* restrict ax, real* restrict ay, real* restrict az,
                                         size_t i0, size_t j0, size_t p, real eps2)
{
    const zv veps = z_set1(eps2);
    for (size_t i = i0; i < i0 + p; i += 4) {          /* 4 target bodies per pass over J */
        Z_ROW(0); Z_ROW(1); Z_ROW(2); Z_ROW(3);
        for (size_t j = j0; j < j0 + p; j += ZW) {
            Z_LOAD_J(j);
            Z_PAIR(xi0, yi0, zi0, mi0, sx0, sy0, sz0, z_rsqrt(r2_));
            Z_PAIR(xi1, yi1, zi1, mi1, sx1, sy1, sz1, z_rsqrt(r2_));
            Z_PAIR(xi2, yi2, zi2, mi2, sx2, sy2, sz2, z_rsqrt(r2_));
            Z_PAIR(xi3, yi3, zi3, mi3, sx3, sy3, sz3, z_rsqrt(r2_));
            Z_STORE_J(j);
        }
        z_add_hsum4(ax + i, sx0, sx1, sx2, sx3);
        z_add_hsum4(ay + i, sy0, sy1, sy2, sy3);
        z_add_hsum4(az + i, sz0, sz1, sz2, sz3);
    }
}

/* diag_avx512()
 * AVX-512 kernel for a diagonal tile (I, I), one target body per pass.
 * For row i it starts at the first vector holding a column > i; in that
 * vector, lanes with j <= i (including j == i) are masked to zero, and every
 * later vector is processed in full. Arguments as diag_generic.
 */
static TARGET_AVX512 void diag_avx512(const real* restrict x, const real* restrict y,
                                      const real* restrict z, const real* restrict m,
                                      real* restrict ax, real* restrict ay, real* restrict az,
                                      size_t i0, size_t p, real eps2)
{
    const zv veps = z_set1(eps2);
    for (size_t li = 0; li < p; li++) {
        const size_t i = i0 + li;
        Z_ROW(0);
        size_t lj = (li + 1) & ~(size_t)(ZW - 1);      /* first vector with a column > li */
        if (lj < p) {
            {   /* that vector straddles the diagonal: mask lanes j <= i (incl. j == i) */
                const zmask keep = (zmask)(~0u << (li + 1 - lj));
                Z_LOAD_J(i0 + lj);
                Z_PAIR(xi0, yi0, zi0, mi0, sx0, sy0, sz0, z_rsqrtz(keep, r2_));
                Z_STORE_J(i0 + lj);
            }
            for (lj += ZW; lj < p; lj += ZW) {
                Z_LOAD_J(i0 + lj);
                Z_PAIR(xi0, yi0, zi0, mi0, sx0, sy0, sz0, z_rsqrt(r2_));
                Z_STORE_J(i0 + lj);
            }
        }
        ax[i] += z_hsum(sx0); ay[i] += z_hsum(sy0); az[i] += z_hsum(sz0);
    }
}

/* ---- AVX2 + FMA --------------------------------------------------------- */

#ifdef USE_FLOAT
typedef __m256 yv;
#define YW       8
#define y_set1   _mm256_set1_ps
#define y_zero   _mm256_setzero_ps
#define y_load   _mm256_load_ps
#define y_store  _mm256_store_ps
#define y_sub    _mm256_sub_ps
#define y_mul    _mm256_mul_ps
#define y_div    _mm256_div_ps
#define y_sqrt   _mm256_sqrt_ps
#define y_and    _mm256_and_ps
#define y_fmadd  _mm256_fmadd_ps
#define y_fnmadd _mm256_fnmadd_ps
#define y_gt(a, b) _mm256_cmp_ps(a, b, _CMP_GT_OQ)
#define y_lanes()  _mm256_setr_ps(0, 1, 2, 3, 4, 5, 6, 7)
#else
typedef __m256d yv;
#define YW       4
#define y_set1   _mm256_set1_pd
#define y_zero   _mm256_setzero_pd
#define y_load   _mm256_load_pd
#define y_store  _mm256_store_pd
#define y_sub    _mm256_sub_pd
#define y_mul    _mm256_mul_pd
#define y_div    _mm256_div_pd
#define y_sqrt   _mm256_sqrt_pd
#define y_and    _mm256_and_pd
#define y_fmadd  _mm256_fmadd_pd
#define y_fnmadd _mm256_fnmadd_pd
#define y_gt(a, b) _mm256_cmp_pd(a, b, _CMP_GT_OQ)
#define y_lanes()  _mm256_setr_pd(0, 1, 2, 3)
#endif

/* y_rinv3_fma()
 * AVX2: returns r^-3 using only multiply/FMA instructions.
 * Starts from the rsqrtps estimate (relative error < 1.5 * 2^-12; for double
 * it goes through float, valid for 1e-38 < r^2 < 3e38) and refines it with
 * the same (1-e)^(-3/2) series as z_rinv3 (2 terms for float, 5 for double).
 */
static inline TARGET_AVX2 yv y_rinv3_fma(yv r2)
{
    const yv one = y_set1(1);
#ifdef USE_FLOAT
    const yv y0 = _mm256_rsqrt_ps(r2);
#else
    const yv y0 = _mm256_cvtps_pd(_mm_rsqrt_ps(_mm256_cvtpd_ps(r2)));
#endif
    const yv y2 = y_mul(y0, y0);
    const yv e  = y_fnmadd(r2, y2, one);
    const yv y3 = y_mul(y2, y0);
#ifdef USE_FLOAT
    yv s = y_fmadd(e, y_set1(15.0f / 8), y_set1(1.5f));
    s    = y_fmadd(e, s, one);
#else
    yv s = y_fmadd(e, y_set1(693.0 / 256), y_set1(315.0 / 128));
    s    = y_fmadd(e, s, y_set1(35.0 / 16));
    s    = y_fmadd(e, s, y_set1(15.0 / 8));
    s    = y_fmadd(e, s, y_set1(1.5));
    s    = y_fmadd(e, s, one);
#endif
    return y_mul(y3, s);
}

/* y_rinv3_div()
 * AVX2: returns r^-3 exactly as 1 / (r2 * sqrt(r2)). It runs on the
 * divider unit, which works in parallel with the FMA units used by
 * y_rinv3_fma; offdiag_avx2 mixes the two to keep both busy.
 */
static inline TARGET_AVX2 yv y_rinv3_div(yv r2)
{
    return y_div(y_set1(1), y_mul(r2, y_sqrt(r2)));
}

/* y_hsum()
 * AVX2: returns the sum of all lanes of v.
 */
static inline TARGET_AVX2 real y_hsum(yv v)
{
#ifdef USE_FLOAT
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_movehdup_ps(s));
    return _mm_cvtss_f32(s);
#else
    __m128d s = _mm_add_pd(_mm256_castpd256_pd128(v), _mm256_extractf128_pd(v, 1));
    s = _mm_add_sd(s, _mm_unpackhi_pd(s, s));
    return _mm_cvtsd_f64(s);
#endif
}

/* W is the r^-3 expression, in terms of r2_ */
#define Y_PAIR(XI, YI, ZI, MI, SX, SY, SZ, W)                                           \
    do {                                                                                \
        const yv dx_ = y_sub(xj, XI), dy_ = y_sub(yj, YI), dz_ = y_sub(zj, ZI);         \
        const yv r2_ = y_fmadd(dz_, dz_, y_fmadd(dy_, dy_, y_fmadd(dx_, dx_, veps)));   \
        const yv w_  = W;                                                               \
        const yv sj_ = y_mul(mj, w_), si_ = y_mul(MI, w_);                              \
        SX  = y_fmadd(sj_, dx_, SX);   SY  = y_fmadd(sj_, dy_, SY);   SZ  = y_fmadd(sj_, dz_, SZ);   \
        axj = y_fnmadd(si_, dx_, axj); ayj = y_fnmadd(si_, dy_, ayj); azj = y_fnmadd(si_, dz_, azj); \
    } while (0)

#define Y_LOAD_J(J)                                                                     \
    const yv xj = y_load(x + (J)), yj = y_load(y + (J)), zj = y_load(z + (J)), mj = y_load(m + (J)); \
    yv axj = y_load(ax + (J)), ayj = y_load(ay + (J)), azj = y_load(az + (J))

#define Y_STORE_J(J) \
    y_store(ax + (J), axj); y_store(ay + (J), ayj); y_store(az + (J), azj)

#define Y_ROW(R)                                                                        \
    const yv xi##R = y_set1(x[i + R]), yi##R = y_set1(y[i + R]),                        \
             zi##R = y_set1(z[i + R]), mi##R = y_set1(m[i + R]);                        \
    yv sx##R = y_zero(), sy##R = y_zero(), sz##R = y_zero()

#define Y_FLUSH(R) \
    ax[i + R] += y_hsum(sx##R); ay[i + R] += y_hsum(sy##R); az[i + R] += y_hsum(sz##R)

/* offdiag_avx2()
 * AVX2 kernel for an off-diagonal tile (I, J), I < J.
 * Each pass takes 2 target bodies and sweeps block J one vector at a time
 * (4 doubles / 8 floats). The first body's r^-3 uses sqrt + divide and the
 * second's uses the FMA estimate, so the divider and the FMA units run at
 * the same time. Arguments as offdiag_generic.
 */
static TARGET_AVX2 void offdiag_avx2(const real* restrict x, const real* restrict y,
                                     const real* restrict z, const real* restrict m,
                                     real* restrict ax, real* restrict ay, real* restrict az,
                                     size_t i0, size_t j0, size_t p, real eps2)
{
    const yv veps = y_set1(eps2);
    for (size_t i = i0; i < i0 + p; i += 2) {          /* 2 target bodies per pass over J */
        Y_ROW(0); Y_ROW(1);
        for (size_t j = j0; j < j0 + p; j += YW) {
            Y_LOAD_J(j);
            Y_PAIR(xi0, yi0, zi0, mi0, sx0, sy0, sz0, y_rinv3_div(r2_));   /* divider */
            Y_PAIR(xi1, yi1, zi1, mi1, sx1, sy1, sz1, y_rinv3_fma(r2_));   /* FMA units */
            Y_STORE_J(j);
        }
        Y_FLUSH(0); Y_FLUSH(1);
    }
}

/* diag_avx2()
 * AVX2 kernel for a diagonal tile (I, I), one target body per pass.
 * The vector straddling the diagonal keeps only lanes with j > i (a compare
 * mask zeroes the rest); later vectors are processed in full.
 * Arguments as diag_generic.
 */
static TARGET_AVX2 void diag_avx2(const real* restrict x, const real* restrict y,
                                  const real* restrict z, const real* restrict m,
                                  real* restrict ax, real* restrict ay, real* restrict az,
                                  size_t i0, size_t p, real eps2)
{
    const yv veps = y_set1(eps2);
    const yv lane = y_lanes();
    for (size_t li = 0; li < p; li++) {
        const size_t i = i0 + li;
        Y_ROW(0);
        size_t lj = (li + 1) & ~(size_t)(YW - 1);      /* first vector with a column > li */
        if (lj < p) {
            {   /* straddles the diagonal: keep lanes with lj + lane > li */
                const yv keep = y_gt(lane, y_set1((real)li - (real)lj));
                Y_LOAD_J(i0 + lj);
                Y_PAIR(xi0, yi0, zi0, mi0, sx0, sy0, sz0, y_and(y_rinv3_div(r2_), keep));
                Y_STORE_J(i0 + lj);
            }
            for (lj += YW; lj < p; lj += YW) {
                Y_LOAD_J(i0 + lj);
                Y_PAIR(xi0, yi0, zi0, mi0, sx0, sy0, sz0, y_rinv3_div(r2_));
                Y_STORE_J(i0 + lj);
            }
        }
        Y_FLUSH(0);
    }
}

#endif /* TILED_X86 */

/* select_kernels()
 * Picks the fastest kernel pair this CPU supports: AVX-512, then AVX2+FMA,
 * then generic. The choice is made at run time, so one build runs anywhere.
 * NBODY_SIMD=avx512|avx2|generic in the environment forces a choice
 * (an unsupported request falls back to the next best).
 */
static Kernels select_kernels(void)
{
    const Kernels generic = { "generic", offdiag_generic, diag_generic };
    const char* want = getenv("NBODY_SIMD");
    if (want && strcmp(want, "generic") == 0) return generic;
#if TILED_X86
    __builtin_cpu_init();
    const bool avx2 = __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
    const bool avx512 = avx2 && __builtin_cpu_supports("avx512f");
    if (avx512 && !(want && strcmp(want, "avx2") == 0))
        return (Kernels){ "avx512", offdiag_avx512, diag_avx512 };
    if (avx2)
        return (Kernels){ "avx2", offdiag_avx2, diag_avx2 };
#endif
    return generic;
}

/* =========================================================================
 *  Workspace
 * ========================================================================= */

/* choose_tile()
 * Returns the tile edge p for n bodies and nthreads threads.
 * NBODY_TILE (read once from the environment) fixes p, rounded up to a
 * multiple of TILE_ALIGN. Otherwise p starts at TILE_SIZE and is halved
 * (not below TILE_MIN) until there are at least TILES_PER_THREAD tiles per
 * thread, so small N still load-balances. p never exceeds n rounded up.
 */
static size_t choose_tile(size_t n, int nthreads)
{
    static long env_tile = -1;
    if (env_tile < 0) {
        const char* env = getenv("NBODY_TILE");
        env_tile = env ? strtol(env, NULL, 10) : 0;
        if (env_tile < 0) env_tile = 0;
    }
    if (env_tile > 0) return round_up((size_t)env_tile, TILE_ALIGN);

    size_t p = round_up(TILE_SIZE, TILE_ALIGN);
    const size_t want = (size_t)nthreads * TILES_PER_THREAD;
    while (p > TILE_MIN) {
        const size_t nb = (n + p - 1) / p;
        if (nb * (nb + 1) / 2 >= want) break;
        p = round_up(p / 2, TILE_ALIGN);
    }
    const size_t cap = round_up(n, TILE_ALIGN);
    return p < cap ? p : cap;
}

/* ws_release()
 * Frees the workspace and resets it, keeping only the selected kernels
 * so the CPU check is not repeated.
 */
static void ws_release(void)
{
    free(ws.raw);
    free(ws.tile_i);
    free(ws.tile_j);
    const Kernels k = ws.k;
    memset(&ws, 0, sizeof ws);
    ws.k = k;
}

/* ws_prepare()
 * Makes sure the workspace fits n bodies and the current thread count,
 * (re)building it only when n, p or the thread count changed. It allocates:
 *   - SoA arrays x, y, z, m (padded to a multiple of p)
 *   - 3 private accumulator arrays per thread, zeroed, each first touched
 *     by its own thread (NUMA-friendly)
 *   - the tile list: off-diagonal tiles row by row, then diagonal tiles
 * Padding bodies get zero mass and are parked far away.
 * Returns the number of threads to use, or 0 if allocation failed.
 */
static int ws_prepare(size_t n)
{
    const int nthreads = omp_get_max_threads();
    const size_t p = choose_tile(n, nthreads);
    if (ws.raw && ws.n == n && ws.p == p && ws.nthreads >= nthreads) return nthreads;

    ws_release();
    if (!ws.k.name) ws.k = select_kernels();

    const size_t npad = round_up(n, p), nb = npad / p;
    const size_t ntiles = nb * (nb + 1) / 2;

    /* One block for x, y, z, m and 3 accumulators per thread. Consecutive
     * arrays are offset by 192 bytes modulo 4 KiB, so the 7 streams a kernel
     * walks through don't all land in the same L1 cache sets. */
    const size_t stride = round_up(npad * sizeof(real), 4096) + 192;
    const size_t bytes = (4 + 3 * (size_t)nthreads) * stride;
    ws.raw    = malloc(bytes + 64);
    ws.tile_i = malloc(ntiles * sizeof *ws.tile_i);
    ws.tile_j = malloc(ntiles * sizeof *ws.tile_j);
    if (!ws.raw || !ws.tile_i || !ws.tile_j) {
        ws_release();
        return 0;
    }
    ws.base     = (char*)(((uintptr_t)ws.raw + 63) & ~(uintptr_t)63);
    ws.stride   = stride;
    ws.n        = n;
    ws.npad     = npad;
    ws.p        = p;
    ws.ntiles   = ntiles;
    ws.nthreads = nthreads;
    ws.x = (real*)(ws.base + 0 * stride);
    ws.y = (real*)(ws.base + 1 * stride);
    ws.z = (real*)(ws.base + 2 * stride);
    ws.m = (real*)(ws.base + 3 * stride);

    /* Off-diagonal tiles first, row by row; the half-cost diagonal tiles
     * last, so the dynamic schedule ends on small work items. */
    size_t t = 0;
    for (size_t I = 0; I < nb; I++)
        for (size_t J = I + 1; J < nb; J++) { ws.tile_i[t] = (uint32_t)I; ws.tile_j[t] = (uint32_t)J; t++; }
    for (size_t I = 0; I < nb; I++) { ws.tile_i[t] = (uint32_t)I; ws.tile_j[t] = (uint32_t)I; t++; }

    /* Padding bodies: zero mass, parked far away at distinct points. */
    for (size_t i = n; i < npad; i++) {
        ws.x[i] = (real)(PAD_POS + 1e15 * (double)(i - n));
        ws.y[i] = ws.z[i] = ws.m[i] = 0;
    }

    /* Accumulators start at zero; each thread first-touches its own. */
#pragma omp parallel num_threads(nthreads)
    for (size_t th = (size_t)omp_get_thread_num(); th < (size_t)nthreads; th += (size_t)omp_get_num_threads())
        memset(acc_of(th, 0), 0, 3 * stride);

    return nthreads;
}

/* =========================================================================
 *  One step (kick = true) or just the accelerations (kick = false)
 * ========================================================================= */

/* tiled_step()
 * Core of the implementation: one OpenMP parallel region with 3 phases.
 *   1. if kick: half-kick v += a dt/2 and drift x += v dt; then copy
 *      positions and G*m into the SoA arrays
 *   2. compute all tiles, schedule(dynamic, 1), into each thread's private
 *      accumulators
 *   3. per chunk of bodies, sum every thread's accumulators (zeroing them
 *      for the next call), store b[i].acc, and if kick do the second
 *      half-kick v += a dt/2
 * With kick = false it only computes accelerations (used at start-up).
 * Exits the program if the workspace can't be allocated.
 */
static void tiled_step(Body* restrict b, size_t n, const Config* cfg, bool kick)
{
    if (n == 0) return;
    const int team = ws_prepare(n);
    if (team == 0) {
        fprintf(stderr, "tiled: out of memory\n");
        exit(EXIT_FAILURE);
    }

    const double  G = cfg->g, dt = cfg->dt, hdt = 0.5 * cfg->dt;
    const real    eps2 = (real)(cfg->softening * cfg->softening);
    const size_t  p = ws.p, npad = ws.npad, ntiles = ws.ntiles;
    const size_t  nchunks = (npad + RED_CHUNK - 1) / RED_CHUNK;
    real* const   x = ws.x, * const y = ws.y, * const z = ws.z, * const m = ws.m;
    const uint32_t* const ti = ws.tile_i, * const tj = ws.tile_j;
    const offdiag_fn offdiag = ws.k.offdiag;
    const diag_fn    diag = ws.k.diag;

#pragma omp parallel num_threads(team)
    {
        const size_t tid = (size_t)omp_get_thread_num();
        const size_t nth = (size_t)omp_get_num_threads();
        real* const ax = acc_of(tid, 0), * const ay = acc_of(tid, 1), * const az = acc_of(tid, 2);

        /* 1. first half-kick + drift, pack SoA */
#pragma omp for schedule(static)
        for (size_t i = 0; i < n; i++) {
            Body* const bi = &b[i];
            if (kick) {
                bi->vel.x += hdt * bi->acc.x; bi->vel.y += hdt * bi->acc.y; bi->vel.z += hdt * bi->acc.z;
                bi->pos.x += dt * bi->vel.x;  bi->pos.y += dt * bi->vel.y;  bi->pos.z += dt * bi->vel.z;
            }
            x[i] = (real)bi->pos.x; y[i] = (real)bi->pos.y; z[i] = (real)bi->pos.z;
            m[i] = (real)(G * bi->mass);
        }

        /* 2. upper-triangle tiles into private accumulators */
#pragma omp for schedule(dynamic, 1)
        for (size_t t = 0; t < ntiles; t++) {
            const size_t I = ti[t], J = tj[t];
            if (I == J) diag(x, y, z, m, ax, ay, az, I * p, p, eps2);
            else        offdiag(x, y, z, m, ax, ay, az, I * p, J * p, p, eps2);
        }

        /* 3. reduce private accumulators (and re-zero them), second half-kick */
#pragma omp for schedule(static)
        for (size_t c = 0; c < nchunks; c++) {
            const size_t lo = c * RED_CHUNK;
            const size_t len = (lo + RED_CHUNK < npad ? lo + RED_CHUNK : npad) - lo;
            double sx[RED_CHUNK] = { 0 }, sy[RED_CHUNK] = { 0 }, sz[RED_CHUNK] = { 0 };
            for (size_t t = 0; t < nth; t++) {
                real* const tx = acc_of(t, 0) + lo, * const ty = acc_of(t, 1) + lo, * const tz = acc_of(t, 2) + lo;
#pragma omp simd
                for (size_t k = 0; k < len; k++) {
                    sx[k] += tx[k]; sy[k] += ty[k]; sz[k] += tz[k];
                    tx[k] = 0;      ty[k] = 0;      tz[k] = 0;
                }
            }
            const size_t hi = lo + len < n ? lo + len : n;
            for (size_t i = lo; i < hi; i++) {
                Body* const bi = &b[i];
                bi->acc.x = sx[i - lo]; bi->acc.y = sy[i - lo]; bi->acc.z = sz[i - lo];
                if (kick) {
                    bi->vel.x += hdt * bi->acc.x; bi->vel.y += hdt * bi->acc.y; bi->vel.z += hdt * bi->acc.z;
                }
            }
        }
    }
}

/* ---- main implementation ------------------------------------------------- */

/* implement_tiled()
 * Driver called from main(). For each thread count 1..cfg.threads:
 * reinitializes the bodies with the same seed, computes initial
 * accelerations and energy (untimed), prints the run header with the
 * kernel and tile size used, then times cfg.num_steps leapfrog steps and
 * stores the total in tiledTimes[threads - 1]. Frees the workspace at the end.
 */
void implement_tiled(Config cfg)
{
    for (size_t thread_t = 1; thread_t <= (size_t)cfg.threads; thread_t++)
    {
        // Set thread count for omp parallelization
        omp_set_num_threads((int)thread_t);

        // Initialize body locations and their forces (also sets up the workspace, untimed)
        srand(cfg.seed);
        init_bodies(bodies, n);
        compute_accelerations_tiled(bodies, n, &cfg);
        double e0 = total_energy_tiled(bodies, n, &cfg);
        printf("\nN=%zu steps=%zu dt=%g softening=%g G=%g seed=%u  E0=%.9e  Threads=%zu  [%s %s, p=%zu]",
            n, cfg.num_steps, cfg.dt, cfg.softening, cfg.g, cfg.seed, e0, thread_t,
            ws.k.name, REAL_NAME, ws.p);

        double elapsed = 0.0;
        // Calculate new energies based on steps
        for (size_t s = 1; s <= cfg.num_steps; s++) {
            double t0 = omp_get_wtime();
            step_leapfrog_tiled(bodies, n, &cfg);
            elapsed += omp_get_wtime() - t0;

            // Optional output to check if results are correct. Compare with other implementations
            /*
            if (s % cfg.report_every == 0 || s == cfg.num_steps) {
                double e = total_energy_tiled(bodies, n, &cfg);
                printf("step %6zu  E=%.9e  dE/E0=%.3e\n", s, e, (e - e0) / fabs(e0));
            }
            */
        }

        tiledTimes[thread_t - 1] = elapsed;
    }

    ws_release();
}


/* ---- force computation ------------------------------------------------- */

/* compute_accelerations_tiled()
 * Fills b[i].acc for all n bodies using the tiled algorithm, without
 * moving any body.
 */
void compute_accelerations_tiled(Body* b, size_t n, const Config* cfg)
{
    tiled_step(b, n, cfg, false);
}

/* ---- integrator -------------------------------------------------------- */

/* step_leapfrog_tiled()
 * Advances all n bodies by one kick-drift-kick leapfrog step of cfg->dt
 * (symplectic, second order), with the force computation in between,
 * all inside one parallel region.
 */
void step_leapfrog_tiled(Body* b, size_t n, const Config* cfg)
{
    tiled_step(b, n, cfg, true);
}

/* ---- diagnostics ------------------------------------------------------- */

/* total_energy_tiled()
 * Returns the total energy, kinetic plus softened potential:
 *   E = sum 1/2 m_i |v_i|^2  -  G sum_{i<j} m_i m_j / sqrt(|r_ij|^2 + eps^2)
 * Always double precision with exact sqrt, whatever USE_FLOAT says, so it
 * can be compared across implementations. Parallel over i with OpenMP.
 */
double total_energy_tiled(const Body* b, size_t n, const Config* cfg)
{
    const double eps2 = cfg->softening * cfg->softening;
    double* const buf = malloc(4 * n * sizeof *buf + 1);
    if (!buf) {
        fprintf(stderr, "tiled: out of memory\n");
        exit(EXIT_FAILURE);
    }
    double* const x = buf, * const y = buf + n, * const z = buf + 2 * n, * const m = buf + 3 * n;
    double ke = 0.0, pe = 0.0;

#pragma omp parallel
    {
#pragma omp for schedule(static) reduction(+ : ke)
        for (size_t i = 0; i < n; i++) {
            x[i] = b[i].pos.x; y[i] = b[i].pos.y; z[i] = b[i].pos.z; m[i] = b[i].mass;
            ke += 0.5 * b[i].mass * (b[i].vel.x * b[i].vel.x + b[i].vel.y * b[i].vel.y + b[i].vel.z * b[i].vel.z);
        }
#pragma omp for schedule(dynamic, 16) reduction(+ : pe)
        for (size_t i = 0; i < n; i++) {
            double s = 0.0;
#pragma omp simd reduction(+ : s)
            for (size_t j = i + 1; j < n; j++) {
                const double dx = x[j] - x[i], dy = y[j] - y[i], dz = z[j] - z[i];
                s += m[j] / sqrt(fabs(dx * dx + dy * dy + dz * dz + eps2));  /* fabs: lets GCC 13+ vectorize */
            }
            pe -= cfg->g * m[i] * s;
        }
    }

    free(buf);
    return ke + pe;
}