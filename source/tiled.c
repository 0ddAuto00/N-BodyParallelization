/* Implemented by Graceson and Larry
* 
*  All-tiled pairs implementation
* 
*/

#include "headers/tiled.h"

#ifndef TILE_P
#define TILE_P 64   /* p: rows per block and columns per tile. Tune: 32..256 */
#endif

/* `omp simd` needs OpenMP 4.0+. MSVC only has it with /openmp:experimental. */
#if defined(_OPENMP) && (_OPENMP >= 201307 || defined(_OPENMP_SIMD))
#define NB_OMP_SIMD _Pragma("omp simd")
#else
#define NB_OMP_SIMD
#endif

static inline size_t nb_min(size_t a, size_t b) { return a < b ? a : b; }

/* ---- packed positions (the float4 {x,y,z,m} array from the paper) ------ */

static double* g_pos = NULL;   /* 4 doubles per body: x, y, z, m */
static size_t  g_cap = 0;



/* ---- main implementation ------------------------------------------------- */

void implement_tiled(Config cfg)
{
    for (size_t thread_t = 1; thread_t <= (size_t)cfg.threads; thread_t++)
    {
        double elapsed = 0.0;

        // Set thread count for omp parallelization
        if (cfg.threads > 0) omp_set_num_threads(thread_t);
        for (int run_number = 0; run_number < cfg.avg_over; run_number++)
        {
            // Initialize body locations and their forces
            srand(cfg.seed);
            init_bodies(bodies, n);
            compute_accelerations_tiled(bodies, n, &cfg);
            double e0 = total_energy_tiled(bodies, n, &cfg);
            printf("\nN=%zu steps=%zu dt=%g softening=%g G=%g seed=%u  E0=%.9e  Threads=%lld",
                n, cfg.num_steps, cfg.dt, cfg.softening, cfg.g, cfg.seed, e0, thread_t);


            // Calculate new energies based on steps
            for (size_t s = 1; s <= cfg.num_steps; s++) {
                double t0 = omp_get_wtime();
                step_leapfrog_tiled(bodies, n, &cfg);
                elapsed += omp_get_wtime() - t0;

                // Optional output to check if results are correct. Compare with other implementations
                /*
                if (s % cfg.report_every == 0 || s == cfg.num_steps) {
                    double e = total_energy_rowwise(bodies, n, &cfg);
                    printf("step %6zu  E=%.9e  dE/E0=%.3e\n", s, e, (e - e0) / fabs(e0));
                }
                */
            }
            printf("\n\tRun number %d: Total time elapsed: %f\n", run_number, elapsed);
        }

        tiledTimes[thread_t - 1] = elapsed / (double)cfg.avg_over;
    }

}


/* ---- force computation ------------------------------------------------- */

static void pack_positions_tiled(const Body* b, size_t n)
{
    if (n > g_cap) {
        free(g_pos);
        g_pos = (double*)malloc(n * 4 * sizeof(double));
        g_cap = n;
    }
    double* pos = g_pos;

#pragma omp parallel for schedule(static)
    for (long long i = 0; i < (long long)n; i++) {
        pos[4 * i + 0] = b[i].pos.x;
        pos[4 * i + 1] = b[i].pos.y;
        pos[4 * i + 2] = b[i].pos.z;
        pos[4 * i + 3] = b[i].mass;
    }
}

/* ---- force computation ------------------------------------------------- */

/* Accumulate the effect of one tile (bj bodies starting at `tile`) on the
 * bi bodies of the current block. This is tile_calculation() +
 * bodyBodyInteraction() from the paper. Each body j is broadcast across all
 * rows (i), exactly like the shared-memory broadcast on the GPU. */
static inline void tile_calculation_tiled(
    const double* tile, size_t bj,
    const double* xi, const double* yi, const double* zi,
    double* ax, double* ay, double* az,
    size_t bi, double eps2)
{
    for (size_t jj = 0; jj < bj; jj++) {
        const double xj = tile[4 * jj + 0];
        const double yj = tile[4 * jj + 1];
        const double zj = tile[4 * jj + 2];
        const double mj = tile[4 * jj + 3];

        NB_OMP_SIMD
            for (size_t ii = 0; ii < bi; ii++) {
                const double dx = xj - xi[ii];
                const double dy = yj - yi[ii];
                const double dz = zj - zi[ii];
                const double d2 = dx * dx + dy * dy + dz * dz + eps2;
                const double inv = 1.0 / sqrt(d2);
                const double s = mj * inv * inv * inv;   /* m_j / d^3 */
                ax[ii] += dx * s;
                ay[ii] += dy * s;
                az[ii] += dz * s;
            }
    }
}

void compute_accelerations_tiled(Body* b, size_t n, const Config* cfg)
{
    if (n == 0) return;

    pack_positions_tiled(b, n);

    const double* pos = g_pos;
    const double  eps2 = cfg->softening * cfg->softening;
    const double  G = cfg->g;
    const long long nblocks = (long long)((n + TILE_P - 1) / TILE_P);

    /* "Grid of thread blocks": each i-block is independent. */
#pragma omp parallel for schedule(static)
    for (long long ib = 0; ib < nblocks; ib++) {
        const size_t i0 = (size_t)ib * TILE_P;
        const size_t bi = nb_min(TILE_P, n - i0);

        double xi[TILE_P], yi[TILE_P], zi[TILE_P];
        double ax[TILE_P], ay[TILE_P], az[TILE_P];

        for (size_t ii = 0; ii < bi; ii++) {
            xi[ii] = pos[4 * (i0 + ii) + 0];
            yi[ii] = pos[4 * (i0 + ii) + 1];
            zi[ii] = pos[4 * (i0 + ii) + 2];
            ax[ii] = ay[ii] = az[ii] = 0.0;
        }

        /* Loop over the N/p tiles (columns of the force grid). */
        for (size_t j0 = 0; j0 < n; j0 += TILE_P) {
            const size_t bj = nb_min(TILE_P, n - j0);
            tile_calculation_tiled(pos + 4 * j0, bj, xi, yi, zi,
                ax, ay, az, bi, eps2);
        }

        for (size_t ii = 0; ii < bi; ii++) {
            b[i0 + ii].acc.x = G * ax[ii];
            b[i0 + ii].acc.y = G * ay[ii];
            b[i0 + ii].acc.z = G * az[ii];
        }
    }
}

/* ---- integrator -------------------------------------------------------- */

/* Kick-drift-kick leapfrog. Assumes b[i].a* holds a(t) on entry (your driver
 * calls compute_accelerations_tiled once before the first step), and leaves
 * a(t+dt) in place on exit. */
void step_leapfrog_tiled(Body* b, size_t n, const Config* cfg)
{
    const double dt = cfg->dt;
    const double hdt = 0.5 * dt;
    const long long N = (long long)n;

#pragma omp parallel for schedule(static)
    for (long long i = 0; i < N; i++) {
        b[i].vel.x += hdt * b[i].acc.x;
        b[i].vel.y += hdt * b[i].acc.y;
        b[i].vel.z += hdt * b[i].acc.z;
        b[i].pos.x += dt * b[i].vel.x;
        b[i].pos.y += dt * b[i].vel.y;
        b[i].pos.z += dt * b[i].vel.z;
    }

    compute_accelerations_tiled(b, n, cfg);

#pragma omp parallel for schedule(static)
    for (long long i = 0; i < N; i++) {
        b[i].vel.x += hdt * b[i].acc.x;
        b[i].vel.y += hdt * b[i].acc.y;
        b[i].vel.z += hdt * b[i].acc.z;
    }
}

/* ---- diagnostics ------------------------------------------------------- */

/* KE + softened Plummer PE. The PE sum is tiled the same way as the forces.
 * The full N x N sum includes i==j terms (m_i^2 / eps), which are removed
 * analytically instead of branching in the inner loop. */
double total_energy_tiled(const Body* b, size_t n, const Config* cfg)
{
    if (n == 0) return 0.0;

    const long long N = (long long)n;
    const double eps2 = cfg->softening * cfg->softening;
    const double eps = cfg->softening;
    const double G = cfg->g;

    double ke = 0.0;
#pragma omp parallel for schedule(static) reduction(+:ke)
    for (long long i = 0; i < N; i++) {
        ke += 0.5 * b[i].mass * (b[i].vel.x * b[i].vel.x +
            b[i].vel.y * b[i].vel.y +
            b[i].vel.z * b[i].vel.z);
    }

    pack_positions_tiled(b, n);
    const double* pos = g_pos;
    const long long nblocks = (long long)((n + TILE_P - 1) / TILE_P);

    double full = 0.0;   /* sum over all (i, j) of m_i m_j / sqrt(r^2 + eps^2) */
    double self = 0.0;   /* sum over i of m_i^2 / eps                          */

#pragma omp parallel for schedule(static) reduction(+:full,self)
    for (long long ib = 0; ib < nblocks; ib++) {
        const size_t i0 = (size_t)ib * TILE_P;
        const size_t bi = nb_min(TILE_P, n - i0);

        double xi[TILE_P], yi[TILE_P], zi[TILE_P], pot[TILE_P];
        for (size_t ii = 0; ii < bi; ii++) {
            xi[ii] = pos[4 * (i0 + ii) + 0];
            yi[ii] = pos[4 * (i0 + ii) + 1];
            zi[ii] = pos[4 * (i0 + ii) + 2];
            pot[ii] = 0.0;
        }

        for (size_t j0 = 0; j0 < n; j0 += TILE_P) {
            const size_t bj = nb_min(TILE_P, n - j0);
            const double* tile = pos + 4 * j0;

            for (size_t jj = 0; jj < bj; jj++) {
                const double xj = tile[4 * jj + 0];
                const double yj = tile[4 * jj + 1];
                const double zj = tile[4 * jj + 2];
                const double mj = tile[4 * jj + 3];

                NB_OMP_SIMD
                    for (size_t ii = 0; ii < bi; ii++) {
                        const double dx = xj - xi[ii];
                        const double dy = yj - yi[ii];
                        const double dz = zj - zi[ii];
                        pot[ii] += mj / sqrt(dx * dx + dy * dy + dz * dz + eps2);
                    }
            }
        }

        for (size_t ii = 0; ii < bi; ii++) {
            const double mi = pos[4 * (i0 + ii) + 3];
            full += mi * pot[ii];
            self += mi * mi / eps;
        }
    }

    const double pe = -0.5 * G * (full - self);
    return ke + pe;
}