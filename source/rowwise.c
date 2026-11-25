/* Implemented by Sakina and Brady
 * 
 * rowwise.c - N-body simulation summing pairwise forces accross rows of matricies in parallel
 * 
 * Based on the sequential.c implementation which is a direct summation O(N^2) gravity
 * with Plummer softening, integrated using kick-drift-kick leapfrog.
 * Parameters are read from a key = value config file.
 * 
 * Inital Conditions:
 *  Each body gets a small random velocity (components in + - 0.05), 0 initial acceleration,
    and mass 1/n, so total mass is always 1
 */

#include "headers/rowwise.h"

/* ---- main implementation ------------------------------------------------- */

void implement_rowwise(Config cfg)
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
            compute_accelerations_rowwise(bodies, n, &cfg);
            double e0 = total_energy_rowwise(bodies, n, &cfg);
            printf("\nN=%zu steps=%zu dt=%g softening=%g G=%g seed=%u  E0=%.9e  Threads=%zu",
                n, cfg.num_steps, cfg.dt, cfg.softening, cfg.g, cfg.seed, e0, thread_t);


            // Calculate new energies based on steps
            for (size_t s = 1; s <= cfg.num_steps; s++) {
                double t0 = omp_get_wtime();
                step_leapfrog_rowwise(bodies, n, &cfg);
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
            
        rowTimes[thread_t - 1] = elapsed/(double) cfg.avg_over;
    }

}


/* ---- force computation ------------------------------------------------- */
// Changed to compute x y z position and include initialization in parallel
// better mirrors that in tiled.c. Still calculates N^2 version (not just top)
void compute_accelerations_rowwise(Body *b, size_t n, const Config *cfg)
{
    const double eps2 = cfg->softening * cfg->softening;
    const double G = cfg->g;

    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < (long long)n; i++) {
        const Vec3 pi = b[i].pos;
        double ax = 0.0, ay = 0.0, az = 0.0;

        for (size_t j = 0; j < n; j++) {
            if (j == (size_t)i) continue;
            const double dx = b[j].pos.x - pi.x;
            const double dy = b[j].pos.y - pi.y;
            const double dz = b[j].pos.z - pi.z;
            const double inv = 1.0 / sqrt(dx*dx + dy*dy + dz*dz + eps2);
            const double s = b[j].mass * inv * inv * inv;
            ax += dx * s;  ay += dy * s;  az += dz * s;
        }
        b[i].acc = (Vec3){ G*ax, G*ay, G*az };
    }
}

/* ---- integrator -------------------------------------------------------- */

/* Kick-drift-kick leapfrog (symplectic, 2nd order). */
void step_leapfrog_rowwise(Body *b, size_t n, const Config *cfg)
{
    const double dt = cfg->dt;

# pragma omp parallel for shared(b, dt) schedule(static)
    for (long long i = 0; i < n; i++) {
        b[i].vel = vec3_add(b[i].vel, vec3_scale(b[i].acc, 0.5 * dt));
        b[i].pos = vec3_add(b[i].pos, vec3_scale(b[i].vel, dt));
    }

    compute_accelerations_rowwise(b, n, cfg);

# pragma omp parallel for shared(b, dt) schedule(static)
    for (long long i = 0; i < n; i++)
        b[i].vel = vec3_add(b[i].vel, vec3_scale(b[i].acc, 0.5 * dt));
}

/* ---- diagnostics ------------------------------------------------------- */

double total_energy_rowwise(const Body *b, size_t n, const Config *cfg)
{
    const double eps2 = cfg->softening * cfg->softening;
    double ke = 0.0, pe = 0.0;

# pragma omp parallel for schedule(static) reduction(+:ke) reduction(-:pe) 
    for (long long i = 0; i < n; i++) {
        ke += 0.5 * b[i].mass * vec3_dot(b[i].vel, b[i].vel);
        for (long long j = i + 1; j < n; j++) {
            Vec3 d = vec3_sub(b[j].pos, b[i].pos);
            pe -= cfg->g * b[i].mass * b[j].mass / sqrt(vec3_dot(d, d) + eps2);
        }
    }
    return ke + pe;
}
