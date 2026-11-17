/* Implemented by Sakina and Brady
 * 
 * rowwise.c - N-body simulation summing pairwise forces accross rows of matricies in parallel
 * 
 * Based on the sequential.c implementation which is a direct summation O(N^2) gravity
 * with Plummer softening, integrated using kick-drift-kick leapfrog.
 * Parameters are read from a key = value config file.
 * 
 * Inital Conditions:
 *  Each body gets a small random velocity (components in ±0.05), 0 initial acceleration,
    and mass 1/n, so total mass is always 1
 */

#include "rowwise.h"

/* ---- main implementation ------------------------------------------------- */

void implement_rowwise(Config cfg)
{
    if (cfg.threads > 0) omp_set_num_threads(cfg.threads);

    // Initialize body locations and their forces
    srand(cfg.seed);
    init_bodies(bodies, n);
    compute_accelerations_rowwise(bodies, n, &cfg);
    double e0 = total_energy_rowwise(bodies, n, &cfg);
    printf("N=%zu steps=%zu dt=%g softening=%g G=%g seed=%u  E0=%.9e\n",
        n, cfg.num_steps, cfg.dt, cfg.softening, cfg.g, cfg.seed, e0);


    // Calculate new energies based on steps
    for (size_t s = 1; s <= cfg.num_steps; s++) {
        step_leapfrog_rowwise(bodies, n, &cfg);

        if (s % cfg.report_every == 0 || s == cfg.num_steps) {
            double e = total_energy_rowwise(bodies, n, &cfg);
            printf("step %6zu  E=%.9e  dE/E0=%.3e\n", s, e, (e - e0) / fabs(e0));
        }
    }

    printf("\n");
}


/* ---- force computation ------------------------------------------------- */
// TODO: Parallelize here
void compute_accelerations_rowwise(Body *b, size_t n, const Config *cfg)
{
    const double eps2 = cfg->softening * cfg->softening;
    size_t i = 0;
    size_t k = 0;
    // Initialize array of n bodies's pairwise interactions and their accelerations
# pragma omp parallel for shared(b) private(k)
    for (k = 0; k < n; k++)
        b[k].acc = (Vec3){0, 0, 0};

//each thread gets a row, j is defined within the inner loop
# pragma omp parallel for shared(b, n, cfg) private(i) schedule(static)
    for (i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) { //No longer start at diagonal
            if(j == i) continue;
            Vec3   d      = vec3_sub(b[j].pos, b[i].pos);
            double r2     = vec3_dot(d, d) + eps2;
            double inv_r  = 1.0 / sqrt(r2);
            double inv_r3 = inv_r * inv_r * inv_r;

            // Compute all pairs individually, parallel implementation will
            // overlap in accessing the j index -> not thread safe
            b[i].acc = vec3_add(b[i].acc, vec3_scale(d,  cfg->g * b[j].mass * inv_r3));
            //b[j].acc = vec3_add(b[j].acc, vec3_scale(d, -cfg->g * b[i].mass * inv_r3));
        }
    }
}

/* ---- integrator -------------------------------------------------------- */

/* Kick-drift-kick leapfrog (symplectic, 2nd order). */
void step_leapfrog_rowwise(Body *b, size_t n, const Config *cfg)
{
    const double dt = cfg->dt;

    for (size_t i = 0; i < n; i++) {
        b[i].vel = vec3_add(b[i].vel, vec3_scale(b[i].acc, 0.5 * dt));
        b[i].pos = vec3_add(b[i].pos, vec3_scale(b[i].vel, dt));
    }

    compute_accelerations_rowwise(b, n, cfg);

    for (size_t i = 0; i < n; i++)
        b[i].vel = vec3_add(b[i].vel, vec3_scale(b[i].acc, 0.5 * dt));
}

/* ---- diagnostics ------------------------------------------------------- */

double total_energy_rowwise(const Body *b, size_t n, const Config *cfg)
{
    const double eps2 = cfg->softening * cfg->softening;
    double ke = 0.0, pe = 0.0;

    for (size_t i = 0; i < n; i++) {
        ke += 0.5 * b[i].mass * vec3_dot(b[i].vel, b[i].vel);
        for (size_t j = i + 1; j < n; j++) {
            Vec3 d = vec3_sub(b[j].pos, b[i].pos);
            pe -= cfg->g * b[i].mass * b[j].mass / sqrt(vec3_dot(d, d) + eps2);
        }
    }
    return ke + pe;
}
