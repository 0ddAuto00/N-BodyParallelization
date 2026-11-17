/*
* Sequential implementation of the N-body problem
* 
*/


#include "sequential.h"

/* ---- main implementation ------------------------------------------------- */

void implement_sequential(Config cfg)
{
    // Initialize body locations and their forces
    srand(cfg.seed);
    init_bodies(bodies, n);
    compute_accelerations(bodies, n, &cfg);
    double e0 = total_energy(bodies, n, &cfg);
    printf("N=%zu steps=%zu dt=%g softening=%g G=%g seed=%u  E0=%.9e\n",
        n, cfg.num_steps, cfg.dt, cfg.softening, cfg.g, cfg.seed, e0);


    // Calculate new energies based on steps
    for (size_t s = 1; s <= cfg.num_steps; s++) {
        step_leapfrog(bodies, n, &cfg);

        if (s % cfg.report_every == 0 || s == cfg.num_steps) {
            double e = total_energy(bodies, n, &cfg);
            printf("step %6zu  E=%.9e  dE/E0=%.3e\n", s, e, (e - e0) / fabs(e0));
        }
    }

    printf("\n");
}


/* ---- force computation ------------------------------------------------- */

void compute_accelerations(Body* b, size_t n, const Config* cfg)
{
    const double eps2 = cfg->softening * cfg->softening;

    for (size_t i = 0; i < n; i++)
        b[i].acc = (Vec3){ 0, 0, 0 };

    for (size_t i = 0; i < n; i++) {
        for (size_t j = i + 1; j < n; j++) {
            Vec3   d = vec3_sub(b[j].pos, b[i].pos);
            double r2 = vec3_dot(d, d) + eps2;
            double inv_r = 1.0 / sqrt(r2);
            double inv_r3 = inv_r * inv_r * inv_r;

            /* Newton's third law: compute each pair once. */
            b[i].acc = vec3_add(b[i].acc, vec3_scale(d, cfg->g * b[j].mass * inv_r3));
            b[j].acc = vec3_add(b[j].acc, vec3_scale(d, -cfg->g * b[i].mass * inv_r3));
        }
    }
}

/* ---- integrator -------------------------------------------------------- */

/* Kick-drift-kick leapfrog (symplectic, 2nd order). */
void step_leapfrog(Body* b, size_t n, const Config* cfg)
{
    const double dt = cfg->dt;

    // add new velocities per 
    for (size_t i = 0; i < n; i++) {
        b[i].vel = vec3_add(b[i].vel, vec3_scale(b[i].acc, 0.5 * dt));
        b[i].pos = vec3_add(b[i].pos, vec3_scale(b[i].vel, dt));
    }

    compute_accelerations(b, n, cfg);

    for (size_t i = 0; i < n; i++)
        b[i].vel = vec3_add(b[i].vel, vec3_scale(b[i].acc, 0.5 * dt));
}

/* ---- diagnostics ------------------------------------------------------- */

double total_energy(const Body* b, size_t n, const Config* cfg)
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