/*
 * nbody.c - skeletal N-body simulation
 *
 * Direct summation O(N^2) gravity with Plummer softening, integrated using
 * kick-drift-kick leapfrog. Parameters are read from a key = value config file.
 *
 * Usage: ./nbody [config_file]      (default: nbody.cfg, then built-in defaults)
 */

#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DEFAULT_CONFIG_PATH "nbody.cfg"

typedef struct {
    double x, y, z;
} Vec3;

typedef struct {
    Vec3   pos;
    Vec3   vel;
    Vec3   acc;
    double mass;
} Body;

/* Simulation parameters (all settable from the config file). */
typedef struct {
    size_t       num_bodies;
    size_t       num_steps;
    size_t       report_every;
    double       dt;
    double       softening;
    double       g;
    unsigned int seed;
} Config;

static Config config_defaults(void)
{
    return (Config){
        .num_bodies   = 256,
        .num_steps    = 1000,
        .report_every = 100,
        .dt           = 1e-3,
        .softening    = 1e-3,
        .g            = 1.0,
        .seed         = 42,
    };
}

/* ---- config file parsing ----------------------------------------------- */

static char *trim(char *s)
{
    while (isspace((unsigned char)*s)) s++;
    char *end = s + strlen(s);
    while (end > s && isspace((unsigned char)end[-1])) *--end = '\0';
    return s;
}

/*
 * Parse lines of the form `key = value`. '#' starts a comment.
 * Returns 0 on success, -1 if the file can't be opened or a value is bad.
 * Unknown keys produce a warning but are not fatal.
 */
static int load_config(const char *path, Config *cfg)
{
    FILE *f = fopen(path, "r");
    if (!f) return -1;

    char line[256];
    int  lineno = 0, status = 0;

    while (fgets(line, sizeof line, f)) {
        lineno++;

        char *hash = strchr(line, '#');
        if (hash) *hash = '\0';

        char *text = trim(line);
        if (*text == '\0') continue;

        char *eq = strchr(text, '=');
        if (!eq) {
            fprintf(stderr, "%s:%d: expected 'key = value'\n", path, lineno);
            status = -1;
            continue;
        }
        *eq = '\0';
        char *key = trim(text);
        char *val = trim(eq + 1);

        if      (strcmp(key, "num_bodies")   == 0) cfg->num_bodies   = strtoul(val, NULL, 10);
        else if (strcmp(key, "num_steps")    == 0) cfg->num_steps    = strtoul(val, NULL, 10);
        else if (strcmp(key, "report_every") == 0) cfg->report_every = strtoul(val, NULL, 10);
        else if (strcmp(key, "dt")           == 0) cfg->dt           = strtod(val, NULL);
        else if (strcmp(key, "softening")    == 0) cfg->softening    = strtod(val, NULL);
        else if (strcmp(key, "g")            == 0) cfg->g            = strtod(val, NULL);
        else if (strcmp(key, "seed")         == 0) cfg->seed         = (unsigned)strtoul(val, NULL, 10);
        else fprintf(stderr, "%s:%d: unknown key '%s' (ignored)\n", path, lineno, key);
    }

    fclose(f);
    return status;
}

static int validate_config(const Config *cfg)
{
    if (cfg->num_bodies < 2)   { fprintf(stderr, "config: num_bodies must be >= 2\n");   return -1; }
    if (cfg->num_steps == 0)   { fprintf(stderr, "config: num_steps must be > 0\n");     return -1; }
    if (cfg->report_every == 0){ fprintf(stderr, "config: report_every must be > 0\n");  return -1; }
    if (cfg->dt <= 0.0)        { fprintf(stderr, "config: dt must be > 0\n");            return -1; }
    if (cfg->softening < 0.0)  { fprintf(stderr, "config: softening must be >= 0\n");    return -1; }
    return 0;
}

/* ---- vector helpers ---------------------------------------------------- */

static inline Vec3 vec3_add(Vec3 a, Vec3 b)      { return (Vec3){a.x + b.x, a.y + b.y, a.z + b.z}; }
static inline Vec3 vec3_sub(Vec3 a, Vec3 b)      { return (Vec3){a.x - b.x, a.y - b.y, a.z - b.z}; }
static inline Vec3 vec3_scale(Vec3 a, double s)  { return (Vec3){a.x * s, a.y * s, a.z * s}; }
static inline double vec3_dot(Vec3 a, Vec3 b)    { return a.x * b.x + a.y * b.y + a.z * b.z; }

/* ---- initial conditions ------------------------------------------------ */

static double rand_unit(void) { return (double)rand() / RAND_MAX; }

/* Random bodies in a unit sphere, equal mass, total momentum zeroed. */
static void init_bodies(Body *b, size_t n)
{
    Vec3 p_total = {0, 0, 0};

    for (size_t i = 0; i < n; i++) {
        Vec3 p;
        do {
            p = (Vec3){2 * rand_unit() - 1, 2 * rand_unit() - 1, 2 * rand_unit() - 1};
        } while (vec3_dot(p, p) > 1.0);

        b[i].pos  = p;
        b[i].vel  = (Vec3){0.1 * (rand_unit() - 0.5),
                           0.1 * (rand_unit() - 0.5),
                           0.1 * (rand_unit() - 0.5)};
        b[i].acc  = (Vec3){0, 0, 0};
        b[i].mass = 1.0 / (double)n;

        p_total = vec3_add(p_total, vec3_scale(b[i].vel, b[i].mass));
    }

    /* Remove net drift (total mass is 1). */
    for (size_t i = 0; i < n; i++)
        b[i].vel = vec3_sub(b[i].vel, p_total);
}

/* ---- force computation ------------------------------------------------- */

static void compute_accelerations(Body *b, size_t n, const Config *cfg)
{
    const double eps2 = cfg->softening * cfg->softening;

    for (size_t i = 0; i < n; i++)
        b[i].acc = (Vec3){0, 0, 0};

    for (size_t i = 0; i < n; i++) {
        for (size_t j = i + 1; j < n; j++) {
            Vec3   d      = vec3_sub(b[j].pos, b[i].pos);
            double r2     = vec3_dot(d, d) + eps2;
            double inv_r  = 1.0 / sqrt(r2);
            double inv_r3 = inv_r * inv_r * inv_r;

            /* Newton's third law: compute each pair once. */
            b[i].acc = vec3_add(b[i].acc, vec3_scale(d,  cfg->g * b[j].mass * inv_r3));
            b[j].acc = vec3_add(b[j].acc, vec3_scale(d, -cfg->g * b[i].mass * inv_r3));
        }
    }
}

/* ---- integrator -------------------------------------------------------- */

/* Kick-drift-kick leapfrog (symplectic, 2nd order). */
static void step_leapfrog(Body *b, size_t n, const Config *cfg)
{
    const double dt = cfg->dt;

    for (size_t i = 0; i < n; i++) {
        b[i].vel = vec3_add(b[i].vel, vec3_scale(b[i].acc, 0.5 * dt));
        b[i].pos = vec3_add(b[i].pos, vec3_scale(b[i].vel, dt));
    }

    compute_accelerations(b, n, cfg);

    for (size_t i = 0; i < n; i++)
        b[i].vel = vec3_add(b[i].vel, vec3_scale(b[i].acc, 0.5 * dt));
}

/* ---- diagnostics ------------------------------------------------------- */

static double total_energy(const Body *b, size_t n, const Config *cfg)
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

/* TODO: optional output (CSV / binary snapshots) for plotting or rendering. */
static void write_snapshot(const Body *b, size_t n, size_t step)
{
    (void)b; (void)n; (void)step;
}

/* ---- main -------------------------------------------------------------- */

int main(int argc, char **argv)
{
    Config cfg = config_defaults();

    // Check if config was loaded properly
    if (argc > 1) {
        if (load_config(argv[1], &cfg) != 0) {
            fprintf(stderr, "failed to load config '%s'\n", argv[1]);
            return EXIT_FAILURE;
        }
    } else if (load_config(DEFAULT_CONFIG_PATH, &cfg) != 0) {
        fprintf(stderr, "note: no usable '%s'; using built-in defaults\n", DEFAULT_CONFIG_PATH);
    }
    if (validate_config(&cfg) != 0)
        return EXIT_FAILURE;


    // Get number of bodies from config and load into array
    size_t n = cfg.num_bodies;
    Body *bodies = malloc(n * sizeof *bodies);
    if (!bodies) {
        perror("malloc");
        return EXIT_FAILURE;
    }


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
            write_snapshot(bodies, n, s);
        }
    }


    free(bodies);
    return EXIT_SUCCESS;
}
