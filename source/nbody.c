/*
 * nbody.c - skeletal N-body simulation
 *
 * Direct summation O(N^2) gravity with Plummer softening, integrated using
 * kick-drift-kick leapfrog. Parameters are read from a key = value config file.
 *
 * Usage: ./nbody [config_file]      (default: nbody.cfg, then built-in defaults)
 */

#include "headers/nbody.h"
#include "headers/sequential.h"
#include "headers/rowwise.h"
#include "headers/tiled.h"

#define PLOT_SCRIPT "source/plot_results.py"
#define CSV_PATH    "docs/prelim_report/results.csv"

Body* bodies;
size_t n;
double start, end;
double *seqTimes, *rowTimes, *tiledTimes;

/* ---- main -------------------------------------------------------------- */

int main(int argc, char** argv)
{
    Config cfg = config_defaults();

    if (!check_config(argc, argv, &cfg)) { return EXIT_FAILURE; }
    if (!create_bodies_arr(&cfg))        { return EXIT_FAILURE; }
    if (!init_times_arr_sizes(cfg))      { return EXIT_FAILURE; }


    implement_sequential(cfg);
    time_elapsed(SEQ, cfg);
    
    implement_rowwise(cfg);
    time_elapsed(ROW, cfg);        

    
    implement_tiled(cfg);
    time_elapsed(TILED, cfg);


    write_to_csv(cfg);
    plot_csv();

    free(bodies);
    return EXIT_SUCCESS;
}


/* ---- functions ----------------------------------------------------------- */

Config config_defaults(void)
{
    return (Config){
        .num_bodies   = 256,
        .num_steps    = 1000,
        .report_every = 100,
        .dt           = 1e-3,
        .softening    = 1e-3,
        .g            = 1.0,
        .seed         = 42,
        .threads      = 1,
        .avg_over     = 5,
    };
}

char *trim(char *s)
{
    while (isspace((unsigned char)*s)) s++;
    char *end = s + strlen(s);
    while (end > s && isspace((unsigned char)end[-1])) *--end = '\0';
    return s;
}

int load_config(const char *path, Config *cfg)
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
        else if (strcmp(key, "threads")      == 0) cfg->threads      = strtod(val, NULL);
        else fprintf(stderr, "%s:%d: unknown key '%s' (ignored)\n", path, lineno, key);
    }

    fclose(f);
    return status;
}

int validate_config(const Config *cfg)
{
    if (cfg->num_bodies < 2)   { fprintf(stderr, "config: num_bodies must be >= 2\n");   return -1; }
    if (cfg->num_steps == 0)   { fprintf(stderr, "config: num_steps must be > 0\n");     return -1; }
    if (cfg->report_every == 0){ fprintf(stderr, "config: report_every must be > 0\n");  return -1; }
    if (cfg->dt <= 0.0)        { fprintf(stderr, "config: dt must be > 0\n");            return -1; }
    if (cfg->softening < 0.0)  { fprintf(stderr, "config: softening must be >= 0\n");    return -1; }
    if (cfg->threads <= 0)     { fprintf(stderr, "config: threads must be > 0\n");       return -1; }
    return 0;
}

bool check_config(int argc, char** argv, Config *cfg)
{
    if (argc > 1) {
        if (load_config(argv[1], cfg) != 0) {
            fprintf(stderr, "failed to load config '%s'\n", argv[1]);
            return false;
        }
    }
    else if (load_config(DEFAULT_CONFIG_PATH, cfg) != 0) {
        fprintf(stderr, "note: no usable '%s'; using built-in defaults\n", DEFAULT_CONFIG_PATH);
    }
    if (validate_config(cfg) != 0)
        return false;

    return true;
}


/* Random bodies in a unit sphere, equal mass, total momentum zeroed. */
void init_bodies(Body *b, size_t n)
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

bool create_bodies_arr(Config* cfg)
{
    n = cfg->num_bodies;
    bodies = malloc(n * sizeof * bodies);
    if (!bodies) {
        perror("malloc");
        return false;
    }

    return true;
}

bool init_times_arr_sizes(Config cfg)
{
    if (cfg.threads <= 0) return false;
    
    seqTimes   = malloc(cfg.threads * sizeof(double));
    rowTimes   = malloc(cfg.threads * sizeof(double));
    tiledTimes = malloc(cfg.threads * sizeof(double));


    return true;
}

void time_elapsed(Implementation currentImp, Config cfg)
{
    const char* impStr;
    switch (currentImp)
    {
    case SEQ:
        impStr = "Sequential";
        break;
    case ROW:
        impStr = "Row-wise";
        break;
    case TILED:
        impStr = "All-tiled Pairs";
        break;
    default:
        impStr = "Unknown";
    }

    printf("\n");
    for (size_t i = 0; i < cfg.threads; i++)
    {
        switch (currentImp)
        {
        case SEQ:
            printf("\nCPU time used for %s with %zu threads: %f seconds", impStr, i + 1, seqTimes[i]);
            break;
        case ROW:
            printf("\nCPU time used for %s with %zu threads: %f seconds", impStr, i + 1, rowTimes[i]);
            break;
        case TILED:
            printf("\nCPU time used for %s with %zu threads: %f seconds", impStr, i + 1, tiledTimes[i]);
            break;
        default:
            impStr = "Unknown";
        }
    }
    printf("\n");
    
}

void write_to_csv(Config cfg)
{
    const char* path = "docs/prelim_report/results.csv";

    FILE* fp = fopen(path, "w");
    if (!fp) {
        perror("write_to_csv: could not open results.csv");
        return;
    }

    fprintf(fp, "threads,sequential,rowwise,tiled\n");

    for (int t = 1; t <= cfg.threads; t++) {
        fprintf(fp, "%d,%.9f,%.9f,%.9f\n",
            t,
            seqTimes[t - 1],
            rowTimes[t - 1],
            tiledTimes[t - 1]);
    }

    fclose(fp);
    printf("Results written to %s\n", path);
}

void plot_csv()
{
    char cmd[512];
    snprintf(cmd, sizeof cmd, "python3 \"%s\" \"%s\"", PLOT_SCRIPT, CSV_PATH);

    int rc = system(cmd);
    if (rc != 0) {
        fprintf(stderr, "plot_csv: command failed (exit status %d): %s\n", rc, cmd);
    }
}