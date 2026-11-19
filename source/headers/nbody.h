#ifndef NBODY_H
#define NBODY_H

#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <omp.h>
#include <time.h>
#include <unistd.h>

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

typedef enum {
    SEQ,
    ROW,
    TILED
} Implementation;

extern size_t n;
extern Body* bodies;

/* Simulation parameters (all settable from the config file). */
typedef struct {
    size_t       num_bodies;
    size_t       num_steps;
    size_t       report_every;
    double       dt;
    double       softening;
    double       g;
    unsigned int seed;
    int          threads;
    int          avg_over;
} Config;

// Function to set variable defaults incase config file not found
Config config_defaults(void);

// Function to parse config file
char* trim(char* s);

// Function to load a given config file
int load_config(const char* path, Config* cfg);

// Function to validate values in a config file
int validate_config(const Config* cfg);

// Function to initialize bodies in a random unit sphere
void init_bodies(Body* b, size_t n);

// Function to get number of bodies and create an array of bodies
bool create_bodies_arr(Config* cfg);

// Function to check if config/default values were loaded successfully
bool check_config(int argc, char** argv, Config *cfg);

// Function to initialize number of threads and array sizes for each implementation to store time taken for each thread number
bool init_times_arr_sizes(Config cfg);

// Function to print elapsed time
void time_elapsed(Implementation currentImp);

// Function to output time array results to csv
void write_to_csv(Config cfg);

// Function to run a python script that plots csv file results
void plot_csv();

/* ---- vector helpers ---------------------------------------------------- */
inline Vec3 vec3_add(Vec3 a, Vec3 b) { return (Vec3) { a.x + b.x, a.y + b.y, a.z + b.z }; }
inline Vec3 vec3_sub(Vec3 a, Vec3 b) { return (Vec3) { a.x - b.x, a.y - b.y, a.z - b.z }; }
inline Vec3 vec3_scale(Vec3 a, double s) { return (Vec3) { a.x* s, a.y* s, a.z* s }; }
inline double vec3_dot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

/* ---- initial conditions ------------------------------------------------ */
static inline double rand_unit(void) { return (double)rand() / RAND_MAX; }

#endif