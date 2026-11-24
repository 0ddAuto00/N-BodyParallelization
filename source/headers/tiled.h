#ifndef TILED_H
#define TILED_H

#include "nbody.h"

extern size_t n;
extern Body* bodies;
extern double* seqTimes, * rowTimes, * tiledTimes;

void compute_accelerations_tiled(Body* b, size_t n, const Config* cfg);

void step_leapfrog_tiled(Body* b, size_t n, const Config* cfg);

double total_energy_tiled(const Body* b, size_t n, const Config* cfg);

void implement_tiled(Config cfg);

#endif