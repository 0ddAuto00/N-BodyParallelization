#ifndef SEQUENTIAL_H
#define SEQUENTIAL_H

#include "nbody.h"

extern size_t n;
extern Body* bodies;
extern double start, end;
extern double* seqTimes, * rowTimes, * tiledTimes;

void compute_accelerations(Body* b, size_t n, const Config* cfg);

void step_leapfrog(Body* b, size_t n, const Config* cfg);

double total_energy(const Body* b, size_t n, const Config* cfg);

void implement_sequential(Config cfg);

#endif