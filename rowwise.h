#ifndef ROWWISE_H
#define ROWWISE_H

#include "nbody.h"

extern size_t n;
extern Body* bodies;

void compute_accelerations_rowwise(Body* b, size_t n, const Config* cfg);

void step_leapfrog_rowwise(Body* b, size_t n, const Config* cfg);

double total_energy_rowwise(const Body* b, size_t n, const Config* cfg);

void implement_rowwise(Config cfg);

#endif