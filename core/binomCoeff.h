#pragma once
#include <stddef.h>

// struct to keep track of whether an unsigned int is beyond the bounds
typedef struct {
	unsigned int i;
	int isInfty;
} rr;

// binomialCoefficient unless second entry indicates it's too big
rr binomialCoefficient(size_t n, size_t k);
