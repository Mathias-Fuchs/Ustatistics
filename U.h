#pragma once

#include <gsl/gsl_vector_double.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_rng.h>

void sampleWithoutReplacement(const size_t populationSize, const size_t sampleSize, size_t * subsample, gsl_rng * r);
double U(const gsl_matrix * data, const gsl_vector * response, const size_t B, const int m, gsl_rng * r, double(*kernel)(const gsl_matrix *, const gsl_vector *));

