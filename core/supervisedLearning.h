#pragma once

#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector_double.h>


void analyzeDataset(const gsl_matrix* X, const gsl_vector* y, size_t B);