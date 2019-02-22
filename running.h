#pragma once

#include <gsl/gsl_vector_double.h>

double runningMean(const gsl_vector * x, const size_t until);