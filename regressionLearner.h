#pragma once

#include <gsl/gsl_vector_double.h>
#include <gsl/gsl_matrix.h>

void workspaceInit(size_t p);
void workspaceDel();
double gamma(const gsl_matrix * data, const gsl_vector * response);
double kernelForThetaSquared(const gsl_matrix * data, const gsl_vector * response);
