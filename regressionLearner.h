#pragma once

#include <gsl/gsl_vector_double.h>
#include <gsl/gsl_matrix.h>

void workspaceInit(size_t p);
void workspaceDel();



/* Gamma is  || betaHat Xtest - Ytest ||^2 = */
/*   (Xtest * (Xlearn^t Xlearn)^(-1) *  Xlearn ^t * Ylearn - Ytest )^2 */


double gamma(const gsl_matrix * data, const gsl_vector * response);
double kernelForThetaSquared(const gsl_matrix * data, const gsl_vector * response);
