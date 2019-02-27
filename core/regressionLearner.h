#pragma once

#include <gsl/gsl_matrix.h>

void workspaceInit(size_t p);
void workspaceDel();

/* kernelTheta is  || betaHat Xtest - Ytest ||^2 = */
/*   (Xtest * (Xlearn^t Xlearn)^(-1) *  Xlearn ^t * Ylearn - Ytest )^2 */
double kernelTheta(const gsl_matrix * data);
double kernelForThetaSquared(const gsl_matrix * data);
