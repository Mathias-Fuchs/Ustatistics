#include <math.h>
#include "U.h"
#include "supervisedLearning.h"
#include <gsl/gsl_rng.h>
#include <assert.h>
#include <gsl/gsl_cdf.h>

// contains the kernel callback functions
#include "regressionLearner.h"


void analyzeDataset(gsl_rng* r, const gsl_matrix* X, const gsl_vector* y, size_t B) {
	assert(X->size1 == y->size);
	size_t n = X->size1;
	size_t p = X->size2;

	gsl_matrix* data = gsl_matrix_alloc(n, p + 1);
	gsl_matrix_view Xv = gsl_matrix_submatrix(data, 0, 0, n, p);
	gsl_vector_view Yv = gsl_matrix_column(data, p);

	gsl_matrix_memcpy(&Xv.matrix, X);
	gsl_vector_memcpy(&Yv.vector, y);

	workspaceInit(3);

	int g = 13;
	fprintf(stdout, "learning set size: %i.\n", (int)g);
	double computationConfIntLower, computationConfIntUpper, thetaConfIntLower, thetaConfIntUpper;
	double estthetasquared;
	double estimatedMean = U(
		data, B, g + 1, r, &kernelTheta, &computationConfIntLower, &computationConfIntUpper, &thetaConfIntLower, &thetaConfIntUpper, &estthetasquared);
	fprintf(stdout, "U-statistic with confidence interval for its exact computation:\n[%f %f %f]\n", computationConfIntLower, estimatedMean, computationConfIntUpper);
	fprintf(stdout, "U-statistic with confidence interval for the population value:\n[%f %f %f]\n", thetaConfIntLower, estimatedMean, thetaConfIntUpper);

	double a0, b0;
	double estimatedVarianceWithZeta1 = U(
		data, B, 2 * g + 1, r, &kernelOverlapOne, &a0, &b0, NULL, NULL, NULL);
	fprintf(stdout, "The variance estimator using zeta was %f .\n", estimatedVarianceWithZeta1)- estthetasquared;
	workspaceDel();
	gsl_matrix_free(data);
}
