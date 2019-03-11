#include <math.h>
#include "U.h"
#include "supervisedLearning.h"
#include <gsl/gsl_rng.h>
#include <assert.h>
#include <gsl/gsl_cdf.h>

// contains the kernel callback functions
#include "regressionLearner.h"


void analyzeDataset(const gsl_matrix* X, const gsl_vector* y, size_t B) {
	assert(X->size1 == y->size);
	size_t n = X->size1;
	size_t p = X->size2;

	gsl_matrix* data = gsl_matrix_alloc(n, p + 1);
	gsl_matrix_view Xv = gsl_matrix_submatrix(data, 0, 0, n, p);
	gsl_vector_view Yv = gsl_matrix_column(data, p);

	gsl_matrix_memcpy(&Xv.matrix, X);
	gsl_vector_memcpy(&Yv.vector, y);

	int seed = 1234;
	gsl_rng * r = gsl_rng_alloc(gsl_rng_taus2);
	gsl_rng_set(r, seed);
	workspaceInit(3);

	for (size_t g = (n - 2) / 4; g < (n - 2) / 2; g++) {
		fprintf(stdout, "learning set size: %i.\n", (int) g);
		double computationConfIntLower, computationConfIntUpper, thetaConfIntLower, thetaConfIntUpper;
		double estimatedMean = U(
			data, B, 1, r, &kernelTheta, &computationConfIntLower, &computationConfIntUpper, &thetaConfIntLower, &thetaConfIntUpper);
		printf("U-statistic with confidence interval for its exact computation:\n[%f %f %f]\n", computationConfIntLower, estimatedMean, computationConfIntUpper);
		printf("U-statistic with confidence interval for the population value:\n[%f %f %f]\n", thetaConfIntLower, estimatedMean, thetaConfIntUpper);
	}
	workspaceDel();
	gsl_matrix_free(data);
	gsl_rng_free(r);
}
