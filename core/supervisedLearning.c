#include "supervisedLearning.h"
#include "regressionLearner.h"
#include <gsl/gsl_rng.h>
#include <assert.h>


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
		double confIntLower1, confIntUpper1, Usquared1, UsquaredLower1, UsquaredUpper1;
		double confIntLower2, confIntUpper2;
		double lpo = U(data, B, g + 1, r, &gamma, &confIntLower1, &confIntUpper1, &Usquared1, &UsquaredLower1, &UsquaredUpper1);
		double t2 = U(data, B, 2 * g + 2, r, &kernelForThetaSquared, &confIntLower2, &confIntUpper2, NULL, NULL, NULL);
		printf("Learning set size: %i\n", g);
		printf("Leave-p-out estimator with confidence interval for its exact computation:\n[%f %f %f]\n", confIntLower1, lpo, confIntUpper1);
		printf("Its square with confidence interval for its computation:\n[%f %f %f]\n", UsquaredLower1, Usquared1, UsquaredUpper1);
		printf("Computation uncertainty in lposquared %f\n", UsquaredUpper1 - UsquaredLower1);
		printf("Computation uncertainty in thetasquared: %f\n", confIntUpper2 - confIntLower2);
		printf("Adjust the Bs by a factor of %f therefore.\n", (UsquaredUpper1 - UsquaredLower1) / (confIntUpper2 - confIntLower2) * (UsquaredUpper1 - UsquaredLower1) / (confIntUpper2 - confIntLower2));
		printf("Computation confidence interval for the variance estimator:\n[%f %f %f]\n", UsquaredLower1 - confIntUpper2, Usquared1 - t2, UsquaredUpper1 - confIntLower2);
		printf("Computation uncertainty in the variance estimator: %f\n", UsquaredUpper1 - confIntLower2 - (UsquaredLower1 - confIntUpper2));
		double t = gsl_cdf_tdist_Pinv(1.0 - 0.05 / 2.0, (double)(n - 1));
		double conservativeSd = sqrt(UsquaredUpper1 - confIntLower2);
		printf("Resulting conservative confidence interval for the supervised learning algorithm using the upper variance computation confidence interval:\n[%f %f %f]\n\n", lpo - t * conservativeSd, lpo, lpo + t * conservativeSd);
	}
	workspaceDel();
	gsl_matrix_free(data);
	gsl_rng_free(r);
}