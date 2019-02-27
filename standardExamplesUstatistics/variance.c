#include "U.h"
#include <gsl/gsl_matrix.h>

// the kernel of the U-statistics for the variance is (x_1 - x_2)^2 / 2
double kern(gsl_matrix* data) {
	return (gsl_matrix_get(data, 0, 0) - gsl_matrix_get(data, 1, 0)) *
		(gsl_matrix_get(data, 0, 0) - gsl_matrix_get(data, 1, 0)) / 2.0;
}

int main() {
	gsl_rng* r = gsl_rng_alloc(gsl_rng_taus2);
	gsl_rng_set(r, 1234);
	size_t B = 1e4;
	gsl_matrix* data = gsl_matrix_alloc(10, 1);
	gsl_matrix_set(data, 0, 0, 2.0);
	gsl_matrix_set(data, 1, 0, 2.0);
	gsl_matrix_set(data, 2, 0, 4.0);
	gsl_matrix_set(data, 3, 0, 6.0);
	gsl_matrix_set(data, 4, 0, 2.0);
	gsl_matrix_set(data, 5, 0, 6.0);
	gsl_matrix_set(data, 6, 0, 4.0);
	gsl_matrix_set(data, 7, 0, 5.0);
	gsl_matrix_set(data, 8, 0, 4.0);
	gsl_matrix_set(data, 9, 0, 6.0);
	double confIntLower1, confIntUpper1, Usquared1, UsquaredLower1, UsquaredUpper1;
	double estimatedMean = U(data, B, 2, r, &kern, &confIntLower1, &confIntUpper1, &Usquared1, &UsquaredLower1, &UsquaredUpper1);
	printf("U-statistic with confidence interval for its exact computation:\n[%f %f %f]\n", confIntLower1, estimatedMean, confIntUpper1);
	printf("Its square with confidence interval for its computation:\n[%f %f %f]\n", UsquaredLower1, Usquared1, UsquaredUpper1);
}
