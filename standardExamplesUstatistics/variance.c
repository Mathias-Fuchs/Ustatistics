// this file reproduces the confidence interval for the population variance
// in R, it is obtained by
/*

> require(ConfIntVariance)
> varwci(c(2, 2, 4, 6, 2, 6, 4, 5, 4, 6))
[1] 1.290407 4.242927
attr(,"point.estimator")
[1] 2.766667
attr(,"conf.level")
[1] 0.95
attr(,"var.SampleVariance")
[1] 0.425873
*/


#include "U.h"
#include <gsl/gsl_matrix.h>

// the kernel of the U-statistics for the variance is (x_1 - x_2)^2 / 2
double kern(const gsl_matrix* data) {
	return (gsl_matrix_get(data, 0, 0) - gsl_matrix_get(data, 1, 0)) *
		(gsl_matrix_get(data, 0, 0) - gsl_matrix_get(data, 1, 0)) / 2.0;
}

int main() {
	gsl_rng* r = gsl_rng_alloc(gsl_rng_taus2);
	gsl_rng_set(r, 1234);
	size_t B = 1e4;
	int n = 10;
	gsl_matrix* data = gsl_matrix_alloc(n, 1);
	if (n == 10) {
	// standard example
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
	}
	else {
		for (int j = 0; j < n; j++) gsl_matrix_set(data, j, 0, (double)gsl_rng_uniform_int(r, 6));
	}

	double computationConfIntLower, computationConfIntUpper, thetaConfIntLower, thetaConfIntUpper;
	double estimatedMean = U(
		data, B, 2, r, kern, &computationConfIntLower, &computationConfIntUpper, &thetaConfIntLower, &thetaConfIntUpper);
	printf("U-statistic with confidence interval for its exact computation:\n[%f %f %f]\n", computationConfIntLower, estimatedMean, computationConfIntUpper);
	printf("U-statistic with confidence interval for the population value:\n[%f %f %f]\n", thetaConfIntLower, estimatedMean, thetaConfIntUpper);
}
