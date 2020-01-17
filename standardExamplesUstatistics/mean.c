
// what the mean is for the t-test,
// this package is for a general U-statistic.
// So, this package allows to give confidence intervals for any U-statistic.
// This is of particular importance for the U-statistic that arises from (a generalized form of) cross-validation in supervised learning.
// Thus, this software uses U-statistics theory to compute confidence intervals for the error rate in supervised learning.
// Estimating the error rate is a problem of paramount importance in Machine Learning.

// executing this file reproduces the t-test's confidence interval.
// For the particular data example  2 2 4 6 2 6 4 5 4 6, the R output is:



/* t.test(c(2, 2, 4, 6, 2, 6, 4, 5, 4, 6)) */

/* 	One Sample t-test */

/* data:   c(2, 2, 4, 6, 2, 6, 4, 5, 4, 6) */
/* t = 7.7948, df = 9, p-value = 2.723e-05 */
/* alternative hypothesis: true mean is not equal to 0 */
/* 95 percent confidence interval: */
/*  2.910125 5.289875 */
/* sample estimates: */
/* mean of x  */
/*       4.1  */




#include "U.h"
#include <gsl/gsl_matrix.h>

// the kernel of the U-statistics for the mean is just the identity function on a one-times-one-matrix
double kern(const gsl_matrix* data) {
	return gsl_matrix_get(data, 0, 0);
}

int main() {
	gsl_rng* r = gsl_rng_alloc(gsl_rng_taus2);
	gsl_rng_set(r, 1234);
#ifdef _DEBUG
	size_t B = 1e4;
#else
	size_t B = 1e7;
#endif
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
	double computationConfIntLower, computationConfIntUpper, thetaConfIntLower, thetaConfIntUpper;
	double estimatedMean = U(
		data, B, 1, r, kern, &computationConfIntLower, &computationConfIntUpper, &thetaConfIntLower, &thetaConfIntUpper, NULL);
	printf("U-statistic with confidence interval for its exact computation:\n[%f %f %f]\n", computationConfIntLower, estimatedMean, computationConfIntUpper);
	printf("U-statistic with confidence interval for the population value:\n[%f %f %f]\n", thetaConfIntLower, estimatedMean, thetaConfIntUpper);
}
