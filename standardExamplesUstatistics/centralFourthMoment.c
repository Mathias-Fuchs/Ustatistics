// this file computes the confidence interval for the unbiased sample fourth moment

#include "U.h"
#include <gsl/gsl_matrix.h>

// the symmetric kernel of the U-statistic for the central fourth moment is
// (caution: clumsy to be unbiased)
/*
  (x_1 ^4 + x_2^4 + x_3^4 + x_4^4)/4 
  - 4(
x_1 * x_2^3 + x_1 * x_3^3 + x_1 * x_4^3
x_2 * x_1^3 + x_2 * x_3^3 + x_2 * x_4^3
x_3 * x_1^3 + x_3 * x_2^3 + x_3 * x_4^3
x_4 * x_1^3 + x_4 * x_2^3 + x_4 * x_3^3
)/12
+ 6(
x_1^2 * x_2 * x_3 + x_1^2 * x_2 * x_4 + x_1^2 * x_3 * x_4 
x_2^2 * x_1 * x_3 + x_2^2 * x_1 * x_4 + x_2^2 * x_3 * x_4 
x_3^2 * x_1 * x_2 + x_3^2 * x_1 * x_4 + x_3^2 * x_2 * x_4 
x_4^2 * x_1 * x_2 + x_4^2 * x_1 * x_3 + x_4^2 * x_2 * x_3
)/12
- 3(x_1 * x_2 * x_3 * x_4); 
*/

double kern(const gsl_matrix* data) {

	double x1 = gsl_matrix_get(data, 0, 0);
	double x2 = gsl_matrix_get(data, 1, 0);
	double x3 = gsl_matrix_get(data, 2, 0);
	double x4 = gsl_matrix_get(data, 3, 0);

	return  (x1*x1*x1*x1 + x2*x2*x2*x2 + x3*x3*x3*x3 + x4*x4*x4*x4)/4.0 
		- (
			x1 * x2*x2*x2 + x1 * x3*x3*x3 + x1 * x4*x4*x4
		+	x2 * x1*x1*x1 + x2 * x3*x3*x3 + x2 * x4*x4*x4
		+	x3 * x1*x1*x1 + x3 * x2*x2*x2 + x3 * x4*x4*x4
		+	x4 * x1*x1*x1 + x4 * x2*x2*x2 + x4 * x3*x3*x3
			)/3.0
		+ (
			x1 * x1 * x2 * x3 + x1 * x1 * x2 * x4 + x1 * x1 * x3 * x4 
		+	x2 * x2 * x1 * x3 + x2 * x2 * x1 * x4 + x2 * x2 * x3 * x4 
		+	x3 * x3 * x1 * x2 + x3 * x3 * x1 * x4 + x3 * x3 * x2 * x4 
		+	x4 * x4 * x1 * x2 + x4 * x4 * x1 * x3 + x4 * x4 * x2 * x3
			)/2.0
	  - 3.0 * (x1 * x2 * x3 * x4)
	  ;
}

int main() {
	gsl_rng* r = gsl_rng_alloc(gsl_rng_taus2);
	gsl_rng_set(r, 1234);
	size_t B = 1e5;
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
		data, B, 4, r, kern, &computationConfIntLower, &computationConfIntUpper, &thetaConfIntLower, &thetaConfIntUpper);
	printf("U-statistic with confidence interval for its exact computation:\n[%f %f %f]\n", computationConfIntLower, estimatedMean, computationConfIntUpper);
	printf("U-statistic with confidence interval for the population value:\n[%f %f %f]\n", thetaConfIntLower, estimatedMean, thetaConfIntUpper);
}
