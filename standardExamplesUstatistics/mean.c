#include "U.h"
#include <gsl/gsl_matrix.h>

// the kernel of the U-statistics for the mean is just the identity function on a one-times-one-matrix
double kern(gsl_matrix* data) {
  return gsl_matrix_get(data, 0, 0);
}

int main() {
  gsl_matrix* data = gsl_matrix_alloc(10, 1);
  gsl_matrix_set(data, 0, 2.0);
  gsl_matrix_set(data, 1, 2.0);
  gsl_matrix_set(data, 2, 4.0);
  gsl_matrix_set(data, 3, 6.0);
  gsl_matrix_set(data, 4, 2.0);
  gsl_matrix_set(data, 5, 6.0);
  gsl_matrix_set(data, 6, 4.0);
  gsl_matrix_set(data, 7, 5.0);
  gsl_matrix_set(data, 8, 4.0);
  gsl_matrix_set(data, 9, 6.0);
  double confIntLower1, confIntUpper1, Usquared1, UsquaredLower1, UsquaredUpper1;
  double estimatedMean = U(data, B, 1, r, &kern, &confIntLower1, &confIntUpper1, &Usquared1, &UsquaredLower1, &UsquaredUpper1);
  printf("U-statistic with confidence interval for its exact computation:\n[%f %f %f]\n", confIntLower1, lpo, confIntUpper1);
  printf("Its square with confidence interval for its computation:\n[%f %f %f]\n", UsquaredLower1, Usquared1, UsquaredUpper1);
}
