


/*
 *  calculates classification loss, and estimates the standard error by a U-statistic
 *  Copyright (C) 2013  Mathias Fuchs
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program. If not, see <http://www.gnu.org/licenses/>.
 */


#include <time.h>

#include <stdio.h>
#include <stdlib.h>
#include "regressionLearner.h"
#include "U.h"
#include <gsl/gsl_vector_double.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_statistics.h>
#include <gsl/gsl_blas.h>

#include <assert.h>





gsl_matrix * RandomData(size_t n, size_t  p, gsl_rng * r) {
	gsl_matrix * data = gsl_matrix_alloc(n, p);
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < p; j++) {
			gsl_matrix_set(data, i, j, gsl_ran_ugaussian(r));
		}
	}
	return data;
}

gsl_vector * RandomResponse(int n, gsl_rng * r) {
	gsl_vector * res = gsl_vector_alloc(n);
	for (int i = 0; i < n; i++) {
		gsl_vector_set(res, i, gsl_ran_ugaussian(r));
	}
	return res;
}

void writeDoubleMatrix(const gsl_matrix * x) {
	int n, p;
	n = x->size1;
	p = x->size2;
	int i, j;
	for (i = 0; i < n; i++) {
		for (j = 0; j < p; j++) {
			printf("%f ", (double)gsl_matrix_get(x, i, j));
		}
		printf("\n");
	}
}

void writeDoubleVector(const gsl_vector * x) {
	int n = x->size;
	for (int i = 0; i < n; i++)
		printf("%f ", gsl_vector_get(x, i));
	printf("\n");
}


/* Gamma is  || betaHat Xtest - Ytest ||^2 = */
/*   (Xtest * (Xlearn^t Xlearn)^(-1) *  Xlearn ^t * Ylearn - Ytest )^2 */




int main(int argc, char ** argv) {


	size_t n = 103;
	size_t p = 3;
	size_t B = 1e5; // number of resample in each iteration
	int seed = 1234; // random seed

	gsl_rng * r = gsl_rng_alloc(gsl_rng_taus2);
	gsl_rng_set(r, seed);

	gsl_matrix * X = RandomData(n, p, r);
	gsl_matrix* y = RandomResponse(n, r);

	for (int g = 30; g < 51; g++) {
	
		workspaceInit(3);
		double lpo = U(X, y, B, g + 1, r, &gamma);
		double t2 = U(X, y, B, 2 * g + 2, r, &kernelForThetaSquared);
		workspaceDel();

		printf("learning set size: %i, leave-p-out estimator: %f, estimator for thetasquared: %f, estimator for its variance: %f\n", g, lpo, t2, gsl_pow_2(lpo) - t2);
	}

	gsl_matrix_free(X);
	gsl_vector_free(y);
	return 0;
}

