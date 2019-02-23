


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

#include <assert.h>






int main(int argc, char ** argv) {


	size_t n = 103;
	size_t p = 3;
	int B = 1e5; // number of resample in each iteration
	int seed = 1234; // random seed
	int g = 50; // learning set size



	size_t Br = (size_t)B;
	gsl_matrix * X = gsl_matrix_alloc(n, p);
	FILE * f = fopen("slump.dat", "rb");

	if (!f) {
		fprintf(stderr, "input file not found!\n");
	}

	int h = gsl_matrix_fscanf(f, X);
	fclose(f);

	gsl_vector * y = gsl_vector_alloc(103);
	f = fopen("slumpResponse.dat", "rb");

	if (!f) {
		fprintf(stderr, "response input file not found!\n");
	}
	gsl_matrix * dummy = gsl_matrix_alloc(1, 103);
	gsl_matrix_fscanf(f, dummy);
	fclose(f);

	for (int i = 0; i < 103; i++) {
		gsl_vector_set(y, i, gsl_matrix_get(dummy, 0, i));
	}

	gsl_matrix_free(dummy);
	gsl_rng * r = gsl_rng_alloc(gsl_rng_taus2);
	gsl_rng_set(r, seed);


	for (int g = 30; g < 51; g++) {
		workspaceInit(3);
		double lpo = U(X, y, Br, g + 1, r, &gamma);
		double t2 = U(X, y, Br, 2 * g + 2, r, &kernelForThetaSquared);
		workspaceDel();


		printf("learning set size: %i, leave-p-out estimator: %f, estimator for thetasquared: %f, estimator for its variance: %f\n", g, lpo, t2, gsl_pow_2(lpo) - t2);
	}
	gsl_matrix_free(X);
	gsl_vector_free(y);

	return 0;
}

