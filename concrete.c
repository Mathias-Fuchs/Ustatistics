


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
#include <gsl/gsl_cdf.h>
#include <assert.h>






int main(int argc, char ** argv) {


	size_t n = 103;
	size_t p = 3;
	size_t B = 1e7; // number of resample in each iteration
	int seed = 1234; // random seed

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

		double confIntLower1, confIntUpper1, Usquared1, UsquaredLower1, UsquaredUpper1;
		double confIntLower2, confIntUpper2, Usquared2, UsquaredLower2, UsquaredUpper2;

		double lpo = U(X, y, B, g + 1, r, &gamma, &confIntLower1, &confIntUpper1, &Usquared1, &UsquaredLower1, &UsquaredUpper1);
		double t2 = U(X, y, B, 2 * g + 2, r, &kernelForThetaSquared, &confIntLower2, &confIntUpper2, &Usquared2, &UsquaredLower2, &UsquaredUpper2);
		workspaceDel();
		printf("learning set size: %i\n", g);
		printf("leave-p-out estimator with confidence interval for its exact computation: [%f %f %f]\n", confIntLower1, lpo, confIntUpper1);
		printf("its square with confidence interval for its computation: [%f %f %f]\n", UsquaredLower1, Usquared1, UsquaredUpper1);
		printf("computation uncertainty in lposquared %f\n", UsquaredUpper1 - UsquaredLower1);
		printf("computation uncertainty in thetasquared: %f\n", confIntUpper2 - confIntLower2);
		printf("Adjust the Bs by a factor of %f therefore.\n", (UsquaredUpper1 - UsquaredLower1) / (confIntUpper2 - confIntLower2) * (UsquaredUpper1 - UsquaredLower1) / (confIntUpper2 - confIntLower2));
		printf("computation confidence interval for the variance estimator: [%f %f %f]\n", UsquaredLower1 - confIntUpper2, Usquared1 - t2, UsquaredUpper1 - confIntLower2);
		printf("computation uncertainty in the variance estimator: %f\n", UsquaredUpper1 - confIntLower2 - (UsquaredLower1 - confIntUpper2));
		double t = gsl_cdf_tdist_Pinv(1.0 - 0.05 / 2.0, (double)(n - 1));
		double conservativeSd = sqrt(UsquaredUpper1 - confIntLower2);
		printf("resulting conservative confidence interval for the supervised learning algorithm using the upper variance computation confidence interval:[%f %f %f]\n\n", lpo - t * conservativeSd, lpo, lpo + t * conservativeSd);

		
	}
	gsl_matrix_free(X);
	gsl_vector_free(y);
	gsl_rng_free(r);
	return 0;
}

