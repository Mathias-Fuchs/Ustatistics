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
#include <gsl/gsl_rng.h>
#include <assert.h>

gsl_matrix * RandomData(size_t n, size_t  p, gsl_rng * r) {
	gsl_matrix * data = gsl_matrix_alloc(n, p);
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < p; j++) {
			double t = gsl_rng_uniform(r);
			gsl_matrix_set(data, i, j, t);
		}
	}
	return data;
}

gsl_vector * RandomResponse(int n, gsl_rng * r) {
	gsl_vector * res = gsl_vector_alloc(n);
	for (int i = 0; i < n; i++) {
		gsl_vector_set(res, i, gsl_rng_uniform(r));
	}
	return res;
}

int main(int argc, char ** argv) {
	size_t n = 103;
	size_t p = 3;
	size_t B = 1e4; // number of resample in each iteration
	gsl_rng * r = gsl_rng_alloc(gsl_rng_taus2);
	gsl_rng_set(r, 1234);
	gsl_matrix * X = RandomData(n, p, r);
	gsl_matrix* y = RandomResponse(n, r);
	gsl_rng_free(r);
	analyzeDataset(X, y, B);
	gsl_matrix_free(X);
	gsl_vector_free(y);
	return 0;
}

