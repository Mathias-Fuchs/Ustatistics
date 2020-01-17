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


#include <stdio.h>
#include <stdlib.h>
#include "U.h"
#include "supervisedLearning.h"
#include <gsl/gsl_vector_double.h>
#include <gsl/gsl_matrix.h>


unsigned long llrand() {
	unsigned long r = 0;

	for (int i = 0; i < 5; ++i) {
		r = (r << 15) | (rand() & 0x7FFF);
	}

	return r & 0xFFFFFFFFFFFFFFFFULL;
}

int main() {

	size_t B = 1e5; // number of resamples in each iteration

	FILE * f = fopen("slump.dat", "rb");
	if (!f) fprintf(stderr, "input file not found!\n");
	gsl_matrix * X = gsl_matrix_alloc(103, 3);
	gsl_matrix_fscanf(f, X);
	fclose(f);
	f = fopen("slumpResponse.dat", "rb");
	if (!f) fprintf(stderr, "response input file not found!\n");
	gsl_matrix * dummy = gsl_matrix_alloc(1, 103);
	gsl_matrix_fscanf(f, dummy);
	fclose(f);
	gsl_vector * y = gsl_vector_alloc(103);
	for (int i = 0; i < 103; i++) 	gsl_vector_set(y, i, gsl_matrix_get(dummy, 0, i));
	gsl_matrix_free(dummy);
	gsl_rng * r = gsl_rng_alloc(gsl_rng_taus2);
	gsl_rng_set(r, llrand());
	analyzeDataset(r, X, y, B);

	gsl_matrix_free(X);
	gsl_vector_free(y);
	return 0;
}

