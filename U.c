
#include "U.h"
#include "running.h"
#include <assert.h>


static inline void sampleWithoutReplacement(const size_t populationSize, const size_t sampleSize, size_t * subsample, gsl_rng * r) {


	int n = sampleSize;
	int N = populationSize;

	int t = 0;
	int m = 0;

	double u;
	while (m < n) {
		u = gsl_rng_uniform_pos(r); // call a uniform(0,1) random number generat

		if ((N - t) *  u >= n - m) {
			t++;
		}
		else {
			subsample[m] = t;
			t++;
			m++;
		}
	}

	// we now have to shuffle the resulting indices, so they are no more sorted
	if (n > 1) {
		for (int i = 0; i < n - 1; i++) {
			size_t j = i + (gsl_rng_get(r) - gsl_rng_min(r)) / (gsl_rng_max(r) / (n - i) + 1);
			size_t t = subsample[j];
			subsample[j] = subsample[i];
			subsample[i] = t;
		}
	}
}


/* computes a U-statistic of degree m by resampling B times */
double U(const gsl_matrix * data, const gsl_vector * response, const size_t B, const int m, gsl_rng * r, double(*kernel)(const gsl_matrix *, const gsl_vector *)) {
	int n = data->size1;
	int p = data->size2;
	assert(response->size == n);

	int b;

	gsl_vector * resamplingResults = gsl_vector_alloc(B);
	gsl_matrix * subsample = gsl_matrix_alloc(m, p);
	gsl_vector * subresponse = gsl_vector_alloc(m);
	gsl_vector * predStorage = gsl_vector_alloc(B);

	// will hold the indices of a subsample
	size_t * indices = (size_t *)malloc(m * sizeof(size_t));

	for (b = 0; b < B; b++) {
		sampleWithoutReplacement(n, m, indices, r);
		for (int i = 0; i < m; i++) {
			// printf("%i ", indices[i]);
			gsl_vector_set(subresponse, i, gsl_vector_get(response, indices[i]));
			for (int j = 0; j < p; j++) {
				gsl_matrix_set(subsample, i, j, gsl_matrix_get(data, indices[i], j));
			}
		}

		gsl_vector_set(resamplingResults, b, kernel(subsample, subresponse));

		if (b && b % (int) 1e5 == 0) {
			printf("%f %f \n", b + 0.0, runningMean(resamplingResults, b));
		}
	}
	free(indices);
	gsl_matrix_free(subsample);
	gsl_vector_free(subresponse);

	double x = runningMean(resamplingResults, B);
	/* double x = gsl_stats_mean( */
	/* 			    resamplingResults -> data, */
	/* 			    resamplingResults -> stride, */
	/* 			    resamplingResults -> size */
	/* 			    ); */

	//  writeDoubleMatrix(coeffStorage);
	/* gsl_vector_view AA = gsl_matrix_subcolumn(coeffStorage, 0, 0, (size_t) B); */
	/* gsl_vector_view BB = gsl_matrix_subcolumn(coeffStorage, 1, 0, (size_t) B); */
	/* gsl_vector_view CC = gsl_matrix_subcolumn(coeffStorage, 2, 0, (size_t) B); */

	gsl_vector_free(predStorage);
	gsl_vector_free(resamplingResults);
	return(x);
}

