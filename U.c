
#include "U.h"
#include <gsl/gsl_rstat.h>
#include <gsl/gsl_statistics.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_cdf.h>
#include <gsl/gsl_math.h>
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
double U(
	const gsl_matrix * data,
	const gsl_vector * response,
	const size_t B,
	const int m,
	gsl_rng * r,
	double(*kernel)(const gsl_matrix *, const gsl_vector *),
	double* confIntLower,
	double* confIntUpper,
	double* Usquared,
	double* UsquaredLower,
	double* UsquaredUpper) {
	int n = data->size1;
	int p = data->size2;
	assert(response->size == n);

	gsl_vector * resamplingResults = gsl_vector_alloc(B);
	gsl_vector * cumSum = gsl_vector_alloc(B);
	gsl_matrix * subsample = gsl_matrix_alloc(m, p);
	gsl_vector * subresponse = gsl_vector_alloc(m);
	gsl_vector * predStorage = gsl_vector_alloc(B);

	// will hold the indices of a subsample
	size_t * indices = malloc(m * sizeof(size_t));

	for (int b = 0; b < B; b++) {
		sampleWithoutReplacement(n, m, indices, r);
		for (int i = 0; i < m; i++) {
			gsl_vector_set(subresponse, i, gsl_vector_get(response, indices[i]));
			for (int j = 0; j < p; j++) gsl_matrix_set(subsample, i, j, gsl_matrix_get(data, indices[i], j));
		}
		double newval = kernel(subsample, subresponse);
		gsl_vector_set(resamplingResults, b, newval);
	}

#ifdef codeToCheckCorrectness
	// old code to check the sum of all products of distinct pairs and triples

	double cumsum = 0, G = 0, GG = 0, GGG = 0, H = 0;
	for (int i = 1; i < B; i++) {
		cumsum += resamplingResults->data[i - 1]; // one needs to take the preceding value
		GG += resamplingResults->data[i] * cumsum / (double)B / (double)(B - 1) * 2.0;
	}

	for (int i = 0; i < B - 1; i++) {
		for (int j = i + 1; j < B; j++) {
			GGG += gsl_vector_get(resamplingResults, i) * gsl_vector_get(resamplingResults, j) / (double)B / (double)(B - 1) * 2.0;
		}
	}

	for (int i = 0; i < B - 2; i++) {
		for (int j = i + 1; j < B - 1; j++) {
			for (int k = j + 1; k < B; k++) {
				H += 
					resamplingResults->data[i] * resamplingResults->data[j] * resamplingResults->data[k];
			}
		}
	}
#endif

	double* N = calloc(4 * B, sizeof(double));
	for (int i = 0; i < B; i++) N[i + B * 0] = (i ? N[i - 1 + B * 0] : 0) + gsl_vector_get(resamplingResults, i);
	for (int i = 1; i < B; i++) for (int j = 1; j < 4; j++) N[i + B * j] = gsl_vector_get(resamplingResults, i) * N[i - 1 + B * (j - 1)] + N[i - 1 + B * j];

	double sumOfProductsOfDistinctPairs = N[B - 1 + B];
	double sumOfProductsOfDistinctTriples = N[B - 1 + B * 2];
	double sumOfProductsOfDistinctQuadruples = N[B - 1 + B * 3];

	double rawSecondMoment = sumOfProductsOfDistinctPairs / (double)B / (double)(B - 1) * 2.0;
	double rawThirdMoment = sumOfProductsOfDistinctTriples / (double)B / (double)(B - 1) / (double) (B - 2) * 6.0;
	double rawFourthMoment = sumOfProductsOfDistinctQuadruples / (double)B / (double)(B - 1) / (double) (B-2) / (double) (B-3) * 24.0;
	free(N);
	
	free(indices);
	gsl_matrix_free(subsample);
	gsl_vector_free(subresponse);

	double mean = gsl_stats_mean(
		resamplingResults->data,
		resamplingResults->stride,
		resamplingResults->size
	);

	if (confIntLower) {
		double df = (double)(B - 1); // degrees of freedom in the estimation of the mean of the resampling results
		double t = gsl_cdf_tdist_Pinv(1.0 - 0.05 / 2.0, df);

		double reSampleSd = gsl_stats_sd_m(
			resamplingResults->data,
			resamplingResults->stride,
			resamplingResults->size,
			mean
		);

		// confint  is [x - t * sd/sqrt(n), x + t * sd/sqrt(n)]
		double K = rawSecondMoment * rawSecondMoment - rawFourthMoment;

		*confIntLower = mean - t * reSampleSd / sqrt((double)B);
		*confIntUpper = mean + t * reSampleSd / sqrt((double)B);
		*Usquared = rawSecondMoment;
		*UsquaredLower = rawSecondMoment - t * sqrt(K);
		*UsquaredUpper = rawSecondMoment + t * sqrt(K);
	}

	gsl_vector_free(predStorage);
	gsl_vector_free(resamplingResults);
	return(mean);
}

