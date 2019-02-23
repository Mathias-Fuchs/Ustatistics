
#include "U.h"
#include <gsl/gsl_rstat.h>
#include <gsl/gsl_statistics.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_cdf.h>
#include <gsl/gsl_math.h>
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
	size_t * indices = (size_t *)malloc(m * sizeof(size_t));

	// estimator for the square of the mean of the subsampling results, will be more accurate than just the square of the mean. See www.mathiasfuchs.de/b2.html
	double G = 0.0;
	for (int b = 0; b < B; b++) {
		sampleWithoutReplacement(n, m, indices, r);
		for (int i = 0; i < m; i++) {
			gsl_vector_set(subresponse, i, gsl_vector_get(response, indices[i]));
			for (int j = 0; j < p; j++) gsl_matrix_set(subsample, i, j, gsl_matrix_get(data, indices[i], j));
		}
		double newval = kernel(subsample, subresponse);
		gsl_vector_set(resamplingResults, b, newval);
		if (b == 0) {
			gsl_vector_set(cumSum, 0, newval);
		} else { 
			gsl_vector_set(cumSum, b, newval + gsl_vector_get(cumSum, b - 1));
		}
		if (b != 0) G += newval * gsl_vector_get(cumSum, b-1) / (double)B / (double) (B - 1) * 2.0;
	}
	free(indices);
	gsl_matrix_free(subsample);
	gsl_vector_free(subresponse);

	// double x = runningMean(resamplingResults, B);
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
		*confIntLower = mean - t * reSampleSd / sqrt((double)B);
		*confIntUpper = mean + t * reSampleSd / sqrt((double)B);
		*Usquared = G;

		// todo: implement the confidence interval for Usquared
	}

	gsl_vector_free(predStorage);
	gsl_vector_free(resamplingResults);
	return(mean);
}

