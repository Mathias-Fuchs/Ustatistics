#include "U.h"
#include "binomCoeff.h"
#include <gsl/gsl_rstat.h>
#include <gsl/gsl_statistics.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_cdf.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_combination.h>
#include <assert.h>
#include <limits.h>
#include <stdio.h>
#include <stdbool.h>

#ifdef RELEASE
#define HAVE_INLINE
#define GSL_NO_BOUNDS_CHECK // etc
#endif


static inline void sampleWithoutReplacement(const size_t populationSize, const size_t sampleSize, size_t* subsample, gsl_rng* r) {
	if (sampleSize > populationSize) { fprintf(stderr, "sampling impossible"); exit(1); }
	if (sampleSize == populationSize) {
		for (unsigned int i = 0; i < sampleSize; i++) subsample[i] = i;
		return;
	}
	sampleWithoutReplacement(populationSize - 1, sampleSize, subsample, r);
	int u = gsl_rng_uniform_int(r, populationSize);
	for (unsigned int i = 0; i < sampleSize; i++)  subsample[i] += (subsample[i] >= u) ? 1 : 0;

	// shuffle the subsample by swapping each array entry with a random one
	for (int i = sampleSize - 1; i >= 0; --i) {
		int j = gsl_rng_uniform_int(r, sampleSize);
		int temp = subsample[i];
		subsample[i] = subsample[j];
		subsample[j] = temp;
	}
}



// number of ordered draws without replacement  unless second entry indicates it's too big. Equals binomial coefficient times factorial of k as long as defined.
static rr drawWithoutReplacementInOrder(size_t n, size_t k) {
	rr b;
	if (k == 0) {
		b.i = 1; b.isInfty = 0; return b;
	}
	if (k == 1) {
		b.i = n; b.isInfty = 0; return b;
	}
	rr o = drawWithoutReplacementInOrder(n - 1, k - 1);
	if (o.isInfty || o.i > INT_MAX / n) {
		b.isInfty = 1; return b;
	}
	b.i = o.i * n;
	b.isInfty = 0;
	return b;
}



double Upure(
	const gsl_matrix* data,
	const size_t B,
	const int m,
	gsl_rng* r,
	kernel_t kernel,
	void* args, // additional arguments to be passed to the kernel
	double* computationConfIntLower,
	double* computationConfIntUpper,
	gsl_vector ** retainResamplingResults) {
	size_t n = data->size1;
	size_t d = data->size2;
	gsl_vector* resamplingResults;
	double Usquared, UsquaredLower, UsquaredUpper;

	// decide if we can generate all subsets
	rr nrDraws = binomialCoefficient(n, m);
	bool explicitResampling = (nrDraws.isInfty == 0 && nrDraws.i < 1e6);
	if (explicitResampling) resamplingResults = gsl_vector_alloc(nrDraws.i);
	else 	resamplingResults = gsl_vector_alloc(B);

	if (explicitResampling) {
		// calculate the U-statistic exactly
		// note that we do assume the kernel is symmetric.

		gsl_combination* cmb = gsl_combination_calloc(n, m);
		gsl_matrix* subsample = gsl_matrix_alloc(m, d);
		int b = 0;
		do {
			for (int i = 0; i < m; i++) {
				size_t* cd = gsl_combination_data(cmb);
				for (unsigned int j = 0; j < d; j++) gsl_matrix_set(subsample, i, j, gsl_matrix_get(data, cd[i], j));
			}
			double newval = kernel(subsample, args);
			//			fprintf(stdout, "kernel evaluation: %f \n", newval);
			gsl_vector_set(resamplingResults, b++, newval);

		} while (gsl_combination_next(cmb) == GSL_SUCCESS);
		gsl_matrix_free(subsample);
		gsl_combination_free(cmb);
	}
	else {
		size_t* indices = malloc(m * sizeof(size_t));
		gsl_matrix* subsample = gsl_matrix_alloc(m, d);
		for (size_t b = 0; b < B; b++) {
			sampleWithoutReplacement(n, m, indices, r);
			for (size_t i = 0; i < (unsigned int)m; i++) {
				for (size_t j = 0; j < (unsigned int)d; j++) gsl_matrix_set(subsample, i, j, gsl_matrix_get(data, indices[i], j));
			}
			double newval = kernel(subsample, args);
			//			fprintf(stdout, "kernel evaluation: %f, mean %f \n", newval, gsl_stats_mean(resamplingResults->data, resamplingResults->stride, b));
			gsl_vector_set(resamplingResults, b, newval);
		}
		gsl_matrix_free(subsample);
		free(indices);
	}

	double mean = gsl_stats_mean(
		resamplingResults->data,
		resamplingResults->stride,
		resamplingResults->size
	);

	if (explicitResampling) {
		fprintf(stdout, "Have calculated the exact U-statistic and its square.");
		if (computationConfIntLower) *computationConfIntLower = mean;
		if (computationConfIntUpper) *computationConfIntUpper = mean;
		Usquared = mean * mean;
		UsquaredLower = mean * mean;
		UsquaredUpper = mean * mean;
	}
	else {
		double reSampleSd = gsl_stats_sd_m(
			resamplingResults->data,
			resamplingResults->stride,
			resamplingResults->size,
			mean
		);
		fprintf(stdout, "In the computation of the U-statistic, a precision of %f has been achieved. This is a relative precision of %f.\n", reSampleSd / sqrt((double)B)
			, reSampleSd / sqrt((double)B) / mean);


		double df = (double)(B - 1); // degrees of freedom in the estimation of the mean of the resampling results
		double t = gsl_cdf_tdist_Pinv(1.0 - 0.05 / 2.0, df);

		// confidence interval for the computation of the U-statistic
		double confIntLower = mean - t * reSampleSd / sqrt((double)B);
		double confIntUpper = mean + t * reSampleSd / sqrt((double)B);

		if (computationConfIntLower) *computationConfIntLower = confIntLower;
		if (computationConfIntUpper) *computationConfIntUpper = confIntUpper;

		double precision = mean / 1e3;

		// we want qt * reSampleSd / sqrt(B) == precision, so (which is the standard sample size calculation)
		float Brequired = (float)(t * t * reSampleSd * reSampleSd / precision / precision);

		fprintf(stdout, "To achieve a relative precision of 1e-3, %i iterations are needed instead of %i,\n", (int)Brequired, (int)B);
		fprintf(stdout, "i.e., %f as many.\n", Brequired / (float)B);
	}
	if (!retainResamplingResults) gsl_vector_free(resamplingResults);
	else *retainResamplingResults = resamplingResults;
	return mean;
}

// the kernel for theta squared.
// the original kernel is supposed to be given as a function pointer, masked as a void pointer
double kernelTS(const gsl_matrix* data, void* kernel) {
	
	kernel_t originalKernel = (kernel_t)kernel;
	size_t m = data->size1 / 2;
	size_t p = data->size2;


	gsl_combination* cmb = gsl_combination_calloc(2 * m, m);
	gsl_matrix* subsample1 = gsl_matrix_alloc(m, p);
	gsl_matrix* subsample2 = gsl_matrix_alloc(m, p);

	int b = 0;
	double k = 0;
	do {
		int* complement = malloc(2 * m * sizeof(int));
		int* complement2 = malloc(m * sizeof(int));
		for (int i = 0; i < 2 * m; i++) complement[i] = i;
		size_t* cd = gsl_combination_data(cmb);

		for (int i = 0; i < m; i++) {
			int co = cd[i];
			complement[co] = -1;
			for (unsigned int j = 0; j < p; j++) gsl_matrix_set(subsample1, i, j, gsl_matrix_get(data, co, j));
		}

		int zz = 0;
		int i = 0;
		do {if (complement[i++] != -1) complement2[zz++] = complement[i - 1];}
			while (zz != m);

		free(complement);
		for (int i = 0; i < m; i++) {
			for (unsigned int j = 0; j < p; j++) gsl_matrix_set(subsample2, i, j, gsl_matrix_get(data, complement2[i], j));
		}


		free(complement2);

		k += originalKernel(subsample1, NULL) * originalKernel(subsample2, NULL);
		b++;
	} while (gsl_combination_next(cmb) == GSL_SUCCESS);
	k /= (double)b;

	gsl_matrix_free(subsample1);
	gsl_matrix_free(subsample2);

	gsl_combination_free(cmb);

	return k;
}


double U(
	const gsl_matrix* data,
	const size_t B,
	const int m,
	gsl_rng* r,
	kernel_t kernel,
	double* computationConfIntLower,
	double* computationConfIntUpper,
	double* thetaConfIntLower,
	double* thetaConfIntUpper,
	double* estthetasquared
) {
	gsl_vector* resamplingResults;
	double mean = Upure(data, B, m, r, kernel, NULL, computationConfIntLower, computationConfIntUpper, &resamplingResults);
	size_t B0 = resamplingResults->size; // in case explicit resampling was done, B might have changed, whence we reset it here.
	size_t n = data->size1;
	size_t d = data->size2;

	if (2 * m <= n) {
		// in that case we prepare for the variance computation
		// now try to estimate the population variance of the U-statistic in case this is possible

		double estimatorThetaSquareLower;
		double estimatorThetaSquareUpper;
		double estimatorThetaSquared = Upure(data, B, 2 * m, r, &kernelTS, kernel, &estimatorThetaSquareLower, &estimatorThetaSquareUpper, NULL);

		// the estimated variance of the U-statistic
		double varianceU = mean * mean - estimatorThetaSquared;

		
		fprintf(stdout, "Variance estimator: %f\n", varianceU);

		// now compute the confidence interval for the U-statistic itself, the most interesting confidence interval.
		// should yield the qt-test confidence interval
		double tt = gsl_cdf_tdist_Pinv(1.0 - 0.05 / 2.0, (double)(n - 1));

		if (varianceU > 0) {
			double conservativeSd = sqrt(varianceU);
			// these values should be close to what the function qt.test outputs for the 95% confidence interval in the one-sample estimation of the mean.
			*thetaConfIntLower = mean - tt * conservativeSd;
			*thetaConfIntUpper = mean + tt * conservativeSd;
		}
		else {
			fprintf(stderr, "variance estimator negative, can't compute confidence interval.\n");
		}
	}
	gsl_vector_free(resamplingResults);
	return(mean);
}

