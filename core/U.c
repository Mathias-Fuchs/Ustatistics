#include "U.h"
#include <gsl/gsl_rstat.h>
#include <gsl/gsl_statistics.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_cdf.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_combination.h>
#include <assert.h>
#include <limits.h>
#include <stdio.h>

static inline void sampleWithoutReplacement(const size_t populationSize, const size_t sampleSize, size_t * subsample, gsl_rng * r) {
	int max = populationSize - 1;
	int min = 0;
	int x = sampleSize;
	size_t rr,i=x,*a;
	a=subsample;
	while (i--) {
		rr=(max-min+1-i);
		a[i]=min+=(rr ? rand()%rr : 0);
		min++;
	}
	while (x>1) {
		rr=a[i=rand()%x--];
		a[i]=a[x];
		a[x]=rr;
	}
	for (unsigned int i = 0; i < sampleSize; i++)
		fprintf(stdout, "%i ", (int) a[i]);
	fprintf(stdout, "\n");
}

// struct to keep track of whether an unsigned int is beyond the bounds
typedef struct {
	unsigned int i;
	int isInfty;
} rr;

// binomialCoefficient unless second entry indicates it's too big
static rr binomialCoefficient(size_t n, size_t k) {
	rr b;
	if (k == 0) {
		b.i = 1; b.isInfty = 0; return b;
	}
	if (k == 1) {
		b.i = n; b.isInfty = 0; return b;
	}
	if (k > n/2) return binomialCoefficient(n, n-k);
	rr o = binomialCoefficient(n - 1, k - 1);
	if (o.isInfty || o.i > INT_MAX / n * k) {
		b.isInfty = 1; return b;
	}
	b.i = o.i * n / k;
	b.isInfty = 0;
	return b;
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



/* computes a U-statistic of degree m by resampling B times */
// compute two confidence intervals: one to express the certainty about the approximation of the true U-statistic by  a finite sample,
// and one to express how well the true U-statistic approximates its estimation target.
// Finally, these two are combined into one confidence interval.

double U(
	const gsl_matrix* data,
	const size_t B,
	const int m,
	gsl_rng* r,
	double(*kernel)(const gsl_matrix *),
	double* computationConfIntLower,
	double* computationConfIntUpper,
	double* thetaConfIntLower,
	double* thetaConfIntUpper
	) {

	size_t n = data->size1;
	size_t d = data->size2;


	gsl_vector * resamplingResults;
	double Usquared, UsquaredLower, UsquaredUpper;
	
	// decide if we can generate all subsets
	rr nrDraws = binomialCoefficient(n, m);

	if (nrDraws.isInfty == 0 && nrDraws.i < 1e6) {
		resamplingResults = gsl_vector_alloc(nrDraws.i);
	}
	else {
		resamplingResults = gsl_vector_alloc(B);
	}

	if (nrDraws.isInfty == 0 && nrDraws.i < 1e6) {

		// calculate the U-statistic exactly
		// note that we do assume the kernel is symmetric.
		
		gsl_combination* cmb = gsl_combination_calloc(n, m);
		gsl_matrix * subsample = gsl_matrix_alloc(m, d);
		int b = 0;
		do {
			for (int i= 0; i < m; i++) {
				for (unsigned int j = 0; j < d; j++) gsl_matrix_set(subsample, i, j, gsl_matrix_get(data, gsl_combination_data(cmb)[i], j));
			}
			double newval = kernel(subsample);
			gsl_vector_set(resamplingResults, b++, newval);
			
		} while (gsl_combination_next(cmb) == GSL_SUCCESS);
		gsl_matrix_free(subsample);
		gsl_combination_free(cmb);
	}
	else {
		size_t * indices = malloc(m * sizeof(size_t));
		gsl_matrix* subsample = gsl_matrix_alloc(m,d );
		for (size_t b = 0; b < B; b++) {
			sampleWithoutReplacement(n, m, indices, r);
			for (size_t i = 0; i < (unsigned int) m; i++) {
				for (size_t j = 0; j < (unsigned int) d; j++) gsl_matrix_set(subsample, i, j, gsl_matrix_get(data, indices[i], j));
			}
			double newval = kernel(subsample);
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

	if (nrDraws.isInfty == 0 && nrDraws.i < 1e6) {
		fprintf(stdout, "Have calculated the exact U-statistic and its square.");
		if (computationConfIntLower) *computationConfIntLower = mean;
		if (computationConfIntUpper) *computationConfIntUpper = mean;
	}



	double reSampleSd = gsl_stats_sd_m(
		resamplingResults->data,
		resamplingResults->stride,
		resamplingResults->size,
		mean
		);
	gsl_vector_free(resamplingResults);
	
	double df = (double)(B - 1); // degrees of freedom in the estimation of the mean of the resampling results
	double t = gsl_cdf_tdist_Pinv(1.0 - 0.05 / 2.0, df);
	
	// confidence interval for the computation of the U-statistic
	double confIntLower = mean - t * reSampleSd / sqrt((double)B);
	double confIntUpper = mean + t * reSampleSd / sqrt((double)B);
	
	if (computationConfIntLower) *computationConfIntLower = confIntLower;
	if (computationConfIntUpper) *computationConfIntUpper = confIntUpper;
	
	double precision = mean / 1e2;

// we want t * reSampleSd / sqrt(B) == precision, so (which is the standard sample size calculation)
	float Brequired = (float)(t * t * reSampleSd * reSampleSd / precision / precision);
	
	fprintf(stdout, "To achieve a relative precision of 1e-2, %i iterations are needed instead of %i,\n", (int)Brequired, (int)B);
	fprintf(stdout, "i.e., %f as many.\n", Brequired / (float)B);
	
	if (2 * m <= n) {
		// in that case we prepare for the variance computation
		double* N = calloc(4 * B, sizeof(double));
		if (!N) { fprintf(stderr, "Out of memory.\n"); exit(1); }
		for (size_t i = 0; i < B; i++) N[i + B * 0] = (i ? N[i - 1 + B * 0] : 0) + gsl_vector_get(resamplingResults, i);
		for (size_t i = 1; i < B; i++) for (int j = 1; j < 4; j++) N[i + B * j] = gsl_vector_get(resamplingResults, i) * N[i - 1 + B * (j - 1)] + N[i - 1 + B * j];
		
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
		
		double sumOfProductsOfDistinctPairs = N[B - 1 + B];
		// double sumOfProductsOfDistinctTriples = N[B - 1 + B * 2];
		double sumOfProductsOfDistinctQuadruples = N[B - 1 + B * 3];
		free(N);
		
		double EstimatedSquareOfMean = sumOfProductsOfDistinctPairs / (double)B / (double)(B - 1) * 2.0;
		double EstimatedFourthPowerOfMean = sumOfProductsOfDistinctQuadruples / (double)B / (double)(B - 1) / (double)(B - 2) / (double)(B - 3) * 24.0;
		
		double K = EstimatedSquareOfMean * EstimatedSquareOfMean - EstimatedFourthPowerOfMean;
		
		// the best estimator for the square of the actual U-statistic
		Usquared = EstimatedSquareOfMean;

// note that we don't need to divide K by B
		UsquaredLower = EstimatedSquareOfMean - t * sqrt(K);
		UsquaredUpper = EstimatedSquareOfMean + t * sqrt(K);
	
		gsl_vector_free(resamplingResults);
		
		// now try to estimate the population variance of the U-statistic in case this is possible
		gsl_matrix* subsample = gsl_matrix_alloc(2 * m, d);
		resamplingResults = gsl_vector_alloc(B);
		
		// one could also compute whether all pairs of disjoint m-subsets of 1...n are few enough to iterate through.
		size_t* indices = malloc(2 * m * sizeof(size_t));
		for (int b = 0; b < B; b++) {
			sampleWithoutReplacement(n, 2 * m, indices, r);
			for (size_t i = 0; i < (unsigned int) (2 * m); i++) {
				for (size_t j = 0; j < (unsigned int) d; j++) gsl_matrix_set(subsample, i, j, gsl_matrix_get(data, indices[i], j));
			}
			gsl_matrix_const_view data1 = gsl_matrix_const_submatrix(subsample, 0, 0, subsample->size1 / 2, subsample->size2);
			gsl_matrix_const_view data2 = gsl_matrix_const_submatrix(subsample, subsample->size1 / 2, 0, subsample->size1 / 2, subsample->size2);
			
			double newval = kernel(&data1.matrix) * kernel(&data2.matrix);
			gsl_vector_set(resamplingResults, b, newval);
		}
		free(indices);
		gsl_matrix_free(subsample);
		double estimatorThetaSquared = gsl_stats_mean(
			resamplingResults->data,
			resamplingResults->stride,
			resamplingResults->size
			);
		
		double tsSd = gsl_stats_sd_m(
			resamplingResults->data,
			resamplingResults->stride,
			resamplingResults->size,
			estimatorThetaSquared
			);
		gsl_vector_free(resamplingResults);


		double estimatorThetaSquareLower = estimatorThetaSquared - t * tsSd / sqrt((double)B);
		double estimatorThetaSquareUpper = estimatorThetaSquared + t * tsSd / sqrt((double)B);
		
// the estimated variance of the U-statistic
		double varianceU = Usquared - estimatorThetaSquared;
		
		// Let's try to compute how confident we can by into the computation accuracy of the variance estimator
		double varianceUUpper = UsquaredUpper - estimatorThetaSquareLower;
		double varianceULower = UsquaredLower - estimatorThetaSquareUpper;

		// now compute the confidence interval for the U-statistic itself, the most interesting confidence interval.
		// should yield the t-test confidence interval
		double tt = gsl_cdf_tdist_Pinv(1.0 - 0.05 / 2.0, (double)(n - 1));
		if (varianceU > 0) {
			double conservativeSd = sqrt(varianceU);
			*thetaConfIntLower = mean - tt * conservativeSd;
			*thetaConfIntUpper = mean + tt * conservativeSd;
		} else {
			fprintf(stderr, "variance estimator negative, can't compute confidence interval.\n");
		}
	}
	return(mean);
}
