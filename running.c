#include "running.h"


double runningMean(const gsl_vector * x, const size_t until) {
	double runningAverage = 0;
	for (int i = 0; i < until; i++) {
		runningAverage = (double)i / ((double)(i + 1)) * runningAverage + gsl_vector_get(x, i) / ((double)(i + 1));
	}
	return runningAverage;
}
