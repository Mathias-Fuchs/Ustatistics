#pragma once

#include <gsl/gsl_matrix.h>
#include <gsl/gsl_rng.h>

// full interface, comprising the computation confidence interval, and the one for theta, as well as an estimated value for the square of theta.


/* computes a U-statistic of degree m by resampling B times */
// compute two confidence intervals: one to express the certainty about the approximation of the true U-statistic by  a finite sample,
// and one to express how well the true U-statistic approximates its estimation target.
// Finally, these two are combined into one confidence interval.
double U(
	const gsl_matrix* data,
	const size_t B,
	const int m,
	gsl_rng* r,
	double(*kernel)(const gsl_matrix*),
	double* computationConfIntLower,
	double* computationConfIntUpper,
	double* thetaConfIntLower,
	double* thetaConfIntUpper,
	double* estthetasquared
);

// the pure interface giving only the computation confidence interval.
double Upure(const gsl_matrix* data,
	const size_t B,
	const int m,
	gsl_rng* r,
	double(*kernel)(const gsl_matrix*),
	double* computationConfIntLower,
	double* computationConfIntUpper,
	gsl_vector** retainResamplingResults
);