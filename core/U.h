#pragma once

#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_rng.h>


// the type a kernel is of: it is a pointer to a function eating const gsl_matrix*, and outputs double
// this is a recursive typedef. Usually, the second argument can be ignored but it will come in handy when one wants to manufacture a new kernel from an old one.
typedef  double (*kernel_t)(const gsl_matrix*, void*); 

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
	kernel_t kernel,
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
	kernel_t kernel,
	void* args,
	double* computationConfIntLower,
	double* computationConfIntUpper,
	gsl_vector** retainResamplingResults
);


// future interface: a struct holding the information needed for 

struct kernel {
	// the degree
	int m; // degree
	int p; // dimension of the data
	// pointer to the actual function 
	double(*kernel)(const gsl_matrix*);
	// should also hold the information on the subgroup with respect to which it is invariant
};

