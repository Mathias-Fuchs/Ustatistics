#include "regressionLearner.h"

#include <gsl/gsl_math.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_statistics.h>
#include <gsl/gsl_blas.h>
#include <assert.h>


#ifdef RELEASE

#define HAVE_INLINE
#undef GSL_RANGE_CHECK

#endif


typedef struct {
	gsl_matrix * ii;
	// CC will hold results of Xlearn^T Xlearn
	gsl_matrix * CC;
	// DD will hold results of Xlearn^T ylearn
	gsl_vector * DD;
	// EE will hold the coefficients, namely the results of (Xlearn^T Xlearn)^{-1} * Xlearn^T ylearn
	gsl_vector * EE;
} regressionLearnerWorkspace;

// static gsl_matrix* ii;
// static gsl_matrix* CC;
// static gsl_vector* DD;
// static gsl_vector* EE;


static regressionLearnerWorkspace ws;

void workspaceInit(size_t p) {
	ws.CC = gsl_matrix_alloc(p, p);
	ws.DD = gsl_vector_alloc(p);
	ws.EE = gsl_vector_alloc(p);
	ws.ii = gsl_matrix_alloc(p, p);

	// CC = gsl_matrix_alloc(p, p);
	// DD = gsl_vector_alloc(p);
	// EE = gsl_vector_alloc(p);
	// ii = gsl_matrix_alloc(p, p);

}

void workspaceDel() {
	gsl_matrix_free(ws.CC);
	gsl_vector_free(ws.DD);
	gsl_vector_free(ws.EE);
	gsl_matrix_free(ws.ii);
}

static inline double meanSquareLoss(double y1, double y2) {
	return gsl_pow_2(y1 - y2);
}

// inversion of symmetric 3-by-3-matrix
static inline void inv2inPlace(const gsl_matrix * m, gsl_matrix* result) {
	assert(m->size1 == 3);
	assert(m->size2 == 3);
	assert(result->size1 == 3);
	assert(result->size2 == 3);

	double p = gsl_matrix_get(m, 2, 2) * gsl_matrix_get(m, 1, 1) - gsl_pow_2(gsl_matrix_get(m, 1, 2));
	double q = gsl_matrix_get(m, 2, 2) * gsl_matrix_get(m, 0, 1) - gsl_matrix_get(m, 2, 1) * gsl_matrix_get(m, 0, 2);
	double r = gsl_matrix_get(m, 1, 2) * gsl_matrix_get(m, 0, 1) - gsl_matrix_get(m, 1, 1) * gsl_matrix_get(m, 0, 2);
	double D = gsl_matrix_get(m, 0, 0) * p - gsl_matrix_get(m, 1, 0) * q + gsl_matrix_get(m, 2, 0) * r;
	assert (!(D < 0.0001 && D > -0.0001));
	gsl_matrix_set(result, 0, 0, p);
	gsl_matrix_set(result, 0, 1, -q);
	gsl_matrix_set(result, 1, 0, -q);
	gsl_matrix_set(result, 0, 2, r);
	gsl_matrix_set(result, 2, 0, r);
	gsl_matrix_set(result, 1, 1, gsl_matrix_get(m, 2, 2) * gsl_matrix_get(m, 0, 0) - gsl_pow_2(gsl_matrix_get(m, 0, 2)));
	gsl_matrix_set(result, 1, 2, gsl_matrix_get(m, 0, 1) * gsl_matrix_get(m, 0, 2) - gsl_matrix_get(m, 0, 0) * gsl_matrix_get(m, 1, 2));
	gsl_matrix_set(result, 2, 1, gsl_matrix_get(result, 1, 2));
	gsl_matrix_set(result, 2, 2, gsl_matrix_get(m, 0, 0) * gsl_matrix_get(m, 1, 1) - gsl_pow_2(gsl_matrix_get(m, 0, 1)));
	gsl_matrix_scale(result, 1 / D);
	return;
}


/* Gamma is  || betaHat Xtest - Ytest ||^2 = */
/*   (Xtest * (Xlearn^t Xlearn)^(-1) *  Xlearn ^t * Ylearn - Ytest )^2 */

double gamma(const gsl_matrix * data, const gsl_vector * response) {

	//  printf("data has dimensions %d %d , response has length %d \n ", data -> size1, data -> size2, response -> size);
	gsl_matrix_const_view Xlearnview = gsl_matrix_const_submatrix(data, 0, 0, data->size1 - 1, data->size2);
	const gsl_matrix * Xlearn = &Xlearnview.matrix;

	gsl_vector_const_view Xtestview = gsl_matrix_const_subrow(data, data->size1 - 1, 0, data->size2);
	const gsl_vector * Xtest = &Xtestview.vector;

	gsl_vector_const_view ylearnview = gsl_vector_const_subvector(response, 0, response->size - 1);
	const gsl_vector * ylearn = &ylearnview.vector;

	//  writeDoubleVector(response);
	double ytest = gsl_vector_get(response, response->size - 1);

	gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, Xlearn, Xlearn, 0.0, ws.CC);
	gsl_blas_dgemv(CblasTrans, 1.0, Xlearn, ylearn, 0.0, ws.DD);
	inv2inPlace(ws.CC, ws.ii);


	gsl_blas_dsymv(CblasUpper, 1.0, ws.ii, ws.DD, 0, ws.EE);



	/* printf("coefficients\n"); */
	/* writeDoubleVector(E); */
	/* printf("\n"); */

	double ypredicted;
	gsl_blas_ddot(Xtest, ws.EE, &ypredicted);

	// gsl_vector_set(predStorage, B, ypredicted);
	//printf("true %f predicted %f difference %f \n", ytest, ypredicted, ytest - ypredicted);
	//xprintf("%f\n", gsl_pow_2(ypredicted - ytest));

	return meanSquareLoss(ypredicted, ytest);
}

double kernelForThetaSquared(const gsl_matrix * data, const gsl_vector * response) {
	gsl_matrix_const_view data1view = gsl_matrix_const_submatrix(data, 0, 0, data->size1 / 2, data->size2);
	const gsl_matrix * data1 = &data1view.matrix;

	gsl_matrix_const_view data2view = gsl_matrix_const_submatrix(data, data->size1 / 2, 0, data->size1 / 2, data->size2);
	const gsl_matrix * data2 = &data2view.matrix;

	gsl_vector_const_view response1view = gsl_vector_const_subvector(response, 0, response->size / 2);
	const gsl_vector * response1 = &response1view.vector;

	gsl_vector_const_view response2view = gsl_vector_const_subvector(response, response->size / 2, response->size / 2);
	const gsl_vector * response2 = &response2view.vector;

	return gamma(data1, response1) * gamma(data2, response2);
}