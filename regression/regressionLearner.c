#include "regressionLearner.h"
#include <gsl/gsl_matrix_double.h>
#include <gsl/gsl_vector_double.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_statistics.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_permutation.h>
#include <assert.h>


#ifdef RELEASE

#define HAVE_INLINE
#undef GSL_RANGE_CHECK

#endif


typedef struct {
	gsl_matrix* ii;
	// CC will hold results of Xlearn^T Xlearn
	gsl_matrix* CC;
	// DD will hold results of Xlearn^T ylearn
	gsl_vector* DD;
	// EE will hold the coefficients, namely the results of (Xlearn^T Xlearn)^{-1} * Xlearn^T ylearn
	gsl_vector* EE;

} regressionLearnerWorkspace;

// a global variable but only in this file
static regressionLearnerWorkspace ws;


void workspaceInit(size_t p) {
	ws.CC = gsl_matrix_alloc(p, p);
	ws.DD = gsl_vector_alloc(p);
	ws.EE = gsl_vector_alloc(p);
	ws.ii = gsl_matrix_alloc(p, p);
}

void workspaceDel() {
	gsl_matrix_free(ws.CC);
	gsl_vector_free(ws.DD);
	gsl_vector_free(ws.EE);
	gsl_matrix_free(ws.ii);
}

static inline double meanSquareLoss(double y1, double y2) { return gsl_pow_2(y1 - y2); }
static inline double boundedMeanSquareLoss(double y1, double y2) { return atan((y1 - y2) * (y1 - y2)) * 2.0 / M_PI; }

// inversion of symmetric 3-by-3-matrix
static inline void inv2inPlace(const gsl_matrix* m, gsl_matrix* result) {
	assert(m->size1 == 3);
	assert(m->size2 == 3);
	assert(result->size1 == 3);
	assert(result->size2 == 3);

	double p = gsl_matrix_get(m, 2, 2) * gsl_matrix_get(m, 1, 1) - gsl_pow_2(gsl_matrix_get(m, 1, 2));
	double q = gsl_matrix_get(m, 2, 2) * gsl_matrix_get(m, 0, 1) - gsl_matrix_get(m, 2, 1) * gsl_matrix_get(m, 0, 2);
	double r = gsl_matrix_get(m, 1, 2) * gsl_matrix_get(m, 0, 1) - gsl_matrix_get(m, 1, 1) * gsl_matrix_get(m, 0, 2);
	double D = gsl_matrix_get(m, 0, 0) * p - gsl_matrix_get(m, 1, 0) * q + gsl_matrix_get(m, 2, 0) * r;
	assert(!(D < 0.0001 && D > -0.0001));
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

// remains to make symmetric
double kernelTheta(const gsl_matrix* data) {
	int g = data->size1 - 1;
	int p = data->size2 - 1;

	// learn data of sizes g * p, and g  * 1 resp
	gsl_matrix_const_view learnX = gsl_matrix_const_submatrix(data, 0, 0, g, p);
	gsl_vector_const_view learnY = gsl_matrix_const_subcolumn(data, p, 0, g);

	// test data of sizes 1 * p, and 1 * 1 resp
	gsl_vector_const_view testX = gsl_matrix_const_subrow(data, g, 0, p);
	const double testY = gsl_matrix_get(data, g, p);

	gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, &learnX.matrix, &learnX.matrix, 0.0, ws.CC);
	gsl_blas_dgemv(CblasTrans, 1.0, &learnX.matrix, &learnY.vector, 0.0, ws.DD);
	inv2inPlace(ws.CC, ws.ii);

	gsl_blas_dsymv(CblasUpper, 1.0, ws.ii, ws.DD, 0, ws.EE);

	/* printf("coefficients\n"); */
	/* writeDoubleVector(E); */
	/* printf("\n"); */

	double ypredicted;
	gsl_blas_ddot(&testX.vector, ws.EE, &ypredicted);

	// gsl_vector_set(predStorage, B, ypredicted);
	//printf("true %f predicted %f difference %f \n", ytest, ypredicted, ytest - ypredicted);
	//xprintf("%f\n", gsl_pow_2(ypredicted - ytest));

	return boundedMeanSquareLoss(ypredicted, testY);
}


double symmetrizedKernelTheta(const gsl_matrix* data) {
	int g = data->size1 - 1;
	int p = data->size2 - 1;
	gsl_permutation* cyclicPerm = gsl_permutation_alloc(g + 1);
	// generate a cyclic permutation that loops around by one
	for (int j = 0; j < g + 1; j++) cyclicPerm->data[j] = (j + 1) % (g + 1);
	

	gsl_matrix* datacopy = gsl_matrix_alloc(g + 1, p + 1);
	gsl_matrix* datacopyTransposed = gsl_matrix_alloc(p + 1, g + 1);
	
	//for (int i = 0; i < g + 1; i++)
	//	for (int j = 0; j < p + 1; j++)
	//		gsl_matrix_set(datacopyTransposed, j, i, gsl_matrix_get(data, i, j));
	//
	gsl_matrix_transpose_memcpy(datacopyTransposed, data);

	double res = 0;
	for (int i = 0; i < g + 1; i++) {
		// ok, permutation has now been set up.
		// we now compute the kernel on the permutated data. We have to permute the rows with the cyclic permutation.

		gsl_permute_matrix(cyclicPerm, datacopyTransposed);
		gsl_matrix_transpose_memcpy(datacopy, datacopyTransposed);
		// now, datacopy contains the permuted data;
		res += kernelTheta(datacopy) / (double)(g + 1);

	}

	gsl_matrix_free(datacopy);
	gsl_matrix_free(datacopyTransposed);
	gsl_permutation_free(cyclicPerm);
	// fprintf(stdout, "%f \n", res);
	return res;
}



// remains to make symmetric
double kernelForThetaSquared(const gsl_matrix* data) {
	assert(data->size1 % 2 == 0);

	// the first half of the rows gets fed into the first gamma, and the second into the other
	gsl_matrix_const_view data1 = gsl_matrix_const_submatrix(data, 0, 0, data->size1 / 2, data->size2);
	gsl_matrix_const_view data2 = gsl_matrix_const_submatrix(data, data->size1 / 2, 0, data->size1 / 2, data->size2);
	return symmetrizedKernelTheta(&data1.matrix) * symmetrizedKernelTheta(&data2.matrix);
}

double kernelOverlapOne(const gsl_matrix* data) {
	// we take two subsets of the data of overlap one, and apply the symmetrized kernel on each.
	int g = (data->size1 - 1) / 2; // there have to be 2g + 1 rows to make this function work.
	int p = data->size2 - 1;

	// the first half of the rows gets fed into the first gamma, and the second into the other
	gsl_matrix_const_view data1 = gsl_matrix_const_submatrix(data, 0, 0, g+1, p+1);
	gsl_matrix_const_view data2 = gsl_matrix_const_submatrix(data, g, 0, g+1, p+1); // this is the gist: we start at g, not g+1, so we obtain an overlap size of one.

	return symmetrizedKernelTheta(&data1.matrix) * symmetrizedKernelTheta(&data2.matrix);
}
