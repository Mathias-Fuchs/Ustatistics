
/*
 *  calculates classification loss, and estimates the standard error by a U-statistic
 *  Copyright (C) 2013  Mathias Fuchs
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program. If not, see <http://www.gnu.org/licenses/>.
 */


#include <time.h>

#include <stdio.h>
#include <stdlib.h>
#include <gsl/gsl_vector_double.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_statistics.h>
#include <gsl/gsl_blas.h>

#include <assert.h>


void sampleWithoutReplacement (const size_t populationSize, const size_t sampleSize, size_t * subsample, gsl_rng * r) {

  // Use Knuth's variable names
  int n = sampleSize;
  int N = populationSize;
  
  int t = 0; // total input records dealt with
  int m = 0; // number of items selected so far

  double u;
  while (m < n) {
    u = gsl_rng_uniform_pos(r); // call a uniform(0,1) random number generat
    
    if ((N - t) *  u >= n - m) {
      t++; 
    } else  {
      subsample[m] = t;
      t++; 
      m++;
    }
  }
  // we now have to shuffle the resulting indices, so they are no more sorted
  if (n > 1)  {
    for (int i = 0; i < n - 1; i++) {
      size_t j = i + (gsl_rng_get(r) - gsl_rng_min(r)) / (gsl_rng_max(r) / (n - i) + 1);
      size_t t = subsample[j];
      subsample[j] = subsample[i];
      subsample[i] = t;
    }
  }
}



double myMean(const gsl_vector * x, const size_t until) {
  double sum = 0;
  for (int i = 0; i < until; i++) {
    sum += gsl_vector_get(x, i);
  }
  return sum/until;
}

double myMean2(const gsl_vector * x, const size_t until) {
  double runningAverage = 0;
  for (int i = 0; i < until; i++) {
    runningAverage = i / (i + 1) * runningAverage + gsl_vector_get(x, i) / (i + 1);
  }
  return runningAverage;
}
  

// inversion of symmetric 3-by-3-matrix
gsl_matrix * inv2 (const gsl_matrix * m) {
  assert(m -> size1 == 3);
  assert(m -> size2 == 3);
  gsl_matrix * a = gsl_matrix_alloc(3,3);
  double p = gsl_matrix_get(m, 2, 2) * gsl_matrix_get(m, 1, 1) - gsl_pow_2(gsl_matrix_get(m, 1, 2));
  double q = gsl_matrix_get(m, 2, 2) * gsl_matrix_get(m, 0, 1) - gsl_matrix_get(m, 2, 1) * gsl_matrix_get(m, 0, 2);
  double r = gsl_matrix_get(m, 1, 2) * gsl_matrix_get(m, 0, 1) - gsl_matrix_get(m, 1, 1) * gsl_matrix_get(m, 0, 2);
  double D = gsl_matrix_get(m, 0,0) * p - gsl_matrix_get(m, 1, 0) * q + gsl_matrix_get(m, 2, 0) * r;
  assert(D != 0);
  gsl_matrix_set(a, 0,0, p);
  gsl_matrix_set(a, 0,1, -q);
  gsl_matrix_set(a, 1, 0, -q);
  gsl_matrix_set(a, 0, 2, r);
  gsl_matrix_set(a, 2, 0, r);
  gsl_matrix_set(a, 1, 1, gsl_matrix_get(m, 2, 2) * gsl_matrix_get(m, 0,0) - gsl_pow_2(gsl_matrix_get(m, 0,2)));
  gsl_matrix_set(a, 1, 2, gsl_matrix_get(m, 0, 1) * gsl_matrix_get(m, 0, 2) - gsl_matrix_get(m, 0,0) * gsl_matrix_get(m, 1, 2));
  gsl_matrix_set(a, 2, 1, gsl_matrix_get(a, 1, 2));
  gsl_matrix_set(a, 2, 2, gsl_matrix_get(m, 0,0) * gsl_matrix_get(m, 1, 1) - gsl_pow_2(gsl_matrix_get(m, 0, 1)));
  gsl_matrix_scale(a, 1/D);
  
  return a;
}



// in place inversion of symmetric 3-by-3-matrix
int inv2inPlace(const gsl_matrix * m, gsl_matrix* result) {
	assert(m->size1 == 3);
	assert(m->size2 == 3);
	assert(result->size1 == 3);
	assert(result->size2 == 3);

	double p = gsl_matrix_get(m, 2, 2) * gsl_matrix_get(m, 1, 1) - gsl_pow_2(gsl_matrix_get(m, 1, 2));
	double q = gsl_matrix_get(m, 2, 2) * gsl_matrix_get(m, 0, 1) - gsl_matrix_get(m, 2, 1) * gsl_matrix_get(m, 0, 2);
	double r = gsl_matrix_get(m, 1, 2) * gsl_matrix_get(m, 0, 1) - gsl_matrix_get(m, 1, 1) * gsl_matrix_get(m, 0, 2);
	double D = gsl_matrix_get(m, 0, 0) * p - gsl_matrix_get(m, 1, 0) * q + gsl_matrix_get(m, 2, 0) * r;
	if (D < 0.00001 && D > -0.00001) return 0;
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

	return 1;
}

gsl_matrix * RandomData (size_t n, size_t  p, gsl_rng * r) {
  gsl_matrix * data = gsl_matrix_alloc(n, p);
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < p; j++) {
      gsl_matrix_set(data, i, j, gsl_ran_ugaussian(r));
    }
  }
  return data;
}
		     
gsl_vector * RandomResponse (int n, gsl_rng * r) {
  gsl_vector * res = gsl_vector_alloc(n);
  for (int i = 0; i < n; i++) {
    gsl_vector_set(res, i, gsl_ran_ugaussian(r));
  }
  return res;
}

double meanSquareLoss(double y1, double y2) {
  return gsl_pow_2(y1 - y2);
}

void writeDoubleMatrix(const gsl_matrix * x) {
  int n, p;
  n = x -> size1;
  p = x -> size2;
  int i, j;
  for (i = 0; i < n; i++) {
    for (j = 0; j < p; j++) {
      printf("%f ", (double) gsl_matrix_get (x, i, j));
    }
    printf("\n");
  }
}

void writeDoubleVector(const gsl_vector * x) {
  int n = x -> size;
  for (int i = 0; i < n; i++)
    printf("%f ", gsl_vector_get (x, i));
  printf("\n");
}


/* Gamma is  || betaHat Xtest - Ytest ||^2 = */
/*   (Xtest * (Xlearn^t Xlearn)^(-1) *  Xlearn ^t * Ylearn - Ytest )^2 */



double gamma(const gsl_matrix * data, const gsl_vector * response) {

  //  printf("data has dimensions %d %d , response has length %d \n ", data -> size1, data -> size2, response -> size);
  gsl_matrix_const_view Xlearnview = gsl_matrix_const_submatrix(data, 0, 0, data -> size1 - 1, data -> size2);
  const gsl_matrix * Xlearn = &Xlearnview.matrix;

  gsl_vector_const_view Xtestview = gsl_matrix_const_subrow(data, data -> size1 - 1, 0, data -> size2);
  const gsl_vector * Xtest = &Xtestview.vector;

  gsl_vector_const_view ylearnview = gsl_vector_const_subvector(response, 0, response -> size - 1);
  const gsl_vector * ylearn = &ylearnview.vector;

  //  writeDoubleVector(response);
  double ytest = gsl_vector_get(response, response -> size - 1);

  gsl_matrix * C = gsl_matrix_alloc(3, 3);

  // C will hold results of Xlearn^T Xlearn
  gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, Xlearn, Xlearn, 0.0, C);

  gsl_vector * D = gsl_vector_alloc(3);

  // D will hold results of Xlearn^T ylearn
  gsl_blas_dgemv(CblasTrans, 1.0, Xlearn, ylearn, 0.0, D);
  
  gsl_vector * E = gsl_vector_alloc(3);

  // E will hold the coefficients, namely the results of (Xlearn^T Xlearn)^{-1} * Xlearn^T ylearn
  gsl_matrix * i = inv2(C);


  gsl_blas_dsymv(CblasUpper, 1.0, i, D, 0, E);



  /* printf("coefficients\n"); */
  /* writeDoubleVector(E); */
  /* printf("\n"); */
  gsl_matrix_free(i);
  
  double ypredicted;
  gsl_blas_ddot(Xtest, E, &ypredicted);
  // gsl_vector_set(predStorage, B, ypredicted);
  //printf("true %f predicted %f difference %f \n", ytest, ypredicted, ytest - ypredicted);
  //xprintf("%f\n", gsl_pow_2(ypredicted - ytest));


  gsl_matrix_free(C);
  gsl_vector_free(D);
  gsl_vector_free(E);
  return gsl_pow_2(ypredicted - ytest);
}
  
double kernelForThetaSquared(const gsl_matrix * data, const gsl_vector * response) {
  gsl_matrix_const_view data1view = gsl_matrix_const_submatrix(data, 0, 0, data -> size1 / 2, data -> size2);
  const gsl_matrix * data1 = &data1view.matrix;

  gsl_matrix_const_view data2view = gsl_matrix_const_submatrix(data, data -> size1 / 2, 0, data -> size1 / 2, data -> size2);
  const gsl_matrix * data2 = &data2view.matrix;

  gsl_vector_const_view response1view = gsl_vector_const_subvector(response, 0, response -> size / 2);
  const gsl_vector * response1 = &response1view.vector;

  gsl_vector_const_view response2view = gsl_vector_const_subvector(response, response -> size / 2, response -> size / 2);
  const gsl_vector * response2 = &response2view.vector;

  return gamma(data1, response1) * gamma(data2, response2);
}




/* computes a U-statistic of degree m by resampling B times */
double U(const gsl_matrix * data, const gsl_vector * response, const size_t B, const int m, gsl_rng * r, double (* kernel)(const gsl_matrix *, const gsl_vector *)) {
  int n = data -> size1;
  int p = data -> size2;
  assert(response -> size == n);

  int b;

  gsl_vector * resamplingResults = gsl_vector_alloc(B);


  gsl_matrix * subsample = gsl_matrix_alloc(m, p);
  gsl_vector * subresponse = gsl_vector_alloc(m);
  gsl_vector * predStorage = gsl_vector_alloc(B);


  // will hold the indices of a subsample
  size_t * indices = (size_t *) malloc(m * sizeof(size_t));
  
  for (b = 0; b < B; b++) {
    sampleWithoutReplacement(n, m, indices, r); 
    for (int i = 0; i < m; i++) {
      // printf("%i ", indices[i]);
      gsl_vector_set(subresponse, i, gsl_vector_get(response, indices[i]));
      for (int j = 0; j < p; j++) {
	gsl_matrix_set(subsample, i, j, gsl_matrix_get(data, indices[i], j));
      }
    }

    gsl_vector_set(resamplingResults, b,  kernel(subsample, subresponse));

    if (b % 2 == 2) {
      printf("%f %f \n", b + 0.0, myMean(resamplingResults, b));
    }
  }
  free(indices);
  gsl_matrix_free(subsample);
  gsl_vector_free(subresponse);

  double x = myMean(resamplingResults, B);
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



int main (int argc, char ** argv) {

  
  size_t n = 103;
  size_t p = 3;
  int B = 1e6;
  int seed = 1234;
  int g = 40;

  if (argc != 4) {
    fprintf(stdout, "need 3 command line arguments: the number of resample in each iteration, a random seed, and the learning set size (needs to be between 3 and 102\n");
  }


  /*
  sscanf(argv[1], "number of resamples in each iteration: %i", &B);
  
  sscanf(argv[2], "random seed: %i", &seed);
  int gg = 0; 
  sscanf(argv[3], "learning set size: %i", &gg);
  size_t g = (size_t) gg;  
  */

  size_t Br = (size_t)B;
  gsl_matrix * X = gsl_matrix_alloc(n, p);
  FILE * f = fopen("slump.dat", "rb");

  if (!f) {
	  fprintf(stderr, "input file not found!\n");
  }

  int h = gsl_matrix_fscanf(f, X);
  fclose(f);

  gsl_vector * y = gsl_vector_alloc(103);
  f = fopen("slumpResponse.dat", "rb");

  if (!f) {
	  fprintf(stderr, "response input file not found!\n");
  }
  gsl_matrix * dummy = gsl_matrix_alloc(1, 103);
  gsl_matrix_fscanf(f, dummy);
  fclose(f);

  for (int i = 0; i < 103; i++) {
    gsl_vector_set(y, i, gsl_matrix_get(dummy, 0, i));
  }

  gsl_matrix_free(dummy);


  gsl_rng * r = gsl_rng_alloc(gsl_rng_taus2);
  gsl_rng_set(r, seed);


  double lpo =  U(X, y, Br, g+1, r, &gamma);
  double t2 =  U(X, y, Br, 2 * g + 2, r, &kernelForThetaSquared);
    
  gsl_matrix_free(X);
  gsl_vector_free(y);

  printf("%f %f %f\n", lpo, t2, gsl_pow_2(lpo) - t2);

  return 0;
}

