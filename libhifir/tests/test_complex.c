/*
///////////////////////////////////////////////////////////////////////////////
//                  This file is part of the HIFIR library                   //
///////////////////////////////////////////////////////////////////////////////
*/

/*

    This file contains a simple example of using the complex double APIs of
    libhifir library. In particular, we access the multilevel triangular solve
    and matrix-vector multiplication functions. Note that these functions
    serve as the core in KSP solvers.

    Author: Qiao Chen
    Level: Beginner

*/

#include <assert.h>
#include <complex.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "libhifir.h"

#define ROW_MAJOR 1

static void compute_A1(const size_t n, const LhfInt *rowptr,
                       const double _Complex *vals, double _Complex *b);

static double compute_error(const size_t n, const double _Complex *ref,
                            const double _Complex *num);

int main() {
  /* variables */
  LhfInt *         rowptr, *colind;
  double _Complex *vals, *b, *x, *b2;
  int              is_sparse, is_real;
  double           err;
  size_t           nrows, ncols, nnz;
  LhfStatus        info;
  LhfzMatrixHdl    A;
  LhfzHifHdl       M;

  /* setup parameteres */
  double params[LHF_NUMBER_PARAMS];
  lhfSetDefaultParams(params);
  /* customize different parameters */
  params[LHF_VERBOSE] = LHF_VERBOSE_NULL;

  /* create matrix */
  if (lhfQueryMmFile("../../examples/demo_inputs/young1c.mtx", &is_sparse,
                     &is_real, &nrows, &ncols, &nnz) != LHF_SUCCESS) {
    fprintf(stderr,
            "Failed querying information from "
            "\"../../examples/demo_inputs/young1c.mtx\".\n");
    return (1);
  }
  if (nrows != ncols || !is_sparse || is_real) {
    fprintf(stderr, "Incorrect matrix data.\n");
    return (1);
  }

  /* allocate data */
  rowptr = (LhfInt *)malloc((nrows + 1) * sizeof(LhfInt));
  colind = (LhfInt *)malloc(nnz * sizeof(LhfInt));
  vals   = (double _Complex *)malloc(nnz * sizeof(double _Complex));
  printf("Successfully created data arrays for the input matrix.\n");
  A = lhfzCreateMatrix(ROW_MAJOR, nrows, rowptr, colind, vals);
  if (!A) {
    fprintf(stderr, "Failed to create matrix handle.\n");
    return (1);
  }
  lhfzWrapMatrix(A, nrows, rowptr, colind, vals);
  /* loading matrix */
  info = lhfzReadSparse("../../examples/demo_inputs/young1c.mtx", A);
  if (info != LHF_SUCCESS) {
    fprintf(stderr,
            "Failed to read sparse data from "
            "\"../../examples/demo_inputs/young1c.mtx\".\n");
    if (info == LHF_HIFIR_ERROR) fprintf(stderr, "%s", lhfGetErrorMsg());
    return (1);
  }
  printf("Successfully loaded data from file for the input matrix.\n");

  /* setup the preconditioner */
  printf("Beginning factorizing HIF preconditioner...\n");
  M = lhfzCreate(A, NULL, params);
  if (!M) {
    fprintf(stderr, "Failed factorizing HIF.\n");
    return (1);
  }
  printf("Successfully factorizing HIF.\n");

  /* RHS = A*1 */
  b = (double _Complex *)malloc(nrows * sizeof(double _Complex));
  x = (double _Complex *)malloc(nrows * sizeof(double _Complex));
  compute_A1(nrows, rowptr, vals, b);

  /* multilevel triangular solve */
  printf("Performing multilevel triangular solve x=M\\b, where b=A*1.\n");
  info = lhfzSolve(M, b, x);
  if (info != LHF_SUCCESS) {
    fprintf(stderr, "Failed to compute x=M\\b.\n");
    if (info == LHF_HIFIR_ERROR) fprintf(stderr, "%s", lhfGetErrorMsg());
    return (1);
  }
  printf("Successfully finished computing x=M\\b.\n");

  /* multilevel matrix-vector multiplication */
  printf("Performing multilevel matrix-vector product b2=M*x.\n");
  b2   = (double _Complex *)malloc(nrows * sizeof(double _Complex));
  info = lhfzApply(M, LHF_M, x, 1, NULL, LHF_DEFAULT_RANK, b2, NULL);
  if (info != LHF_SUCCESS) {
    fprintf(stderr, "Failed to compute b2=M*x.\n");
    if (info == LHF_HIFIR_ERROR) fprintf(stderr, "%s", lhfGetErrorMsg());
    return (1);
  }
  printf("Successfully finished computing b2=M*x.\n");

  /* error analysis */
  printf("Performing error analysis.\n");
  err = compute_error(nrows, b, b2);
  printf("norm2(b2-b)/norm2(b)=%g.\n", err);

  /* cleanup */
  printf("Cleaning up memory allocations... ");
  free(rowptr);
  free(colind);
  free(vals);
  lhfzDestroyMatrix(A);
  lhfzDestroy(M);
  free(b);
  free(x);
  free(b2);
  printf("And done!\n");

  assert((err <= 1e-10) && "Relative error is too large!");

  return (0);
}

static void compute_A1(const size_t n, const LhfInt *rowptr,
                       const double _Complex *vals, double _Complex *b) {
  /* compute b = A*1 */
  double _Complex v;
  size_t i;
  LhfInt k;
  for (i = 0u; i < n; ++i) {
    v = 0.0 + 0.0 * _Complex_I;
    for (k = rowptr[i]; k < rowptr[i + 1]; ++k) v += vals[k];
    b[i] = v;
  }
}

static double compute_error(const size_t n, const double _Complex *ref,
                            const double _Complex *num) {
  double _Complex *buf;
  double           err, nrm;
  size_t           i;
  buf = (double _Complex *)malloc(n * sizeof(double _Complex));
  /* compute num-ref */
  for (i = 0; i < n; ++i) buf[i] = num[i] - ref[i];
  /* compute norm2(ref) and norm2(buf) */
  err = nrm = 0.0;
  for (i = 0; i < n; ++i) {
    nrm += creal(ref[i]) * creal(ref[i]) + cimag(ref[i]) * cimag(ref[i]);
    err += creal(buf[i]) * creal(buf[i]) + cimag(buf[i]) * cimag(buf[i]);
  }
  free(buf);
  return sqrt(err / nrm);
}
