/*
///////////////////////////////////////////////////////////////////////////////
//                  This file is part of the HIFIR library                   //
///////////////////////////////////////////////////////////////////////////////
*/

/*

    This file contains a simple example of using the real double APIs of
    libhifir library. In particular, we access the multilevel triangular solve
    and matrix-vector multiplication functions. Note that these functions
    serve as the core in KSP solvers.

    Author: Qiao Chen
    Level: Beginner

*/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "libhifir.h"

#define ROW_MAJOR 1

static double compute_error(const size_t n, const double *ref,
                            const double *num);

int main() {
  /* variables */
  LhfInt *      rowptr, *colind;
  double *      vals, *b, *x, *b2;
  int           is_sparse, is_real;
  double        err;
  size_t        nrows, ncols, nnz;
  LhfStatus     info;
  LhfdMatrixHdl A;
  LhfdHifHdl    M;

  /* setup parameteres */
  double params[LHF_NUMBER_PARAMS];
  lhfSetDefaultParams(params);
  /* customize different parameters */
  lhfSetDroptol(1e-2, params);
  lhfSetAlpha(3.0, params);
  lhfSetKappa(5.0, params);
  params[LHF_VERBOSE] = LHF_VERBOSE_NULL;

  /* create matrix */
  if (lhfQueryMmFile("../demo_inputs/A.mm", &is_sparse, &is_real, &nrows,
                     &ncols, &nnz) != LHF_SUCCESS) {
    fprintf(stderr,
            "Failed querying information from \"../demo_inputs/A.mm\".\n");
    return (1);
  }
  if (nrows != ncols || !is_sparse || !is_real) {
    fprintf(stderr, "Incorrect matrix data.\n");
    return (1);
  }
  /* allocate data */
  rowptr = (LhfInt *)malloc((nrows + 1) * sizeof(LhfInt));
  colind = (LhfInt *)malloc(nnz * sizeof(LhfInt));
  vals   = (double *)malloc(nnz * sizeof(double));
  printf("Successfully created data arrays for the input matrix.\n");
  A = lhfdCreateMatrix(ROW_MAJOR, nrows, rowptr, colind, vals);
  if (!A) {
    fprintf(stderr, "Failed to create matrix handle.\n");
    return (1);
  }
  lhfdWrapMatrix(A, nrows, rowptr, colind, vals);
  /* loading matrix */
  info = lhfdReadSparse("../demo_inputs/A.mm", A);
  if (info != LHF_SUCCESS) {
    fprintf(stderr,
            "Failed to read sparse data from \"../demo_inputs/A.mm\".\n");
    if (info == LHF_HIFIR_ERROR) fprintf(stderr, "%s", lhfGetErrorMsg());
    return (1);
  }
  printf("Successfully loaded data from file for the input matrix.\n");

  /* setup the preconditioner */
  printf("Beginning factorizing HIF preconditioner...\n");
  M = lhfdCreate(A, NULL, params);
  if (!M) {
    fprintf(stderr, "Failed factorizing HIF.\n");
    return (1);
  }
  printf("Successfully factorizing HIF.\n");

  /* load RHS */
  b = (double *)malloc(nrows * sizeof(double));
  x = (double *)malloc(nrows * sizeof(double));
  /* loading vector */
  info = lhfdReadVector("../demo_inputs/b.mm", nrows, b);
  if (info != LHF_SUCCESS) {
    fprintf(stderr,
            "Failed to read sparse data from \"../demo_inputs/A.mm\".\n");
    if (info == LHF_HIFIR_ERROR) fprintf(stderr, "%s", lhfGetErrorMsg());
    return (1);
  }

  /* multilevel triangular solve */
  printf("Performing multilevel triangular solve x=M\\b.\n");
  info = lhfdSolve(M, b, x);
  if (info != LHF_SUCCESS) {
    fprintf(stderr, "Failed to compute x=M\\b.\n");
    if (info == LHF_HIFIR_ERROR) fprintf(stderr, "%s", lhfGetErrorMsg());
    return (1);
  }
  printf("Successfully finished computing x=M\\b.\n");

  /* multilevel matrix-vector multiplication */
  printf("Performing multilevel matrix-vector product b2=M*x.\n");
  b2   = (double *)malloc(nrows * sizeof(double));
  info = lhfdApply(M, LHF_M, x, 1, NULL, LHF_DEFAULT_RANK, b2, NULL);
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
  lhfdDestroyMatrix(A);
  lhfdDestroy(M);
  free(b);
  free(x);
  free(b2);
  printf("And done!\n");

  return (0);
}

static double compute_error(const size_t n, const double *ref,
                            const double *num) {
  double *buf, err, nrm;
  size_t  i;
  buf = (double *)malloc(n * sizeof(double));
  /* compute num-ref */
  for (i = 0; i < n; ++i) buf[i] = num[i] - ref[i];
  /* compute norm2(ref) and norm2(buf) */
  err = nrm = 0.0;
  for (i = 0; i < n; ++i) {
    nrm += ref[i] * ref[i];
    err += buf[i] * buf[i];
  }
  free(buf);
  return sqrt(err / nrm);
}
