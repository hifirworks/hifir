//@HEADER
//----------------------------------------------------------------------------
//                Copyright (C) 2019 The PSMILU AUTHORS
//----------------------------------------------------------------------------
//@HEADER

/// \file psmilu_lapack/qrcp.hpp
/// \brief Interface wrappers for QR with column pivoting routines in LAPACK
/// \authors Qiao,
/// \todo add interfaces for complex types

#ifndef _PSMILU_LAPACK_QRCP_HPP
#define _PSMILU_LAPACK_QRCP_HPP

#include "psmilu_lapack/config.h"

#ifndef DOXYGEN_SHOULD_SKIP_THIS

extern "C" {

// factorization

// double
void PSMILU_FC(dgeqp3, DGEQP3)(psmilu_lapack_int *, psmilu_lapack_int *,
                               double *, psmilu_lapack_int *,
                               psmilu_lapack_int *, double *, double *,
                               psmilu_lapack_int *, psmilu_lapack_int *);

// float
void PSMILU_FC(sgeqp3, SGEQP3)(psmilu_lapack_int *, psmilu_lapack_int *,
                               float *, psmilu_lapack_int *,
                               psmilu_lapack_int *, float *, float *,
                               psmilu_lapack_int *, psmilu_lapack_int *);

// triangle cond estimator

// double
void PSMILU_FC(dtrcon, DTRCON)(char *, char *, char *, psmilu_lapack_int *,
                               double *, psmilu_lapack_int *, double *,
                               double *, psmilu_lapack_int *,
                               psmilu_lapack_int *);

// single
void PSMILU_FC(strcon, STRCON)(char *, char *, char *, psmilu_lapack_int *,
                               float *, psmilu_lapack_int *, float *, float *,
                               psmilu_lapack_int *, psmilu_lapack_int *);

// Householder reflector multiplication
void PSMILU_FC(dormqr, DORMQR)(char *, char *, psmilu_lapack_int *,
                               psmilu_lapack_int *, psmilu_lapack_int *,
                               double *, psmilu_lapack_int *, double *,
                               double *c, psmilu_lapack_int *, double *,
                               psmilu_lapack_int *, psmilu_lapack_int *);

// single
void PSMILU_FC(sormqr, SORMQR)(char *, char *, psmilu_lapack_int *,
                               psmilu_lapack_int *, psmilu_lapack_int *,
                               float *, psmilu_lapack_int *, float *, float *c,
                               psmilu_lapack_int *, float *,
                               psmilu_lapack_int *, psmilu_lapack_int *);
}

#endif  // DOXYGEN_SHOULD_SKIP_THIS

namespace psmilu {
namespace internal {

/// \brief double version QR with column pivoting
inline psmilu_lapack_int geqp3(const psmilu_lapack_int m,
                               const psmilu_lapack_int n, double *a,
                               const psmilu_lapack_int lda,
                               psmilu_lapack_int *jpvt, double *tau,
                               double *work, const psmilu_lapack_int lwork) {
  psmilu_lapack_int info;
  PSMILU_FC(dgeqp3, DGEQP3)
  ((psmilu_lapack_int *)&m, (psmilu_lapack_int *)&n, a,
   (psmilu_lapack_int *)&lda, jpvt, tau, work, (psmilu_lapack_int *)&lwork,
   &info);
  return info;
}

/// \brief single version QR with column pivoting
inline psmilu_lapack_int geqp3(const psmilu_lapack_int m,
                               const psmilu_lapack_int n, float *a,
                               const psmilu_lapack_int lda,
                               psmilu_lapack_int *jpvt, float *tau, float *work,
                               const psmilu_lapack_int lwork) {
  psmilu_lapack_int info;
  PSMILU_FC(sgeqp3, SGEQP3)
  ((psmilu_lapack_int *)&m, (psmilu_lapack_int *)&n, a,
   (psmilu_lapack_int *)&lda, jpvt, tau, work, (psmilu_lapack_int *)&lwork,
   &info);
  return info;
}

/// \brief double version triangular condition number estimator
/// \note This is needed to determined the rank for QRCP, i.e. TQRCP
inline psmilu_lapack_int trcon(const char norm, const char uplo,
                               const char diag, const psmilu_lapack_int n,
                               const double *a, const psmilu_lapack_int lda,
                               double &rcond, double *work,
                               psmilu_lapack_int *iwork) {
  psmilu_lapack_int info;
  PSMILU_FC(dtrcon, DTRCON)
  ((char *)&norm, (char *)&uplo, (char *)&diag, (psmilu_lapack_int *)&n,
   (double *)a, (psmilu_lapack_int *)&lda, &rcond, work, iwork, &info);
  return info;
}

/// \brief single version triangular condition number estimator
/// \note This is needed to determined the rank for QRCP, i.e. TQRCP
inline psmilu_lapack_int trcon(const char norm, const char uplo,
                               const char diag, const psmilu_lapack_int n,
                               const float *a, const psmilu_lapack_int lda,
                               float &rcond, float *work,
                               psmilu_lapack_int *iwork) {
  psmilu_lapack_int info;
  PSMILU_FC(strcon, STRCON)
  ((char *)&norm, (char *)&uplo, (char *)&diag, (psmilu_lapack_int *)&n,
   (float *)a, (psmilu_lapack_int *)&lda, &rcond, work, iwork, &info);
  return info;
}

/// \brief double version for handling Householder reflectors
inline psmilu_lapack_int ormqr(const char side, const char trans,
                               const psmilu_lapack_int m,
                               const psmilu_lapack_int n,
                               const psmilu_lapack_int k, const double *a,
                               const psmilu_lapack_int lda, const double *tau,
                               double *c, const psmilu_lapack_int ldc,
                               double *work, const psmilu_lapack_int lwork) {
  psmilu_lapack_int info;
  PSMILU_FC(dormqr, DORMQR)
  ((char *)&side, (char *)&trans, (psmilu_lapack_int *)&m,
   (psmilu_lapack_int *)&n, (psmilu_lapack_int *)&k, (double *)a,
   (psmilu_lapack_int *)&lda, (double *)tau, c, (psmilu_lapack_int *)&ldc, work,
   (psmilu_lapack_int *)&lwork, &info);
  return info;
}

/// \brief single version for handling Householder reflectors
inline psmilu_lapack_int ormqr(const char side, const char trans,
                               const psmilu_lapack_int m,
                               const psmilu_lapack_int n,
                               const psmilu_lapack_int k, const float *a,
                               const psmilu_lapack_int lda, const float *tau,
                               float *c, const psmilu_lapack_int ldc,
                               float *work, const psmilu_lapack_int lwork) {
  psmilu_lapack_int info;
  PSMILU_FC(sormqr, SORMQR)
  ((char *)&side, (char *)&trans, (psmilu_lapack_int *)&m,
   (psmilu_lapack_int *)&n, (psmilu_lapack_int *)&k, (float *)a,
   (psmilu_lapack_int *)&lda, (float *)tau, c, (psmilu_lapack_int *)&ldc, work,
   (psmilu_lapack_int *)&lwork, &info);
  return info;
}

}  // namespace internal
}  // namespace psmilu

#endif  // _PSMILU_LAPACK_QRCP_HPP