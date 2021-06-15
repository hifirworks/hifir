///////////////////////////////////////////////////////////////////////////////
//                  This file is part of the HIFIR library                   //
///////////////////////////////////////////////////////////////////////////////

/*!
 * \file hif/small_scale/lapackf.hpp
 * \brief Low-level Fortran-based LAPACK interfaces
 * \author Qiao Chen

\verbatim
Copyright (C) 2021 NumGeom Group at Stony Brook University

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
\endverbatim

 */

#ifndef _HIF_SMALLSCALE_LAPACKF_HPP
#define _HIF_SMALLSCALE_LAPACKF_HPP

#include <complex>

#include "hif/small_scale/config.hpp"

#ifndef DOXYGEN_SHOULD_SKIP_THIS
#  if HIF_HAS_MKL == 0
extern "C" {

///////////////////////
// LU
///////////////////////

// double
void HIF_FC(dgetrf, DGETRF)(hif_lapack_int *, hif_lapack_int *, double *,
                            hif_lapack_int *, hif_lapack_int *,
                            hif_lapack_int *);
// single
void HIF_FC(sgetrf, SGETRF)(hif_lapack_int *, hif_lapack_int *, float *,
                            hif_lapack_int *, hif_lapack_int *,
                            hif_lapack_int *);

// complex double
void HIF_FC(zgetrf, ZGETRF)(hif_lapack_int *, hif_lapack_int *, void *,
                            hif_lapack_int *, hif_lapack_int *,
                            hif_lapack_int *);

// complex single
void HIF_FC(cgetrf, CGETRF)(hif_lapack_int *, hif_lapack_int *, void *,
                            hif_lapack_int *, hif_lapack_int *,
                            hif_lapack_int *);

// solving

// double
void HIF_FC(dgetrs, DGETRS)(char *, hif_lapack_int *, hif_lapack_int *,
                            double *, hif_lapack_int *, hif_lapack_int *,
                            double *, hif_lapack_int *, hif_lapack_int *);
// single
void HIF_FC(sgetrs, SGETRS)(char *, hif_lapack_int *, hif_lapack_int *, float *,
                            hif_lapack_int *, hif_lapack_int *, float *,
                            hif_lapack_int *, hif_lapack_int *);

// complex double
void HIF_FC(zgetrs, ZGETRS)(char *, hif_lapack_int *, hif_lapack_int *, void *,
                            hif_lapack_int *, hif_lapack_int *, void *,
                            hif_lapack_int *, hif_lapack_int *);

// complex single
void HIF_FC(cgetrs, CGETRS)(char *, hif_lapack_int *, hif_lapack_int *, void *,
                            hif_lapack_int *, hif_lapack_int *, void *,
                            hif_lapack_int *, hif_lapack_int *);

///////////////////////
// QRCP
///////////////////////

// double
void HIF_FC(dgeqp3, DGEQP3)(hif_lapack_int *, hif_lapack_int *, double *,
                            hif_lapack_int *, hif_lapack_int *, double *,
                            double *, hif_lapack_int *, hif_lapack_int *);

// float
void HIF_FC(sgeqp3, SGEQP3)(hif_lapack_int *, hif_lapack_int *, float *,
                            hif_lapack_int *, hif_lapack_int *, float *,
                            float *, hif_lapack_int *, hif_lapack_int *);

// complex double
void HIF_FC(zgeqp3, ZGEQP3)(hif_lapack_int *, hif_lapack_int *, void *,
                            hif_lapack_int *, hif_lapack_int *, void *, void *,
                            hif_lapack_int *, double *, hif_lapack_int *);

// complex float
void HIF_FC(cgeqp3, CGEQP3)(hif_lapack_int *, hif_lapack_int *, void *,
                            hif_lapack_int *, hif_lapack_int *, void *, void *,
                            hif_lapack_int *, float *, hif_lapack_int *);

// triangle cond estimator

// double
void HIF_FC(dtrcon, DTRCON)(char *, char *, char *, hif_lapack_int *, double *,
                            hif_lapack_int *, double *, double *,
                            hif_lapack_int *, hif_lapack_int *);

// single
void HIF_FC(strcon, STRCON)(char *, char *, char *, hif_lapack_int *, float *,
                            hif_lapack_int *, float *, float *,
                            hif_lapack_int *, hif_lapack_int *);

// complex double
void HIF_FC(ztrcon, ZTRCON)(char *, char *, char *, hif_lapack_int *, void *,
                            hif_lapack_int *, double *, void *, double *,
                            hif_lapack_int *);

// complex float
void HIF_FC(ctrcon, CTRCON)(char *, char *, char *, hif_lapack_int *, void *,
                            hif_lapack_int *, float *, void *, float *,
                            hif_lapack_int *);

// Householder reflector multiplication
void HIF_FC(dormqr, DORMQR)(char *, char *, hif_lapack_int *, hif_lapack_int *,
                            hif_lapack_int *, double *, hif_lapack_int *,
                            double *, double *c, hif_lapack_int *, double *,
                            hif_lapack_int *, hif_lapack_int *);

// single
void HIF_FC(sormqr, SORMQR)(char *, char *, hif_lapack_int *, hif_lapack_int *,
                            hif_lapack_int *, float *, hif_lapack_int *,
                            float *, float *c, hif_lapack_int *, float *,
                            hif_lapack_int *, hif_lapack_int *);

// complex double
void HIF_FC(zunmqr, ZUNMQR)(char *, char *, hif_lapack_int *, hif_lapack_int *,
                            hif_lapack_int *, void *, hif_lapack_int *, void *,
                            void *c, hif_lapack_int *, void *, hif_lapack_int *,
                            hif_lapack_int *);

// complex single
void HIF_FC(cunmqr, CUNMQR)(char *, char *, hif_lapack_int *, hif_lapack_int *,
                            hif_lapack_int *, void *, hif_lapack_int *, void *,
                            void *c, hif_lapack_int *, void *, hif_lapack_int *,
                            hif_lapack_int *);

// increment 2-norm condition number estimator
void HIF_FC(dlaic1, DLAIC1)(hif_lapack_int *, hif_lapack_int *, double *,
                            double *, double *, double *, double *, double *,
                            double *);

// single
void HIF_FC(slaic1, SLAIC1)(hif_lapack_int *, hif_lapack_int *, float *,
                            float *, float *, float *, float *, float *,
                            float *);

// complex double
void HIF_FC(zlaic1, ZLAIC1)(hif_lapack_int *, hif_lapack_int *, void *,
                            double *, void *, void *, double *, void *, void *);

// complex single
void HIF_FC(claic1, CLAIC1)(hif_lapack_int *, hif_lapack_int *, void *, float *,
                            void *, void *, float *, void *, void *);

///////////////////////
// Sy/He Eigenvalue
///////////////////////

// double
void HIF_FC(dsyev, DSYEV)(char *, char *, hif_lapack_int *, double *,
                          hif_lapack_int *, double *, double *,
                          hif_lapack_int *, hif_lapack_int *);
// single
void HIF_FC(ssyev, SSYEV)(char *, char *, hif_lapack_int *, float *,
                          hif_lapack_int *, float *, float *, hif_lapack_int *,
                          hif_lapack_int *);
// complex double
void HIF_FC(zheev, ZHEEV)(char *, char *, hif_lapack_int *, void *,
                          hif_lapack_int *, double *, void *, hif_lapack_int *,
                          double *, hif_lapack_int *);
// complex single
void HIF_FC(cheev, CHEEV)(char *, char *, hif_lapack_int *, void *,
                          hif_lapack_int *, float *, void *, hif_lapack_int *,
                          float *, hif_lapack_int *);

///////////////////////
// Common
///////////////////////

// matrix-vector, used in solve

// double
void HIF_FC(dgemv, DGEMV)(char *, hif_lapack_int *, hif_lapack_int *, double *,
                          double *, hif_lapack_int *, double *,
                          hif_lapack_int *, double *, double *,
                          hif_lapack_int *);

// single
void HIF_FC(sgemv, SGEMV)(char *, hif_lapack_int *, hif_lapack_int *, float *,
                          float *, hif_lapack_int *, float *, hif_lapack_int *,
                          float *, float *, hif_lapack_int *);

// complex double
void HIF_FC(zgemv, ZGEMV)(char *, hif_lapack_int *, hif_lapack_int *, void *,
                          void *, hif_lapack_int *, void *, hif_lapack_int *,
                          void *, void *, hif_lapack_int *);

// complex single
void HIF_FC(cgemv, CGEMV)(char *, hif_lapack_int *, hif_lapack_int *, void *,
                          void *, hif_lapack_int *, void *, hif_lapack_int *,
                          void *, void *, hif_lapack_int *);

// triangular solve

// double
void HIF_FC(dtrsv, DTRSV)(char *, char *, char *, hif_lapack_int *, double *,
                          hif_lapack_int *, double *, hif_lapack_int *);

// single
void HIF_FC(strsv, STRSV)(char *, char *, char *, hif_lapack_int *, float *,
                          hif_lapack_int *, float *, hif_lapack_int *);

// complex double
void HIF_FC(ztrsv, ZTRSV)(char *, char *, char *, hif_lapack_int *, void *,
                          hif_lapack_int *, void *, hif_lapack_int *);

// complex single
void HIF_FC(ctrsv, CTRSV)(char *, char *, char *, hif_lapack_int *, void *,
                          hif_lapack_int *, void *, hif_lapack_int *);

// triangular matrix-vector

// double
void HIF_FC(dtrmv, DTRMV)(char *, char *, char *, hif_lapack_int *, double *,
                          hif_lapack_int *, double *, hif_lapack_int *);

// single
void HIF_FC(strmv, STRMV)(char *, char *, char *, hif_lapack_int *, float *,
                          hif_lapack_int *, float *, hif_lapack_int *);

// double complex
void HIF_FC(ztrmv, ZTRMV)(char *, char *, char *, hif_lapack_int *, void *,
                          hif_lapack_int *, void *, hif_lapack_int *);

// single complex
void HIF_FC(ctrmv, CTRMV)(char *, char *, char *, hif_lapack_int *, void *,
                          hif_lapack_int *, void *, hif_lapack_int *);
}
#  endif
#endif  // DOXYGEN_SHOULD_SKIP_THIS

namespace hif {
namespace internal {

/*!
 * \addtogroup sss
 * @{
 */

/// \brief double version of LU with partial pivoting
inline hif_lapack_int getrf(const hif_lapack_int m, const hif_lapack_int n,
                            double *A, const hif_lapack_int lda,
                            hif_lapack_int *ipiv) {
  hif_lapack_int info;
  HIF_FC(dgetrf, DGETRF)
  ((hif_lapack_int *)&m, (hif_lapack_int *)&n, A, (hif_lapack_int *)&lda, ipiv,
   &info);
  return info;
}

/// \brief single version of LU with partial pivoting
inline hif_lapack_int getrf(const hif_lapack_int m, const hif_lapack_int n,
                            float *A, const hif_lapack_int lda,
                            hif_lapack_int *ipiv) {
  hif_lapack_int info;
  HIF_FC(sgetrf, SGETRF)
  ((hif_lapack_int *)&m, (hif_lapack_int *)&n, A, (hif_lapack_int *)&lda, ipiv,
   &info);
  return info;
}

/// \brief complex double version of LU with partial pivoting
inline hif_lapack_int getrf(const hif_lapack_int m, const hif_lapack_int n,
                            std::complex<double> *A, const hif_lapack_int lda,
                            hif_lapack_int *ipiv) {
  hif_lapack_int info;
  HIF_FC(zgetrf, ZGETRF)
  ((hif_lapack_int *)&m, (hif_lapack_int *)&n, (void *)A,
   (hif_lapack_int *)&lda, ipiv, &info);
  return info;
}

/// \brief complex single version of LU with partial pivoting
inline hif_lapack_int getrf(const hif_lapack_int m, const hif_lapack_int n,
                            std::complex<float> *A, const hif_lapack_int lda,
                            hif_lapack_int *ipiv) {
  hif_lapack_int info;
  HIF_FC(cgetrf, CGETRF)
  ((hif_lapack_int *)&m, (hif_lapack_int *)&n, (void *)A,
   (hif_lapack_int *)&lda, ipiv, &info);
  return info;
}

/// \brief solve with factorized matrix, double version
inline hif_lapack_int getrs(const char tran, const hif_lapack_int n,
                            const hif_lapack_int nrhs, const double *a,
                            const hif_lapack_int  lda,
                            const hif_lapack_int *ipiv, double *b,
                            const hif_lapack_int ldb) {
  hif_lapack_int info;
  HIF_FC(dgetrs, DGETRS)
  ((char *)&tran, (hif_lapack_int *)&n, (hif_lapack_int *)&nrhs, (double *)a,
   (hif_lapack_int *)&lda, (hif_lapack_int *)ipiv, b, (hif_lapack_int *)&ldb,
   &info);
  return info;
}

/// \brief solve with factorized matrix, single version
inline hif_lapack_int getrs(const char tran, const hif_lapack_int n,
                            const hif_lapack_int nrhs, const float *a,
                            const hif_lapack_int  lda,
                            const hif_lapack_int *ipiv, float *b,
                            const hif_lapack_int ldb) {
  hif_lapack_int info;
  HIF_FC(sgetrs, SGETRS)
  ((char *)&tran, (hif_lapack_int *)&n, (hif_lapack_int *)&nrhs, (float *)a,
   (hif_lapack_int *)&lda, (hif_lapack_int *)ipiv, b, (hif_lapack_int *)&ldb,
   &info);
  return info;
}

/// \brief solve with factorized matrix, complex double version
inline hif_lapack_int getrs(const char tran, const hif_lapack_int n,
                            const hif_lapack_int        nrhs,
                            const std::complex<double> *a,
                            const hif_lapack_int        lda,
                            const hif_lapack_int *ipiv, std::complex<double> *b,
                            const hif_lapack_int ldb) {
  hif_lapack_int info;
  HIF_FC(zgetrs, ZGETRS)
  ((char *)&tran, (hif_lapack_int *)&n, (hif_lapack_int *)&nrhs, (void *)a,
   (hif_lapack_int *)&lda, (hif_lapack_int *)ipiv, (void *)b,
   (hif_lapack_int *)&ldb, &info);
  return info;
}

/// \brief solve with factorized matrix, complex single version
inline hif_lapack_int getrs(const char tran, const hif_lapack_int n,
                            const hif_lapack_int       nrhs,
                            const std::complex<float> *a,
                            const hif_lapack_int       lda,
                            const hif_lapack_int *ipiv, std::complex<float> *b,
                            const hif_lapack_int ldb) {
  hif_lapack_int info;
  HIF_FC(cgetrs, CGETRS)
  ((char *)&tran, (hif_lapack_int *)&n, (hif_lapack_int *)&nrhs, (void *)a,
   (hif_lapack_int *)&lda, (hif_lapack_int *)ipiv, (void *)b,
   (hif_lapack_int *)&ldb, &info);
  return info;
}

/// \brief double version QR with column pivoting
inline hif_lapack_int geqp3(const hif_lapack_int m, const hif_lapack_int n,
                            double *a, const hif_lapack_int lda,
                            hif_lapack_int *jpvt, double *tau, double *work,
                            const hif_lapack_int lwork,
                            double * /* rwork */ = nullptr) {
  hif_lapack_int info;
  HIF_FC(dgeqp3, DGEQP3)
  ((hif_lapack_int *)&m, (hif_lapack_int *)&n, a, (hif_lapack_int *)&lda, jpvt,
   tau, work, (hif_lapack_int *)&lwork, &info);
  return info;
}

/// \brief single version QR with column pivoting
inline hif_lapack_int geqp3(const hif_lapack_int m, const hif_lapack_int n,
                            float *a, const hif_lapack_int lda,
                            hif_lapack_int *jpvt, float *tau, float *work,
                            const hif_lapack_int lwork,
                            float * /* rwork */ = nullptr) {
  hif_lapack_int info;
  HIF_FC(sgeqp3, SGEQP3)
  ((hif_lapack_int *)&m, (hif_lapack_int *)&n, a, (hif_lapack_int *)&lda, jpvt,
   tau, work, (hif_lapack_int *)&lwork, &info);
  return info;
}

/// \brief complex double version QR with column pivoting
inline hif_lapack_int geqp3(const hif_lapack_int m, const hif_lapack_int n,
                            std::complex<double> *a, const hif_lapack_int lda,
                            hif_lapack_int *jpvt, std::complex<double> *tau,
                            std::complex<double> *work,
                            const hif_lapack_int lwork, double *rwork) {
  hif_lapack_int info;
  HIF_FC(zgeqp3, ZGEQP3)
  ((hif_lapack_int *)&m, (hif_lapack_int *)&n, (void *)a,
   (hif_lapack_int *)&lda, jpvt, (void *)tau, (void *)work,
   (hif_lapack_int *)&lwork, rwork, &info);
  return info;
}

/// \brief complex single version QR with column pivoting
inline hif_lapack_int geqp3(const hif_lapack_int m, const hif_lapack_int n,
                            std::complex<float> *a, const hif_lapack_int lda,
                            hif_lapack_int *jpvt, std::complex<float> *tau,
                            std::complex<float> *work,
                            const hif_lapack_int lwork, float *rwork) {
  hif_lapack_int info;
  HIF_FC(cgeqp3, CGEQP3)
  ((hif_lapack_int *)&m, (hif_lapack_int *)&n, (void *)a,
   (hif_lapack_int *)&lda, jpvt, (void *)tau, (void *)work,
   (hif_lapack_int *)&lwork, rwork, &info);
  return info;
}

/// \brief double version triangular condition number estimator
/// \note This is needed to determined the rank for QRCP, i.e. TQRCP
inline hif_lapack_int trcon(const char norm, const char uplo, const char diag,
                            const hif_lapack_int n, const double *a,
                            const hif_lapack_int lda, double &rcond,
                            double *work, hif_lapack_int *iwork) {
  hif_lapack_int info;
  HIF_FC(dtrcon, DTRCON)
  ((char *)&norm, (char *)&uplo, (char *)&diag, (hif_lapack_int *)&n,
   (double *)a, (hif_lapack_int *)&lda, &rcond, work, iwork, &info);
  return info;
}

/// \brief single version triangular condition number estimator
/// \note This is needed to determined the rank for QRCP, i.e. TQRCP
inline hif_lapack_int trcon(const char norm, const char uplo, const char diag,
                            const hif_lapack_int n, const float *a,
                            const hif_lapack_int lda, float &rcond, float *work,
                            hif_lapack_int *iwork) {
  hif_lapack_int info;
  HIF_FC(strcon, STRCON)
  ((char *)&norm, (char *)&uplo, (char *)&diag, (hif_lapack_int *)&n,
   (float *)a, (hif_lapack_int *)&lda, &rcond, work, iwork, &info);
  return info;
}

/// \brief complex double version triangular condition number estimator
/// \note This is needed to determined the rank for QRCP, i.e. TQRCP
inline hif_lapack_int trcon(const char norm, const char uplo, const char diag,
                            const hif_lapack_int        n,
                            const std::complex<double> *a,
                            const hif_lapack_int lda, double &rcond,
                            std::complex<double> *work, double *rwork) {
  hif_lapack_int info;
  HIF_FC(ztrcon, ZTRCON)
  ((char *)&norm, (char *)&uplo, (char *)&diag, (hif_lapack_int *)&n, (void *)a,
   (hif_lapack_int *)&lda, &rcond, (void *)work, rwork, &info);
  return info;
}

/// \brief complex float version triangular condition number estimator
/// \note This is needed to determined the rank for QRCP, i.e. TQRCP
inline hif_lapack_int trcon(const char norm, const char uplo, const char diag,
                            const hif_lapack_int       n,
                            const std::complex<float> *a,
                            const hif_lapack_int lda, float &rcond,
                            std::complex<float> *work, float *rwork) {
  hif_lapack_int info;
  HIF_FC(ctrcon, CTRCON)
  ((char *)&norm, (char *)&uplo, (char *)&diag, (hif_lapack_int *)&n, (void *)a,
   (hif_lapack_int *)&lda, &rcond, (void *)work, rwork, &info);
  return info;
}

/// \brief double version for handling Householder reflectors
inline hif_lapack_int ormqr(const char side, const char trans,
                            const hif_lapack_int m, const hif_lapack_int n,
                            const hif_lapack_int k, const double *a,
                            const hif_lapack_int lda, const double *tau,
                            double *c, const hif_lapack_int ldc, double *work,
                            const hif_lapack_int lwork) {
  hif_lapack_int info;
  HIF_FC(dormqr, DORMQR)
  ((char *)&side, (char *)&trans, (hif_lapack_int *)&m, (hif_lapack_int *)&n,
   (hif_lapack_int *)&k, (double *)a, (hif_lapack_int *)&lda, (double *)tau, c,
   (hif_lapack_int *)&ldc, work, (hif_lapack_int *)&lwork, &info);
  return info;
}

/// \brief single version for handling Householder reflectors
inline hif_lapack_int ormqr(const char side, const char trans,
                            const hif_lapack_int m, const hif_lapack_int n,
                            const hif_lapack_int k, const float *a,
                            const hif_lapack_int lda, const float *tau,
                            float *c, const hif_lapack_int ldc, float *work,
                            const hif_lapack_int lwork) {
  hif_lapack_int info;
  HIF_FC(sormqr, SORMQR)
  ((char *)&side, (char *)&trans, (hif_lapack_int *)&m, (hif_lapack_int *)&n,
   (hif_lapack_int *)&k, (float *)a, (hif_lapack_int *)&lda, (float *)tau, c,
   (hif_lapack_int *)&ldc, work, (hif_lapack_int *)&lwork, &info);
  return info;
}

/// \brief complex double version for handling Householder reflectors
/// \note We use \a ormqr instead of unmqr for consistency. Also, we will
///       reset \a trans to be 'C' from 'T' for the sake of user-friendliness.
inline hif_lapack_int ormqr(const char side, char trans, const hif_lapack_int m,
                            const hif_lapack_int n, const hif_lapack_int k,
                            const std::complex<double> *a,
                            const hif_lapack_int        lda,
                            const std::complex<double> *tau,
                            std::complex<double> *c, const hif_lapack_int ldc,
                            std::complex<double> *work,
                            const hif_lapack_int  lwork) {
  hif_lapack_int info;
  if (trans == 'T') trans = 'C';
  HIF_FC(zunmqr, ZUNMQR)
  ((char *)&side, &trans, (hif_lapack_int *)&m, (hif_lapack_int *)&n,
   (hif_lapack_int *)&k, (void *)a, (hif_lapack_int *)&lda, (void *)tau,
   (void *)c, (hif_lapack_int *)&ldc, (void *)work, (hif_lapack_int *)&lwork,
   &info);
  return info;
}

/// \brief complex single version for handling Householder reflectors
/// \note We use \a ormqr instead of unmqr for consistency. Also, we will
///       reset \a trans to be 'C' from 'T' for the sake of user-friendliness.
inline hif_lapack_int ormqr(const char side, char trans, const hif_lapack_int m,
                            const hif_lapack_int n, const hif_lapack_int k,
                            const std::complex<float> *a,
                            const hif_lapack_int       lda,
                            const std::complex<float> *tau,
                            std::complex<float> *c, const hif_lapack_int ldc,
                            std::complex<float> *work,
                            const hif_lapack_int lwork) {
  hif_lapack_int info;
  if (trans == 'T') trans = 'C';
  HIF_FC(cunmqr, CUNMQR)
  ((char *)&side, &trans, (hif_lapack_int *)&m, (hif_lapack_int *)&n,
   (hif_lapack_int *)&k, (void *)a, (hif_lapack_int *)&lda, (void *)tau,
   (void *)c, (hif_lapack_int *)&ldc, (void *)work, (hif_lapack_int *)&lwork,
   &info);
  return info;
}

/// \brief double version of incremental 2-norm condition number estimator
inline hif_lapack_int laic1(const hif_lapack_int job, const hif_lapack_int j,
                            const double *x, const double sest, const double *w,
                            const double gamma, double &sestpr, double &s,
                            double &c) {
  HIF_FC(dlaic1, DLAIC1)
  ((hif_lapack_int *)&job, (hif_lapack_int *)&j, (double *)x, (double *)&sest,
   (double *)w, (double *)&gamma, &sestpr, &s, &c);
  return 0;
}

/// \brief single version of incremental 2-norm condition number estimator
inline hif_lapack_int laic1(const hif_lapack_int job, const hif_lapack_int j,
                            const float *x, const float sest, const float *w,
                            const float gamma, float &sestpr, float &s,
                            float &c) {
  HIF_FC(slaic1, SLAIC1)
  ((hif_lapack_int *)&job, (hif_lapack_int *)&j, (float *)x, (float *)&sest,
   (float *)w, (float *)&gamma, &sestpr, &s, &c);
  return 0;
}

/// \brief double version of incremental 2-norm condition number estimator
inline hif_lapack_int laic1(const hif_lapack_int job, const hif_lapack_int j,
                            const std::complex<double> *x, const double sest,
                            const std::complex<double> *w,
                            const std::complex<double> gamma, double &sestpr,
                            std::complex<double> &s, std::complex<double> &c) {
  HIF_FC(zlaic1, ZLAIC1)
  ((hif_lapack_int *)&job, (hif_lapack_int *)&j, (void *)x, (double *)&sest,
   (void *)w, (void *)&gamma, &sestpr, (void *)&s, (void *)&c);
  return 0;
}

/// \brief double version of incremental 2-norm condition number estimator
inline hif_lapack_int laic1(const hif_lapack_int job, const hif_lapack_int j,
                            const std::complex<float> *x, const float sest,
                            const std::complex<float> *w,
                            const std::complex<float> gamma, float &sestpr,
                            std::complex<float> &s, std::complex<float> &c) {
  HIF_FC(claic1, CLAIC1)
  ((hif_lapack_int *)&job, (hif_lapack_int *)&j, (void *)x, (float *)&sest,
   (void *)w, (void *)&gamma, &sestpr, (void *)&s, (void *)&c);
  return 0;
}

/// \brief double version of handling symmetric eigendecomposition
inline hif_lapack_int syev(const char uplo, const hif_lapack_int n, double *a,
                           const hif_lapack_int lda, double *w, double *work,
                           const hif_lapack_int lwork,
                           double * /* rwork */ = nullptr) {
  static char    jobz('V');
  hif_lapack_int info;
  HIF_FC(dsyev, DSYEV)
  (&jobz, (char *)&uplo, (hif_lapack_int *)&n, a, (hif_lapack_int *)&lda, w,
   work, (hif_lapack_int *)&lwork, &info);
  return info;
}

/// \brief single version of handling symmetric eigendecomposition
inline hif_lapack_int syev(const char uplo, const hif_lapack_int n, float *a,
                           const hif_lapack_int lda, float *w, float *work,
                           const hif_lapack_int lwork,
                           float * /* rwork */ = nullptr) {
  static char    jobz('V');
  hif_lapack_int info;
  HIF_FC(ssyev, SSYEV)
  (&jobz, (char *)&uplo, (hif_lapack_int *)&n, a, (hif_lapack_int *)&lda, w,
   work, (hif_lapack_int *)&lwork, &info);
  return info;
}

/// \brief complex double version of handling symmetric eigendecomposition
inline hif_lapack_int syev(const char uplo, const hif_lapack_int n,
                           std::complex<double> *a, const hif_lapack_int lda,
                           double *w, std::complex<double> *work,
                           const hif_lapack_int lwork, double *rwork) {
  static char    jobz('V');
  hif_lapack_int info;
  HIF_FC(zheev, ZHEEV)
  (&jobz, (char *)&uplo, (hif_lapack_int *)&n, (void *)a,
   (hif_lapack_int *)&lda, w, (void *)work, (hif_lapack_int *)&lwork, rwork,
   &info);
  return info;
}

/// \brief complex single version of handling symmetric eigendecomposition
inline hif_lapack_int syev(const char uplo, const hif_lapack_int n,
                           std::complex<float> *a, const hif_lapack_int lda,
                           float *w, std::complex<float> *work,
                           const hif_lapack_int lwork, float *rwork) {
  static char    jobz('V');
  hif_lapack_int info;
  HIF_FC(cheev, CHEEV)
  (&jobz, (char *)&uplo, (hif_lapack_int *)&n, (void *)a,
   (hif_lapack_int *)&lda, w, (void *)work, (hif_lapack_int *)&lwork, rwork,
   &info);
  return info;
}

/// \brief double version of matrix-vector product
inline void gemv(const char trans, const hif_lapack_int m,
                 const hif_lapack_int n, const double alpha, const double *a,
                 const hif_lapack_int lda, const double *x, const double beta,
                 double *y) {
  static hif_lapack_int inc(1);
  HIF_FC(dgemv, DGEMV)
  ((char *)&trans, (hif_lapack_int *)&m, (hif_lapack_int *)&n, (double *)&alpha,
   (double *)a, (hif_lapack_int *)&lda, (double *)x, &inc, (double *)&beta, y,
   &inc);
}

/// \brief single version of matrix-vector product
inline void gemv(const char trans, const hif_lapack_int m,
                 const hif_lapack_int n, const float alpha, const float *a,
                 const hif_lapack_int lda, const float *x, const float beta,
                 float *y) {
  static hif_lapack_int inc(1);
  HIF_FC(sgemv, SGEMV)
  ((char *)&trans, (hif_lapack_int *)&m, (hif_lapack_int *)&n, (float *)&alpha,
   (float *)a, (hif_lapack_int *)&lda, (float *)x, &inc, (float *)&beta, y,
   &inc);
}

/// \brief complex double version of matrix-vector product
inline void gemv(const char trans, const hif_lapack_int m,
                 const hif_lapack_int n, const std::complex<double> alpha,
                 const std::complex<double> *a, const hif_lapack_int lda,
                 const std::complex<double> *x, const std::complex<double> beta,
                 std::complex<double> *y) {
  static hif_lapack_int inc(1);
  HIF_FC(zgemv, ZGEMV)
  ((char *)&trans, (hif_lapack_int *)&m, (hif_lapack_int *)&n, (void *)&alpha,
   (void *)a, (hif_lapack_int *)&lda, (void *)x, &inc, (void *)&beta, (void *)y,
   &inc);
}

/// \brief complex single version of matrix-vector product
inline void gemv(const char trans, const hif_lapack_int m,
                 const hif_lapack_int n, const std::complex<float> alpha,
                 const std::complex<float> *a, const hif_lapack_int lda,
                 const std::complex<float> *x, const std::complex<float> beta,
                 std::complex<float> *y) {
  static hif_lapack_int inc(1);
  HIF_FC(cgemv, CGEMV)
  ((char *)&trans, (hif_lapack_int *)&m, (hif_lapack_int *)&n, (void *)&alpha,
   (void *)a, (hif_lapack_int *)&lda, (void *)x, &inc, (void *)&beta, (void *)y,
   &inc);
}

/// \brief double version of triangular solve
inline void trsv(const char uplo, const char trans, const char diag,
                 const hif_lapack_int n, const double *a,
                 const hif_lapack_int lda, double *x,
                 const hif_lapack_int incx) {
  HIF_FC(dtrsv, DTRSV)
  ((char *)&uplo, (char *)&trans, (char *)&diag, (hif_lapack_int *)&n,
   (double *)a, (hif_lapack_int *)&lda, x, (hif_lapack_int *)&incx);
}

/// \brief single version of triangular solve
inline void trsv(const char uplo, const char trans, const char diag,
                 const hif_lapack_int n, const float *a,
                 const hif_lapack_int lda, float *x,
                 const hif_lapack_int incx) {
  HIF_FC(strsv, STRSV)
  ((char *)&uplo, (char *)&trans, (char *)&diag, (hif_lapack_int *)&n,
   (float *)a, (hif_lapack_int *)&lda, x, (hif_lapack_int *)&incx);
}

/// \brief complex double version of triangular solve
inline void trsv(const char uplo, const char trans, const char diag,
                 const hif_lapack_int n, const std::complex<double> *a,
                 const hif_lapack_int lda, std::complex<double> *x,
                 const hif_lapack_int incx) {
  HIF_FC(ztrsv, ZTRSV)
  ((char *)&uplo, (char *)&trans, (char *)&diag, (hif_lapack_int *)&n,
   (void *)a, (hif_lapack_int *)&lda, (void *)x, (hif_lapack_int *)&incx);
}

/// \brief complex single version of triangular solve
inline void trsv(const char uplo, const char trans, const char diag,
                 const hif_lapack_int n, const std::complex<float> *a,
                 const hif_lapack_int lda, std::complex<float> *x,
                 const hif_lapack_int incx) {
  HIF_FC(ctrsv, CTRSV)
  ((char *)&uplo, (char *)&trans, (char *)&diag, (hif_lapack_int *)&n,
   (void *)a, (hif_lapack_int *)&lda, (void *)x, (hif_lapack_int *)&incx);
}

/// \brief double version of triangular matrix-vector
inline void trmv(const char uplo, const char trans, const char diag,
                 const hif_lapack_int n, const double *a,
                 const hif_lapack_int lda, double *x,
                 const hif_lapack_int incx) {
  HIF_FC(dtrmv, DTRMV)
  ((char *)&uplo, (char *)&trans, (char *)&diag, (hif_lapack_int *)&n,
   (double *)a, (hif_lapack_int *)&lda, x, (hif_lapack_int *)&incx);
}

/// \brief single version of triangular matrix-vector
inline void trmv(const char uplo, const char trans, const char diag,
                 const hif_lapack_int n, const float *a,
                 const hif_lapack_int lda, float *x,
                 const hif_lapack_int incx) {
  HIF_FC(strmv, STRMV)
  ((char *)&uplo, (char *)&trans, (char *)&diag, (hif_lapack_int *)&n,
   (float *)a, (hif_lapack_int *)&lda, x, (hif_lapack_int *)&incx);
}

/// \brief complex double version of triangular matrix-vector
inline void trmv(const char uplo, const char trans, const char diag,
                 const hif_lapack_int n, const std::complex<double> *a,
                 const hif_lapack_int lda, std::complex<double> *x,
                 const hif_lapack_int incx) {
  HIF_FC(ztrmv, ZTRMV)
  ((char *)&uplo, (char *)&trans, (char *)&diag, (hif_lapack_int *)&n,
   (void *)a, (hif_lapack_int *)&lda, (void *)x, (hif_lapack_int *)&incx);
}

/// \brief complex single version of triangular matrix-vector
inline void trmv(const char uplo, const char trans, const char diag,
                 const hif_lapack_int n, const std::complex<float> *a,
                 const hif_lapack_int lda, std::complex<float> *x,
                 const hif_lapack_int incx) {
  HIF_FC(ctrmv, CTRMV)
  ((char *)&uplo, (char *)&trans, (char *)&diag, (hif_lapack_int *)&n,
   (void *)a, (hif_lapack_int *)&lda, (void *)x, (hif_lapack_int *)&incx);
}

/*!
 * @}
 */  // sss group

}  // namespace internal
}  // namespace hif

#endif  // _HIF_SMALLSCALE_LAPACKF_HPP