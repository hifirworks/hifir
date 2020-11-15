///////////////////////////////////////////////////////////////////////////////
//                  This file is part of HIF project                         //
///////////////////////////////////////////////////////////////////////////////

/*!
 * \file hif/small_scale/qrcp_lapack.hpp
 * \brief Interface wrappers for QR with column pivoting routines in LAPACK
 * \author Qiao Chen
 * \todo add interfaces for complex types

\verbatim
Copyright (C) 2019 NumGeom Group at Stony Brook University

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
\endverbatim

 */

#ifndef _HIF_SMALLSCALE_LAPACK_QRCP_HPP
#define _HIF_SMALLSCALE_LAPACK_QRCP_HPP

#include "hif/small_scale/config.hpp"

#ifndef DOXYGEN_SHOULD_SKIP_THIS
#  if HIF_HAS_MKL == 0
extern "C" {

// factorization

// double
void HIF_FC(dgeqp3, DGEQP3)(hif_lapack_int *, hif_lapack_int *, double *,
                            hif_lapack_int *, hif_lapack_int *, double *,
                            double *, hif_lapack_int *, hif_lapack_int *);

// float
void HIF_FC(sgeqp3, SGEQP3)(hif_lapack_int *, hif_lapack_int *, float *,
                            hif_lapack_int *, hif_lapack_int *, float *,
                            float *, hif_lapack_int *, hif_lapack_int *);

// triangle cond estimator

// double
void HIF_FC(dtrcon, DTRCON)(char *, char *, char *, hif_lapack_int *, double *,
                            hif_lapack_int *, double *, double *,
                            hif_lapack_int *, hif_lapack_int *);

// single
void HIF_FC(strcon, STRCON)(char *, char *, char *, hif_lapack_int *, float *,
                            hif_lapack_int *, float *, float *,
                            hif_lapack_int *, hif_lapack_int *);

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

// increment 2-norm condition number estimator
void HIF_FC(dlaic1, DLAIC1)(hif_lapack_int *, hif_lapack_int *, double *,
                            double *, double *, double *, double *, double *,
                            double *);

// single
void HIF_FC(slaic1, SLAIC1)(hif_lapack_int *, hif_lapack_int *, float *,
                            float *, float *, float *, float *, float *,
                            float *);
}
#  endif
#endif  // DOXYGEN_SHOULD_SKIP_THIS

namespace hif {
namespace internal {

/*!
 * \addtogroup sss
 * @{
 */

/// \brief double version QR with column pivoting
inline hif_lapack_int geqp3(const hif_lapack_int m, const hif_lapack_int n,
                            double *a, const hif_lapack_int lda,
                            hif_lapack_int *jpvt, double *tau, double *work,
                            const hif_lapack_int lwork) {
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
                            const hif_lapack_int lwork) {
  hif_lapack_int info;
  HIF_FC(sgeqp3, SGEQP3)
  ((hif_lapack_int *)&m, (hif_lapack_int *)&n, a, (hif_lapack_int *)&lda, jpvt,
   tau, work, (hif_lapack_int *)&lwork, &info);
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

/*!
 * @}
 */  // sss group

}  // namespace internal
}  // namespace hif

#endif  // _HIF_SMALLSCALE_LAPACK_QRCP_HPP