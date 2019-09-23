///////////////////////////////////////////////////////////////////////////////
//                This file is part of HILUCSI project                       //
///////////////////////////////////////////////////////////////////////////////

/*!
 * \file hilucsi/small_scale/qrcp.hpp
 * \brief Interface wrappers for QR with column pivoting routines in LAPACK
 * \authors Qiao,
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

#ifndef _HILUCSI_SMALLSCALE_LAPACK_QRCP_HPP
#define _HILUCSI_SMALLSCALE_LAPACK_QRCP_HPP

#include "hilucsi/small_scale/config.hpp"

#ifndef DOXYGEN_SHOULD_SKIP_THIS

extern "C" {

// factorization

// double
void HILUCSI_FC(dgeqp3, DGEQP3)(hilucsi_lapack_int *, hilucsi_lapack_int *,
                                double *, hilucsi_lapack_int *,
                                hilucsi_lapack_int *, double *, double *,
                                hilucsi_lapack_int *, hilucsi_lapack_int *);

// float
void HILUCSI_FC(sgeqp3, SGEQP3)(hilucsi_lapack_int *, hilucsi_lapack_int *,
                                float *, hilucsi_lapack_int *,
                                hilucsi_lapack_int *, float *, float *,
                                hilucsi_lapack_int *, hilucsi_lapack_int *);

// triangle cond estimator

// double
void HILUCSI_FC(dtrcon, DTRCON)(char *, char *, char *, hilucsi_lapack_int *,
                                double *, hilucsi_lapack_int *, double *,
                                double *, hilucsi_lapack_int *,
                                hilucsi_lapack_int *);

// single
void HILUCSI_FC(strcon, STRCON)(char *, char *, char *, hilucsi_lapack_int *,
                                float *, hilucsi_lapack_int *, float *, float *,
                                hilucsi_lapack_int *, hilucsi_lapack_int *);

// Householder reflector multiplication
void HILUCSI_FC(dormqr, DORMQR)(char *, char *, hilucsi_lapack_int *,
                                hilucsi_lapack_int *, hilucsi_lapack_int *,
                                double *, hilucsi_lapack_int *, double *,
                                double *c, hilucsi_lapack_int *, double *,
                                hilucsi_lapack_int *, hilucsi_lapack_int *);

// single
void HILUCSI_FC(sormqr, SORMQR)(char *, char *, hilucsi_lapack_int *,
                                hilucsi_lapack_int *, hilucsi_lapack_int *,
                                float *, hilucsi_lapack_int *, float *,
                                float *c, hilucsi_lapack_int *, float *,
                                hilucsi_lapack_int *, hilucsi_lapack_int *);
}

#endif  // DOXYGEN_SHOULD_SKIP_THIS

namespace hilucsi {
namespace internal {

/*!
 * \addtogroup sss
 * @{
 */

/// \brief double version QR with column pivoting
inline hilucsi_lapack_int geqp3(const hilucsi_lapack_int m,
                                const hilucsi_lapack_int n, double *a,
                                const hilucsi_lapack_int lda,
                                hilucsi_lapack_int *jpvt, double *tau,
                                double *work, const hilucsi_lapack_int lwork) {
  hilucsi_lapack_int info;
  HILUCSI_FC(dgeqp3, DGEQP3)
  ((hilucsi_lapack_int *)&m, (hilucsi_lapack_int *)&n, a,
   (hilucsi_lapack_int *)&lda, jpvt, tau, work, (hilucsi_lapack_int *)&lwork,
   &info);
  return info;
}

/// \brief single version QR with column pivoting
inline hilucsi_lapack_int geqp3(const hilucsi_lapack_int m,
                                const hilucsi_lapack_int n, float *a,
                                const hilucsi_lapack_int lda,
                                hilucsi_lapack_int *jpvt, float *tau,
                                float *work, const hilucsi_lapack_int lwork) {
  hilucsi_lapack_int info;
  HILUCSI_FC(sgeqp3, SGEQP3)
  ((hilucsi_lapack_int *)&m, (hilucsi_lapack_int *)&n, a,
   (hilucsi_lapack_int *)&lda, jpvt, tau, work, (hilucsi_lapack_int *)&lwork,
   &info);
  return info;
}

/// \brief double version triangular condition number estimator
/// \note This is needed to determined the rank for QRCP, i.e. TQRCP
inline hilucsi_lapack_int trcon(const char norm, const char uplo,
                                const char diag, const hilucsi_lapack_int n,
                                const double *a, const hilucsi_lapack_int lda,
                                double &rcond, double *work,
                                hilucsi_lapack_int *iwork) {
  hilucsi_lapack_int info;
  HILUCSI_FC(dtrcon, DTRCON)
  ((char *)&norm, (char *)&uplo, (char *)&diag, (hilucsi_lapack_int *)&n,
   (double *)a, (hilucsi_lapack_int *)&lda, &rcond, work, iwork, &info);
  return info;
}

/// \brief single version triangular condition number estimator
/// \note This is needed to determined the rank for QRCP, i.e. TQRCP
inline hilucsi_lapack_int trcon(const char norm, const char uplo,
                                const char diag, const hilucsi_lapack_int n,
                                const float *a, const hilucsi_lapack_int lda,
                                float &rcond, float *work,
                                hilucsi_lapack_int *iwork) {
  hilucsi_lapack_int info;
  HILUCSI_FC(strcon, STRCON)
  ((char *)&norm, (char *)&uplo, (char *)&diag, (hilucsi_lapack_int *)&n,
   (float *)a, (hilucsi_lapack_int *)&lda, &rcond, work, iwork, &info);
  return info;
}

/// \brief double version for handling Householder reflectors
inline hilucsi_lapack_int ormqr(const char side, const char trans,
                                const hilucsi_lapack_int m,
                                const hilucsi_lapack_int n,
                                const hilucsi_lapack_int k, const double *a,
                                const hilucsi_lapack_int lda, const double *tau,
                                double *c, const hilucsi_lapack_int ldc,
                                double *work, const hilucsi_lapack_int lwork) {
  hilucsi_lapack_int info;
  HILUCSI_FC(dormqr, DORMQR)
  ((char *)&side, (char *)&trans, (hilucsi_lapack_int *)&m,
   (hilucsi_lapack_int *)&n, (hilucsi_lapack_int *)&k, (double *)a,
   (hilucsi_lapack_int *)&lda, (double *)tau, c, (hilucsi_lapack_int *)&ldc,
   work, (hilucsi_lapack_int *)&lwork, &info);
  return info;
}

/// \brief single version for handling Householder reflectors
inline hilucsi_lapack_int ormqr(const char side, const char trans,
                                const hilucsi_lapack_int m,
                                const hilucsi_lapack_int n,
                                const hilucsi_lapack_int k, const float *a,
                                const hilucsi_lapack_int lda, const float *tau,
                                float *c, const hilucsi_lapack_int ldc,
                                float *work, const hilucsi_lapack_int lwork) {
  hilucsi_lapack_int info;
  HILUCSI_FC(sormqr, SORMQR)
  ((char *)&side, (char *)&trans, (hilucsi_lapack_int *)&m,
   (hilucsi_lapack_int *)&n, (hilucsi_lapack_int *)&k, (float *)a,
   (hilucsi_lapack_int *)&lda, (float *)tau, c, (hilucsi_lapack_int *)&ldc,
   work, (hilucsi_lapack_int *)&lwork, &info);
  return info;
}

/*!
 * @}
 */  // sss group

}  // namespace internal
}  // namespace hilucsi

#endif  // _HILUCSI_SMALLSCALE_LAPACK_QRCP_HPP