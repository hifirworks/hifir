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

}  // namespace internal
}  // namespace psmilu

#endif  // _PSMILU_LAPACK_QRCP_HPP