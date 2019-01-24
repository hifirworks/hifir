//@HEADER
//----------------------------------------------------------------------------
//                Copyright (C) 2019 The PSMILU AUTHORS
//----------------------------------------------------------------------------
//@HEADER

/// \file psmilu_lapack/lup.hpp
/// \brief Interface wrappers for LU with partial pivoting routines in LAPACK
/// \authors Qiao,
/// \todo add interfaces for complex types

#ifndef _PSMILU_LAPACK_LUP_HPP
#define _PSMILU_LAPACK_LUP_HPP

#include "psmilu_lapack/config.h"

#ifndef DOXYGEN_SHOULD_SKIP_THIS

extern "C" {

// factorization

// double
void PSMILU_FC(dgetrf, DGETRF)(psmilu_lapack_int *, psmilu_lapack_int *,
                               double *, psmilu_lapack_int *,
                               psmilu_lapack_int *, psmilu_lapack_int *);
// single
void PSMILU_FC(sgetrf, SGETRF)(psmilu_lapack_int *, psmilu_lapack_int *,
                               float *, psmilu_lapack_int *,
                               psmilu_lapack_int *, psmilu_lapack_int *);

// solving

// double
void PSMILU_FC(dgetrs, DGETRS)(char *, psmilu_lapack_int *, psmilu_lapack_int *,
                               double *, psmilu_lapack_int *,
                               psmilu_lapack_int *, double *,
                               psmilu_lapack_int *, psmilu_lapack_int *);
// single
void PSMILU_FC(sgetrs, SGETRS)(char *, psmilu_lapack_int *, psmilu_lapack_int *,
                               float *, psmilu_lapack_int *,
                               psmilu_lapack_int *, float *,
                               psmilu_lapack_int *, psmilu_lapack_int *);
}

#endif  // DOXYGEN_SHOULD_SKIP_THIS

namespace psmilu {
namespace internal {
/// \brief double version of LU with partial pivoting
inline int getrf(const psmilu_lapack_int m, const psmilu_lapack_int n,
                 double *A, const psmilu_lapack_int lda,
                 psmilu_lapack_int *ipiv) {
  psmilu_lapack_int info;
  PSMILU_FC(dgetrf, DGETRF)
  ((psmilu_lapack_int *)&m, (psmilu_lapack_int *)&n, A,
   (psmilu_lapack_int *)&lda, ipiv, &info);
  return info;
}

/// \brief single version of LU with partial pivoting
inline int getrf(const psmilu_lapack_int m, const psmilu_lapack_int n, float *A,
                 const psmilu_lapack_int lda, psmilu_lapack_int *ipiv) {
  psmilu_lapack_int info;
  PSMILU_FC(sgetrf, SGETRF)
  ((psmilu_lapack_int *)&m, (psmilu_lapack_int *)&n, A,
   (psmilu_lapack_int *)&lda, ipiv, &info);
  return info;
}

/// \brief solve with factorized matrix, double version
///
/// The parameter \a tran should be 0, 1, 2, which means 'N', 'T', and 'C',
/// resp, for the original lapack routine
inline int getrs(const int tran, const psmilu_lapack_int n,
                 const psmilu_lapack_int nrhs, const double *a,
                 const psmilu_lapack_int lda, const psmilu_lapack_int *ipiv,
                 double *b, const psmilu_lapack_int ldb) {
  static char       trans[3] = {'N', 'T', 'C'};
  psmilu_lapack_int info;
  PSMILU_FC(dgetrs, DGETRS)
  (trans + tran, (psmilu_lapack_int *)&n, (psmilu_lapack_int *)&nrhs,
   (double *)a, (psmilu_lapack_int *)&lda, (psmilu_lapack_int *)ipiv, b,
   (psmilu_lapack_int *)&ldb, &info);
  return info;
}

/// \brief solve with factorized matrix, single version
///
/// The parameter \a tran should be 0, 1, 2, which means 'N', 'T', and 'C',
/// resp, for the original lapack routine
inline int getrs(const int tran, const psmilu_lapack_int n,
                 const psmilu_lapack_int nrhs, const float *a,
                 const psmilu_lapack_int lda, const psmilu_lapack_int *ipiv,
                 float *b, const psmilu_lapack_int ldb) {
  static char       trans[3] = {'N', 'T', 'C'};
  psmilu_lapack_int info;
  PSMILU_FC(sgetrs, SGETRS)
  (trans + tran, (psmilu_lapack_int *)&n, (psmilu_lapack_int *)&nrhs,
   (float *)a, (psmilu_lapack_int *)&lda, (psmilu_lapack_int *)ipiv, b,
   (psmilu_lapack_int *)&ldb, &info);
  return info;
}

}  // namespace internal
}  // namespace psmilu

#endif  // _PSMILU_LAPACK_LUP_HPP
