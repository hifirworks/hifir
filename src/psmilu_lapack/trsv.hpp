//@HEADER
//----------------------------------------------------------------------------
//                Copyright (C) 2019 The PSMILU AUTHORS
//----------------------------------------------------------------------------
//@HEADER

/// \file psmilu_lapack/trsv.hpp
/// \brief Interface wrappers for solving triangular systems
/// \authors Qiao,
/// \todo add interfaces for complex types

#ifndef _PSMILU_LAPACK_TRSV_HPP
#define _PSMILU_LAPACK_TRSV_HPP

#include "psmilu_lapack/config.h"

#ifndef DOXYGEN_SHOULD_SKIP_THIS

extern "C" {

// double
void PSMILU_FC(dtrsv, DTRSV)(char *, char *, char *, psmilu_lapack_int *,
                             double *, psmilu_lapack_int *, double *,
                             psmilu_lapack_int *);

// single
void PSMILU_FC(strsv, STRSV)(char *, char *, char *, psmilu_lapack_int *,
                             float *, psmilu_lapack_int *, float *,
                             psmilu_lapack_int *);
}

#endif  // DOXYGEN_SHOULD_SKIP_THIS

namespace psmilu {
namespace internal {

// double
inline void trsv(const char uplo, const char trans, const char diag,
                 const psmilu_lapack_int n, const double *a,
                 const psmilu_lapack_int lda, double *x,
                 const psmilu_lapack_int incx) {
  PSMILU_FC(dtrsv, DTRSV)
  ((char *)&uplo, (char *)&trans, (char *)&diag, (psmilu_lapack_int *)&n,
   (double *)a, (psmilu_lapack_int *)&lda, x, (psmilu_lapack_int *)&incx);
}

// single
inline void trsv(const char uplo, const char trans, const char diag,
                 const psmilu_lapack_int n, const float *a,
                 const psmilu_lapack_int lda, float *x,
                 const psmilu_lapack_int incx) {
  PSMILU_FC(strsv, STRSV)
  ((char *)&uplo, (char *)&trans, (char *)&diag, (psmilu_lapack_int *)&n,
   (float *)a, (psmilu_lapack_int *)&lda, x, (psmilu_lapack_int *)&incx);
}

}  // namespace internal
}  // namespace psmilu

#endif  // _PSMILU_LAPACK_TRSV_HPP
