//@HEADER
//----------------------------------------------------------------------------
//                Copyright (C) 2019 The HILUCSI AUTHORS
//----------------------------------------------------------------------------
//@HEADER

/// \file hilucsi/small_scale/trsv.hpp
/// \brief Interface wrappers for solving triangular systems
/// \authors Qiao,
/// \todo add interfaces for complex types

#ifndef _HILUCSI_SMALLSCALE_LAPACK_TRSV_HPP
#define _HILUCSI_SMALLSCALE_LAPACK_TRSV_HPP

#include "hilucsi/small_scale/config.hpp"

#ifndef DOXYGEN_SHOULD_SKIP_THIS

extern "C" {

// double
void HILUCSI_FC(dtrsv, DTRSV)(char *, char *, char *, hilucsi_lapack_int *,
                              double *, hilucsi_lapack_int *, double *,
                              hilucsi_lapack_int *);

// single
void HILUCSI_FC(strsv, STRSV)(char *, char *, char *, hilucsi_lapack_int *,
                              float *, hilucsi_lapack_int *, float *,
                              hilucsi_lapack_int *);
}

#endif  // DOXYGEN_SHOULD_SKIP_THIS

namespace hilucsi {
namespace internal {

/*!
 * \addtogroup sss
 * @{
 */

/// \brief double version of triangular solve
inline void trsv(const char uplo, const char trans, const char diag,
                 const hilucsi_lapack_int n, const double *a,
                 const hilucsi_lapack_int lda, double *x,
                 const hilucsi_lapack_int incx) {
  HILUCSI_FC(dtrsv, DTRSV)
  ((char *)&uplo, (char *)&trans, (char *)&diag, (hilucsi_lapack_int *)&n,
   (double *)a, (hilucsi_lapack_int *)&lda, x, (hilucsi_lapack_int *)&incx);
}

/// \brief single version of triangular solve
inline void trsv(const char uplo, const char trans, const char diag,
                 const hilucsi_lapack_int n, const float *a,
                 const hilucsi_lapack_int lda, float *x,
                 const hilucsi_lapack_int incx) {
  HILUCSI_FC(strsv, STRSV)
  ((char *)&uplo, (char *)&trans, (char *)&diag, (hilucsi_lapack_int *)&n,
   (float *)a, (hilucsi_lapack_int *)&lda, x, (hilucsi_lapack_int *)&incx);
}

/*!
 * @}
 */  // sss group

}  // namespace internal
}  // namespace hilucsi

#endif  // _HILUCSI_SMALLSCALE_LAPACK_TRSV_HPP
