///////////////////////////////////////////////////////////////////////////////
//                This file is part of HILUCSI project                       //
///////////////////////////////////////////////////////////////////////////////

/*!
 * \file hilucsi/small_scale/trsv_lapack.hpp
 * \brief Interface wrappers for solving triangular systems
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

#ifndef _HILUCSI_SMALLSCALE_LAPACK_TRSV_HPP
#define _HILUCSI_SMALLSCALE_LAPACK_TRSV_HPP

#include "hilucsi/small_scale/config.hpp"

#ifndef DOXYGEN_SHOULD_SKIP_THIS
#  if HILUCSI_HAS_MKL == 0
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
#  endif
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
