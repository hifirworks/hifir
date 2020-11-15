///////////////////////////////////////////////////////////////////////////////
//                  This file is part of HIF project                         //
///////////////////////////////////////////////////////////////////////////////

/*!
 * \file hif/small_scale/trsv_lapack.hpp
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

#ifndef _HIF_SMALLSCALE_LAPACK_TRSV_HPP
#define _HIF_SMALLSCALE_LAPACK_TRSV_HPP

#include "hif/small_scale/config.hpp"

#ifndef DOXYGEN_SHOULD_SKIP_THIS
#  if HIF_HAS_MKL == 0
extern "C" {

// double
void HIF_FC(dtrsv, DTRSV)(char *, char *, char *, hif_lapack_int *, double *,
                          hif_lapack_int *, double *, hif_lapack_int *);

// single
void HIF_FC(strsv, STRSV)(char *, char *, char *, hif_lapack_int *, float *,
                          hif_lapack_int *, float *, hif_lapack_int *);
}
#  endif
#endif  // DOXYGEN_SHOULD_SKIP_THIS

namespace hif {
namespace internal {

/*!
 * \addtogroup sss
 * @{
 */

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

/*!
 * @}
 */  // sss group

}  // namespace internal
}  // namespace hif

#endif  // _HIF_SMALLSCALE_LAPACK_TRSV_HPP
