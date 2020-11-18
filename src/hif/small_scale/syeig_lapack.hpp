///////////////////////////////////////////////////////////////////////////////
//                  This file is part of HIF project                         //
///////////////////////////////////////////////////////////////////////////////

/*!
 * \file hif/small_scale/syeig_lapack.hpp
 * \brief Interface wrappers for symmetric eigen routines in LAPACK
 * \author Qiao Chen
 * \todo add interfaces for complex Hermitian types

\verbatim
Copyright (C) 2020 NumGeom Group at Stony Brook University

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

#ifndef _HIF_SMALLSCALE_LAPACK_SYEIG_HPP
#define _HIF_SMALLSCALE_LAPACK_SYEIG_HPP

#include "hif/small_scale/config.hpp"

#ifndef DOXYGEN_SHOULD_SKIP_THIS
#  if HIF_HAS_MKL == 0
extern "C" {

// factorization

// double
void HIF_FC(dsyev, DSYEV)(char *, char *, hif_lapack_int *, double *,
                          hif_lapack_int *, double *, double *,
                          hif_lapack_int *, hif_lapack_int *);
// single
void HIF_FC(ssyev, SSYEV)(char *, char *, hif_lapack_int *, float *,
                          hif_lapack_int *, float *, float *, hif_lapack_int *,
                          hif_lapack_int *);

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
}
#  endif
#endif  // DOXYGEN_SHOULD_SKIP_THIS

namespace hif {

namespace internal {

/*!
 * \addtogroup sss
 * @{
 */

/// \brief double version of handling symmetric eigendecomposition
inline hif_lapack_int syev(const char uplo, const hif_lapack_int n, double *a,
                           const hif_lapack_int lda, double *w, double *work,
                           const hif_lapack_int lwork) {
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
                           const hif_lapack_int lwork) {
  static char    jobz('V');
  hif_lapack_int info;
  HIF_FC(ssyev, SSYEV)
  (&jobz, (char *)&uplo, (hif_lapack_int *)&n, a, (hif_lapack_int *)&lda, w,
   work, (hif_lapack_int *)&lwork, &info);
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

/*!
 * @}
 */  // sss group

}  // namespace internal

}  // namespace hif

#endif  // _HIF_SMALLSCALE_LAPACK_SYEIG_HPP