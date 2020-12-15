///////////////////////////////////////////////////////////////////////////////
//                  This file is part of HIF project                         //
///////////////////////////////////////////////////////////////////////////////

/*!
 * \file hif/small_scale/lup_lapack.hpp
 * \brief Interface wrappers for LU with partial pivoting routines in LAPACK
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

#ifndef _HIF_SMALLSCALE_LAPACK_LUP_HPP
#define _HIF_SMALLSCALE_LAPACK_LUP_HPP

#include <complex>

#include "hif/small_scale/config.hpp"

#ifndef DOXYGEN_SHOULD_SKIP_THIS
#  if HIF_HAS_MKL == 0
extern "C" {

// factorization

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

/*!
 * @}
 */  // sss group

}  // namespace internal
}  // namespace hif

#endif  // _HIF_SMALLSCALE_LAPACK_LUP_HPP
