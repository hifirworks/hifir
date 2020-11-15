///////////////////////////////////////////////////////////////////////////////
//                This file is part of HILUCSI project                       //
///////////////////////////////////////////////////////////////////////////////

/*!
 * \file hilucsi/small_scale/lup_lapack.hpp
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

#ifndef _HILUCSI_SMALLSCALE_LAPACK_LUP_HPP
#define _HILUCSI_SMALLSCALE_LAPACK_LUP_HPP

#include "hilucsi/small_scale/config.hpp"

#ifndef DOXYGEN_SHOULD_SKIP_THIS
#  if HILUCSI_HAS_MKL == 0
extern "C" {

// factorization

// double
void HILUCSI_FC(dgetrf, DGETRF)(hilucsi_lapack_int *, hilucsi_lapack_int *,
                                double *, hilucsi_lapack_int *,
                                hilucsi_lapack_int *, hilucsi_lapack_int *);
// single
void HILUCSI_FC(sgetrf, SGETRF)(hilucsi_lapack_int *, hilucsi_lapack_int *,
                                float *, hilucsi_lapack_int *,
                                hilucsi_lapack_int *, hilucsi_lapack_int *);

// solving

// double
void HILUCSI_FC(dgetrs,
                DGETRS)(char *, hilucsi_lapack_int *, hilucsi_lapack_int *,
                        double *, hilucsi_lapack_int *, hilucsi_lapack_int *,
                        double *, hilucsi_lapack_int *, hilucsi_lapack_int *);
// single
void HILUCSI_FC(sgetrs,
                SGETRS)(char *, hilucsi_lapack_int *, hilucsi_lapack_int *,
                        float *, hilucsi_lapack_int *, hilucsi_lapack_int *,
                        float *, hilucsi_lapack_int *, hilucsi_lapack_int *);
}
#  endif
#endif  // DOXYGEN_SHOULD_SKIP_THIS

namespace hilucsi {
namespace internal {

/*!
 * \addtogroup sss
 * @{
 */

/// \brief double version of LU with partial pivoting
inline hilucsi_lapack_int getrf(const hilucsi_lapack_int m,
                                const hilucsi_lapack_int n, double *A,
                                const hilucsi_lapack_int lda,
                                hilucsi_lapack_int *     ipiv) {
  hilucsi_lapack_int info;
  HILUCSI_FC(dgetrf, DGETRF)
  ((hilucsi_lapack_int *)&m, (hilucsi_lapack_int *)&n, A,
   (hilucsi_lapack_int *)&lda, ipiv, &info);
  return info;
}

/// \brief single version of LU with partial pivoting
inline hilucsi_lapack_int getrf(const hilucsi_lapack_int m,
                                const hilucsi_lapack_int n, float *A,
                                const hilucsi_lapack_int lda,
                                hilucsi_lapack_int *     ipiv) {
  hilucsi_lapack_int info;
  HILUCSI_FC(sgetrf, SGETRF)
  ((hilucsi_lapack_int *)&m, (hilucsi_lapack_int *)&n, A,
   (hilucsi_lapack_int *)&lda, ipiv, &info);
  return info;
}

/// \brief solve with factorized matrix, double version
inline hilucsi_lapack_int getrs(const char tran, const hilucsi_lapack_int n,
                                const hilucsi_lapack_int nrhs, const double *a,
                                const hilucsi_lapack_int  lda,
                                const hilucsi_lapack_int *ipiv, double *b,
                                const hilucsi_lapack_int ldb) {
  hilucsi_lapack_int info;
  HILUCSI_FC(dgetrs, DGETRS)
  ((char *)&tran, (hilucsi_lapack_int *)&n, (hilucsi_lapack_int *)&nrhs,
   (double *)a, (hilucsi_lapack_int *)&lda, (hilucsi_lapack_int *)ipiv, b,
   (hilucsi_lapack_int *)&ldb, &info);
  return info;
}

/// \brief solve with factorized matrix, single version
inline hilucsi_lapack_int getrs(const char tran, const hilucsi_lapack_int n,
                                const hilucsi_lapack_int nrhs, const float *a,
                                const hilucsi_lapack_int  lda,
                                const hilucsi_lapack_int *ipiv, float *b,
                                const hilucsi_lapack_int ldb) {
  hilucsi_lapack_int info;
  HILUCSI_FC(sgetrs, SGETRS)
  ((char *)&tran, (hilucsi_lapack_int *)&n, (hilucsi_lapack_int *)&nrhs,
   (float *)a, (hilucsi_lapack_int *)&lda, (hilucsi_lapack_int *)ipiv, b,
   (hilucsi_lapack_int *)&ldb, &info);
  return info;
}

/*!
 * @}
 */  // sss group

}  // namespace internal
}  // namespace hilucsi

#endif  // _HILUCSI_SMALLSCALE_LAPACK_LUP_HPP
