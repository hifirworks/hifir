///////////////////////////////////////////////////////////////////////////////
//                This file is part of HILUCSI project                       //
///////////////////////////////////////////////////////////////////////////////

/*!
 * \file hilucsi/utils/mt_mv.hpp
 * \brief Multithreaded matrix-vector product for \a CRS
 * \author Qiao Chen

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

#ifndef _HILUCSI_UTILS_MTMV_HPP
#define _HILUCSI_UTILS_MTMV_HPP

#include <functional>
#include <type_traits>

#include "hilucsi/utils/log.hpp"
#include "hilucsi/utils/mt.hpp"

namespace hilucsi {
namespace mt {

/*!
 * \addtogroup ds
 * @{
 */

/// \brief matrix vector in parallel with openmp
/// \tparam CsType compressed storage type, e.g. \a CRS
/// \tparam IArray input array type
/// \tparam OArray output array type
/// \param[in] A input matrix
/// \param[in] x input array
/// \param[out] y output array
/// \note Sizes must match
template <class CsType, class IArray, class OArray, typename T = void>
inline typename std::enable_if<CsType::ROW_MAJOR, T>::type mv_nt(
    const CsType &A, const IArray &x, OArray &y) {
  hilucsi_error_if(A.nrows() != y.size() || A.ncols() != x.size(),
                   "matrix vector multiplication unmatched sizes!");
  int nthreads = get_nthreads();
  if (nthreads == 0) nthreads = get_nthreads(-1);
  if (nthreads == 1 || (A.nrows() < 1000u && A.nnz() / (double)A.nrows() <= 20))
    return A.mv_nt(x, y);
#ifdef _OPENMP
#  pragma omp parallel num_threads(nthreads)
#endif
  do {
    const auto part = uniform_partition(A.nrows(), nthreads, get_thread());
    A.mv_nt_low(x.data(), part.first, part.second - part.first, y.data());
  } while (false);  // parallel region
}

/// \brief matrix vector in parallel with openmp
/// \tparam CsType compressed storage type, e.g. \a CRS
/// \tparam Vx other value type for \a x
/// \tparam Vy other value type for \a y
/// \param[in] A input matrix
/// \param[in] x input array
/// \param[out] y output array
/// \note Sizes must match
template <class CsType, class Vx, class Vy, typename T = void>
inline typename std::enable_if<CsType::ROW_MAJOR, T>::type mv_nt_low(
    const CsType &A, const Vx *x, Vy *y) {
  int nthreads = get_nthreads();
  if (nthreads == 0) nthreads = get_nthreads(-1);
  if (nthreads == 1 || (A.nrows() < 1000u && A.nnz() / (double)A.nrows() <= 20))
    return A.mv_nt_low(x, y);
#ifdef _OPENMP
#  pragma omp parallel num_threads(nthreads)
#endif
  do {
    const auto part = uniform_partition(A.nrows(), nthreads, get_thread());
    A.mv_nt_low(x, part.first, part.second - part.first, y);
  } while (false);  // parallel region
}

/// \brief CCS API compatibility
template <class CsType, class IArray, class OArray, typename T = void>
inline typename std::enable_if<!CsType::ROW_MAJOR, T>::type mv_nt(
    const CsType &A, const IArray &x, OArray &y) {
  hilucsi_warning("CCS does not support threaded matrix-vector!");
  return A.mv_nt(x, y);
}

/// \brief CCS API compatibility
template <class CsType, class Vx, class Vy, typename T = void>
inline typename std::enable_if<!CsType::ROW_MAJOR, T>::type mv_nt_low(
    const CsType &A, const Vx *x, Vy *y) {
  hilucsi_warning("CCS does not support threaded matrix-vector!");
  A.mv_nt_low(x, y);
}

// enable mt::mv_nt, i.e., multi-threaded matrix-vector no transpose for
// functor A. Notice that because mt::mv_nt is used in both KSP and iterative
// refinement interfaces, thus, overloading this function is the easiest way
// to enable user callback for computing matrix-vector product.
// NOTE: This is for interface compatibility!

/// \brief use user functor for computing "matrix"-vector product
/// \tparam ArrayType array type, see \ref Array
/// \param[in] A callable functor
/// \param[in] x array to multiply with
/// \param[out] y y=A*x
/// \note A(x, y) should return, conceptually, y=A*x
/// \ingroup ksp
template <class ArrayType>
inline void mv_nt(const std::function<void(const ArrayType &, ArrayType &)> &A,
                  const ArrayType &x, ArrayType &y) {
  A(x, y);
}

/*!
 * @}
 */

}  // namespace mt
}  // namespace hilucsi

#endif  // _HILUCSI_UTILS_MTMV_HPP