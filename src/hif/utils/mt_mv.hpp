///////////////////////////////////////////////////////////////////////////////
//                  This file is part of HIF project                         //
///////////////////////////////////////////////////////////////////////////////

/*!
 * \file hif/utils/mt_mv.hpp
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

#ifndef _HIF_UTILS_MTMV_HPP
#define _HIF_UTILS_MTMV_HPP

#include <array>
#include <functional>
#include <type_traits>

#include "hif/ds/Array.hpp"
#include "hif/utils/common.hpp"
#include "hif/utils/log.hpp"
#include "hif/utils/mt.hpp"

namespace hif {
namespace mt {

/*!
 * \addtogroup alg
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
  hif_error_if(A.nrows() != y.size() || A.ncols() != x.size(),
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
  hif_warning("CCS does not support threaded matrix-vector!");
  return A.mv_nt(x, y);
}

/// \brief CCS API compatibility
template <class CsType, class Vx, class Vy, typename T = void>
inline typename std::enable_if<!CsType::ROW_MAJOR, T>::type mv_nt_low(
    const CsType &A, const Vx *x, Vy *y) {
  hif_warning("CCS does not support threaded matrix-vector!");
  A.mv_nt_low(x, y);
}

// multiple RHS

/// \brief matrix vector in parallel with openmp for multiple RHS
/// \tparam CsType compressed storage type, e.g. \a CRS
/// \tparam InType input data type
/// \tparam OutType output data type
/// \tparam Nrhs number of RHS
/// \param[in] A input matrix
/// \param[in] x input array
/// \param[out] y output array
template <class CsType, class InType, class OutType, std::size_t Nrhs,
          typename T = void>
inline typename std::enable_if<CsType::ROW_MAJOR, T>::type mv_mrhs_nt(
    const CsType &A, const Array<std::array<InType, Nrhs>> &x,
    Array<std::array<OutType, Nrhs>> &y) {
  hif_error_if(A.nrows() != y.size() || A.ncols() != x.size(),
               "matrix vector multiplication unmatched sizes!");
  int nthreads = get_nthreads();
  if (nthreads == 0) nthreads = get_nthreads(-1);
  if (nthreads == 1 ||
      (A.nrows() < 1000u && Nrhs * A.nnz() / (double)A.nrows() <= 20))
    return A.mv_mrhs_nt(x, y);
#ifdef _OPENMP
#  pragma omp parallel num_threads(nthreads)
#endif
  do {
    const auto part = uniform_partition(A.nrows(), nthreads, get_thread());
    A.mv_mrhs_nt_low<Nrhs>(x[0].data(), part.first, part.second - part.first,
                           y[0].data());
  } while (false);  // parallel region
}

/// \brief matrix vector in parallel with openmp for multiple RHS
/// \tparam Nrhs number of RHS
/// \tparam CsType compressed storage type, e.g. \a CRS
/// \tparam Vx other value type for \a x
/// \tparam Vy other value type for \a y
/// \param[in] A input matrix
/// \param[in] x input array
/// \param[out] y output array
/// \note Sizes must match
template <std::size_t Nrhs, class CsType, class Vx, class Vy, typename T = void>
inline typename std::enable_if<CsType::ROW_MAJOR, T>::type mv_mrhs_nt_low(
    const CsType &A, const Vx *x, Vy *y) {
  int nthreads = get_nthreads();
  if (nthreads == 0) nthreads = get_nthreads(-1);
  if (nthreads == 1 ||
      (A.nrows() < 1000u && Nrhs * A.nnz() / (double)A.nrows() <= 20))
    return A.mv_mrhs_nt_low<Nrhs>(x, y);
#ifdef _OPENMP
#  pragma omp parallel num_threads(nthreads)
#endif
  do {
    const auto part = uniform_partition(A.nrows(), nthreads, get_thread());
    A.mv_mrhs_nt_low<Nrhs>(x, part.first, part.second - part.first, y);
  } while (false);  // parallel region
}

/// \brief matrix vector in parallel with openmp for multiple RHS for \a CCS
template <class CsType, class InType, class OutType, std::size_t Nrhs,
          typename T = void>
inline typename std::enable_if<!CsType::ROW_MAJOR, T>::type mv_mrhs_nt(
    const CsType &A, const Array<std::array<InType, Nrhs>> &x,
    Array<std::array<OutType, Nrhs>> &y) {
  hif_warning("CCS does not support threaded matrix-vector!");
  return A.mv_mrhs_nt(x, y);
}

/// \brief matrix vector in parallel with openmp for multiple RHS for \a CCS
template <std::size_t Nrhs, class CsType, class Vx, class Vy, typename T = void>
inline typename std::enable_if<!CsType::ROW_MAJOR, T>::type mv_mrhs_nt_low(
    const CsType &A, const Vx *x, Vy *y) {
  hif_warning("CCS does not support threaded matrix-vector!");
  return A.mv_mrhs_nt_low<Nrhs>(x, y);
}

/*!
 * @}
 */

// enable mt::mv_nt, i.e., multi-threaded matrix-vector no transpose for
// functor A. Notice that because mt::mv_nt is used in both KSP and iterative
// refinement interfaces, thus, overloading this function is the easiest way
// to enable user callback for computing matrix-vector product.
// NOTE: This is for interface compatibility!

/// \brief use user functor for computing "matrix"-vector product
/// \tparam IArray input array type
/// \tparam OArray output array type
/// \param[in] Afunc callable functor
/// \param[in] x array to multiply with
/// \param[out] y y=A*x
/// \ingroup ksp
template <class IArray, class OArray>
inline void mv_nt(
    const std::function<void(const void *, const typename IArray::size_type,
                             const char, void *, const char, const bool)>
        &         Afunc,
    const IArray &x, OArray &y) {
  using i_value_type = typename IArray::value_type;
  using o_value_type = typename OArray::value_type;
  Afunc((const void *)x.data(), x.size(),
        ValueTypeTrait<i_value_type>::signature, (void *)y.data(),
        ValueTypeTrait<o_value_type>::signature, false);
}

#if 0
/// \brief use user functor for computing "matrix"-vector product with
///        multiple RHS
/// \tparam InType input data type
/// \tparam OutType output data type
/// \tparam Nrhs number of RHS
/// \param[in] Afunc callable functor
/// \param[in] x array to multiply with
/// \param[out] y y=A*x
/// \ingroup ksp
template <class InType, class OutType, std::size_t Nrhs>
inline void mv_mrhs_nt(
    const std::function<void(const void *, const typename std::size_t,
                             const typename std::size_t, const char, void *,
                             const char)> &Afunc,
    const Array<std::array<InType, Nrhs>> &x,
    Array<std::array<OutType, Nrhs>> &     y) {
  if (x.size() && y.size())
    Afunc((const void *)x[0].data(), x.size(), Nrhs,
          ValueTypeTrait<InType>::signature, (void *)y[0].data(),
          ValueTypeTrait<OutType>::signature);
}
#endif

}  // namespace mt
}  // namespace hif

#endif  // _HIF_UTILS_MTMV_HPP