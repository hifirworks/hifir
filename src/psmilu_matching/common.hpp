//@HEADER
//----------------------------------------------------------------------------
//                Copyright (C) 2019 The PSMILU AUTHORS
//----------------------------------------------------------------------------
//@HEADER

/// \file psmilu_matching/common.hpp
/// \brief common routines shared by matching/scaling
/// \authors Qiao,

#ifndef _PSMILU_MATCHING_COMMON_HPP
#define _PSMILU_MATCHING_COMMON_HPP

#include <algorithm>
#include <new>

#include "psmilu_log.hpp"
#include "psmilu_utils.hpp"

namespace psmilu {

/*!
 * \addtogroup pre
 * @{
 */

template <bool IsSymm, class CrsType, class CcsType, class T = void>
inline void scale_extreme_values(const CcsType &A, CrsType &B,
                                 typename CrsType::array_type &rs,
                                 typename CrsType::array_type &cs,
                                 const bool ensure_fortran_index = true) {
  static_assert(CrsType::ROW_MAJOR, "must be CRS");
  static_assert(!CcsType::ROW_MAJOR, "must be CCS");
  using value_type                      = typename CrsType::value_type;
  using size_type                       = typename CrsType::size_type;
  constexpr static value_type ZERO      = Const<value_type>::ZERO;
  constexpr static bool       ONE_BASED = CrsType::ONE_BASED;

  const size_type n = B.nrows();

  // we scale row first
  for (size_type row(0); row < n; ++row) {
    value_type tmp = B.nnz_in_row(row)
                         ? std::abs(*std::max_element(
                               B.val_cbegin(row), B.val_cend(row),
                               [](const value_type l, const value_type r) {
                                 return std::abs(l) < std::abs(r);
                               }))
                         : ZERO;
    if (tmp == ZERO) tmp = 1;
    tmp     = 1. / tmp;
    rs[row] = tmp;
    for (auto itr = B.val_begin(row), last = B.val_end(row); itr != last; ++itr)
      *itr *= tmp;
  }

  if (IsSymm)
    std::copy_n(rs.cbegin(), n, cs.begin());
  else
    for (size_type col(0); col < n; ++col) {
      // NOTE that B might be a leading block of A, but we just take the
      // whole A into consideration. This may only happen with real PS systems
      value_type tmp = A.nnz_in_col(col)
                           ? std::abs(*std::max_element(
                                 A.val_cbegin(col), A.val_cend(col),
                                 [](const value_type l, const value_type r) {
                                   return std::abs(l) < std::abs(r);
                                 }))
                           : ZERO;
      if (tmp == ZERO) tmp = 1;
      cs[col] = 1. / tmp;
    }

  // scale B col-wise
  const size_type nz = B.nnz();
  for (size_type i(0); i < nz; ++i)
    B.vals()[i] *= cs[B.col_ind()[i] - ONE_BASED];

  if (!ONE_BASED && ensure_fortran_index) {
    using index_type = typename CrsType::index_type;
    std::for_each(B.row_start().begin(), B.row_start().end(),
                  [](index_type &i) { ++i; });
    std::for_each(B.col_ind().begin(), B.col_ind().end(),
                  [](index_type &i) { ++i; });
  }
}

template <class T, class IndexArray>
inline T *ensure_type_consistency(const IndexArray &v,
                                  const bool        copy_if_needed = true) {
  constexpr static bool consist =
      sizeof(T) == sizeof(typename IndexArray::value_type);
  if (consist) return (T *)v.data();
  T *ptr = new (std::nothrow) T[v.size()];
  psmilu_error_if(!ptr, "memory allocation failed");
  if (copy_if_needed) std::copy(v.cbegin(), v.cend(), ptr);
  return ptr;
}

/*!
 * @}
 */

}  // namespace psmilu

#endif  // _PSMILU_MATCHING_COMMON_HPP