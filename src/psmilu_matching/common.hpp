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
#include <cmath>
#include <new>

#include "psmilu_log.hpp"
#include "psmilu_utils.hpp"

namespace psmilu {

/*!
 * \addtogroup pre
 * @{
 */

template <bool IsSymm, class CrsType, class T = void>
inline void scale_extreme_values(CrsType &B, typename CrsType::array_type &rs,
                                 typename CrsType::array_type &cs,
                                 const bool ensure_fortran_index = true) {
  static_assert(CrsType::ROW_MAJOR, "must be CRS");
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
  else {
    // set all values to be 0
    std::fill_n(cs.begin(), n, value_type(0));
    const size_type nnz     = B.nnz();
    const auto &    indices = B.col_ind();
    const auto &    vals    = B.vals();
    for (size_type i(0); i < nnz; ++i) {
      const auto j = indices[i] - ONE_BASED;
      cs[j]        = std::max(std::abs(vals[i]), cs[j]);
    }
    for (size_type i(0); i < n; ++i) {
      if (cs[i] == ZERO)
        cs[i] = 1;
      else
        cs[i] = 1. / cs[i];
    }
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

template <bool IsSymm, class CrsType>
inline void iterative_scale(CrsType &A, typename CrsType::array_type &rs,
                            typename CrsType::array_type &    cs,
                            const double                      tol       = 1e-10,
                            const typename CrsType::size_type max_iters = 5,
                            const bool ensure_fortran_index = true) {
  // This implementation is based on Scaling.h in Eigen
  // see
  // https://bitbucket.org/eigen/eigen/src/bc7e634886a41aa1808e9884446cfbbe3fc16c7b/unsupported/Eigen/src/IterativeSolvers/Scaling.h

  using value_type                = typename CrsType::value_type;
  using array_type                = typename CrsType::array_type;
  using size_type                 = typename CrsType::size_type;
  constexpr static bool ONE_BASED = CrsType::ONE_BASED;

  const size_type m(A.nrows()), n(A.ncols());
  array_type      res_rs(m), res_cs, local_rs(m), local_cs;
  if (!IsSymm) {
    res_cs.resize(n);
    local_cs.resize(n);
  }

  // set all ones
  std::fill_n(rs.begin(), m, 1);
  if (!IsSymm) std::fill_n(cs.begin(), n, 1);
  double    res_r(1.0), res_c(1.0);
  size_type iters(0);
  do {
    // reset all local scaling
    // std::fill(local_rs.begin(), local_rs.end(), 0);
    if (!IsSymm) std::fill(local_cs.begin(), local_cs.end(), 0);
    // loop through all entries
    for (size_type row(0); row < m; ++row) {
      value_type tmp(0);
      auto       last  = A.col_ind_cend(row);
      auto       v_itr = A.val_cbegin(row);
      for (auto itr = A.col_ind_cbegin(row); itr != last; ++itr, ++v_itr) {
        tmp = std::max(std::abs(*v_itr), tmp);
        if (!IsSymm) {
          const auto j = *itr - ONE_BASED;
          local_cs[j]  = std::max(std::abs(*v_itr), local_cs[j]);
        }
      }
      local_rs[row] = 1. / std::sqrt(tmp);
    }
    for (size_type i(0); i < m; ++i) rs[i] *= local_rs[i];
    if (!IsSymm) {
      for (size_type i(0); i < n; ++i) {
        const auto tmp = 1. / std::sqrt(local_cs[i]);
        local_cs[i]    = tmp;
        cs[i] *= tmp;
      }
    }
    if (!IsSymm) std::fill(res_cs.begin(), res_cs.end(), 0);
    for (size_type row(0); row < m; ++row) {
      value_type       tmp(0);
      auto             last  = A.col_ind_cend(row);
      auto             v_itr = A.val_begin(row);
      const value_type r     = local_rs[row];
      for (auto itr = A.col_ind_cbegin(row); itr != last; ++itr, ++v_itr) {
        const auto j = *itr - ONE_BASED;
        if (IsSymm)
          *v_itr *= r * local_rs[j];
        else
          *v_itr *= r * local_cs[j];
        tmp = std::max(std::abs(*v_itr), tmp);
        if (!IsSymm) res_cs[j] = std::max(std::abs(*v_itr), res_cs[j]);
      }
      res_rs[row] = tmp;
    }
    value_type tmp(0);
    for (size_type i(0); i < m; ++i)
      tmp = std::max(std::abs(1 - res_rs[i]), tmp);
    res_r = tmp;
    if (!IsSymm) {
      tmp = 0;
      for (size_type i(0); i < n; ++i)
        tmp = std::max(std::abs(1 - res_cs[i]), tmp);
    }
    res_c = tmp;
    ++iters;
#ifndef NDEBUG
    psmilu_info("iter-scaling, iter=%zd, res_r=%g, res_c=%g", iters, res_r,
                res_c);
#endif
  } while ((res_r > tol || res_c > tol) && iters < max_iters);

  if (IsSymm) std::copy_n(rs.cbegin(), m, cs.begin());

  if (!ONE_BASED && ensure_fortran_index) {
    using index_type = typename CrsType::index_type;
    std::for_each(A.row_start().begin(), A.row_start().end(),
                  [](index_type &i) { ++i; });
    std::for_each(A.col_ind().begin(), A.col_ind().end(),
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