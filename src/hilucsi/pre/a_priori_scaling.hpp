///////////////////////////////////////////////////////////////////////////////
//                This file is part of HILUCSI project                       //
///////////////////////////////////////////////////////////////////////////////

/*!
 * \file hilucsi/pre/a_priori_scaling.hpp
 * \brief *a priori* scaling before official matching/scaling and reordering
 * \author Qiao Chen

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

#ifndef _HILUCSI_PRE_APRIORISCALING_HPP
#define _HILUCSI_PRE_APRIORISCALING_HPP

#include <algorithm>
#include <cmath>
#include <new>

#include "hilucsi/utils/common.hpp"
#include "hilucsi/utils/log.hpp"

namespace hilucsi {

/*!
 * \addtogroup pre
 * @{
 */

/// \brief do not perform a priori scaling
/// \tparam CrsType crs matrix type, see \ref CRS
/// \param[in,out] B input and output matrix
/// \param[out] rs row scaling
/// \param[out] cs column scaling
/// \param[in] ensure_fortran_index if \a true (default), then the index values
///            in \a B will be advanced by 1
///
/// Notice that \a rs and \a cs will be set to be 1.
template <class CrsType>
inline void scale_eye(CrsType &B, typename CrsType::array_type &rs,
                      typename CrsType::array_type &cs,
                      const bool ensure_fortran_index = true) {
  using size_type  = typename CrsType::size_type;
  using index_type = typename CrsType::index_type;

  const size_type n = B.nrows();
  for (size_type i(0); i < n; ++i) rs[i] = cs[i] = 1;
  if (ensure_fortran_index) {
    std::for_each(B.row_start().begin(), B.row_start().end(),
                  [](index_type &i) { ++i; });
    std::for_each(B.col_ind().begin(), B.col_ind().end(),
                  [](index_type &i) { ++i; });
  }
}

/// \brief scale based on extreme values, do it asynchronously
/// \tparam IsSymm if \a true, then assume the input type is symmetric
/// \tparam CrsType crs matrix type, see \ref CRS
/// \param[in,out] B input and output matrix
/// \param[out] rs row scaling
/// \param[out] cs column scaling
/// \param[in] ensure_fortran_index if \a true (default), then the index values
///            in \a B will be advanced by 1
///
/// This function performs a single step of asynchronous scaling (Gauss-Seidel)
/// scaling, thus the row will be scaled before column scaling computes the
/// local extreme values.
template <bool IsSymm, class CrsType>
inline void scale_extreme_values(CrsType &B, typename CrsType::array_type &rs,
                                 typename CrsType::array_type &cs,
                                 const bool ensure_fortran_index = true) {
  static_assert(CrsType::ROW_MAJOR, "must be CRS");
  using value_type                 = typename CrsType::value_type;
  using size_type                  = typename CrsType::size_type;
  constexpr static value_type ZERO = Const<value_type>::ZERO;

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
    for (size_type i(0); i < nnz; ++i)
      cs[indices[i]] = std::max(std::abs(vals[i]), cs[indices[i]]);
    for (size_type i(0); i < n; ++i) {
      if (cs[i] == ZERO)
        cs[i] = 1;
      else
        cs[i] = 1. / cs[i];
    }
  }

  // scale B col-wise
  const size_type nz = B.nnz();
  for (size_type i(0); i < nz; ++i) B.vals()[i] *= cs[B.col_ind()[i]];

  if (ensure_fortran_index) {
    using index_type = typename CrsType::index_type;
    std::for_each(B.row_start().begin(), B.row_start().end(),
                  [](index_type &i) { ++i; });
    std::for_each(B.col_ind().begin(), B.col_ind().end(),
                  [](index_type &i) { ++i; });
  }
}

/// \brief iterative scaling based on Jacobi operation
/// \tparam IsSymm if \a true, then assume the input type is symmetric
/// \tparam PermType user-defined row permutation (or symmetric if \a IsSymm
///         is \a true.)
/// \tparam CrsType crs matrix type, see \ref CRS
/// \param[in] p user permutation with \a operator[] returns permutation
/// \param[in,out] A input and output operator
/// \param[out] rs row scaling
/// \param[out] cs column scaling
/// \param[in] tol termination tolerance, default is 1e-10
/// \param[in] max_iters maximum number of iterations
/// \param[in] ensure_fortran_index if \a true (default), then the index values
///            in \a B will be advanced by 1
///
/// References : D. Ruiz and B. Ucar, A Symmetry Preserving Algorithm for
///              Matrix Scaling, SIAM J. MATRIX ANAL. APPL. 2014
template <bool IsSymm, class PermType, class CrsType>
inline void iterative_scale_p(const PermType &p, CrsType &A,
                              typename CrsType::array_type &    rs,
                              typename CrsType::array_type &    cs,
                              const double                      tol = 1e-10,
                              const typename CrsType::size_type max_iters = 5,
                              const bool ensure_fortran_index = true) {
  // This implementation is based on Scaling.h in Eigen under unsupported

  using value_type = typename CrsType::value_type;
  using array_type = typename CrsType::array_type;
  using size_type  = typename CrsType::size_type;

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
      auto       last  = A.col_ind_cend(p[row]);
      auto       v_itr = A.val_cbegin(p[row]);
      for (auto itr = A.col_ind_cbegin(p[row]); itr != last; ++itr, ++v_itr) {
        tmp = std::max(std::abs(*v_itr), tmp);
        if (!IsSymm) {
          // assume identity on the column
          const auto j = *itr;
          local_cs[j]  = std::max(std::abs(*v_itr), local_cs[j]);
        }
      }
      local_rs[p[row]] = 1. / std::sqrt(tmp);
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
      auto             last  = A.col_ind_cend(p[row]);
      auto             v_itr = A.val_begin(p[row]);
      const value_type r     = local_rs[p[row]];
      for (auto itr = A.col_ind_cbegin(p[row]); itr != last; ++itr, ++v_itr) {
        const auto j = *itr;
        if (IsSymm)
          *v_itr *= r * local_rs[p[j]];
        else
          *v_itr *= r * local_cs[j];
        tmp = std::max(std::abs(*v_itr), tmp);
        if (!IsSymm) res_cs[j] = std::max(std::abs(*v_itr), res_cs[j]);
      }
      // no need to permute residual array
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
    hilucsi_info("iter-scaling, iter=%zd, res_r=%g, res_c=%g", iters, res_r,
                 res_c);
#endif
  } while ((res_r > tol || res_c > tol) && iters < max_iters);

  if (IsSymm) std::copy_n(rs.cbegin(), m, cs.begin());

  if (ensure_fortran_index) {
    using index_type = typename CrsType::index_type;
    std::for_each(A.row_start().begin(), A.row_start().end(),
                  [](index_type &i) { ++i; });
    std::for_each(A.col_ind().begin(), A.col_ind().end(),
                  [](index_type &i) { ++i; });
  }
}

/// \brief iterative scaling based on Jacobi operation
/// \tparam IsSymm if \a true, then assume the input type is symmetric
/// \tparam CrsType crs matrix type, see \ref CRS
/// \param[in,out] A input and output operator
/// \param[out] rs row scaling
/// \param[out] cs column scaling
/// \param[in] tol termination tolerance, default is 1e-10
/// \param[in] max_iters maximum number of iterations
/// \param[in] ensure_fortran_index if \a true (default), then the index values
///            in \a B will be advanced by 1
///
/// References : D. Ruiz and B. Ucar, A Symmetry Preserving Algorithm for
///              Matrix Scaling, SIAM J. MATRIX ANAL. APPL. 2014
template <bool IsSymm, class CrsType>
inline void iterative_scale(CrsType &A, typename CrsType::array_type &rs,
                            typename CrsType::array_type &    cs,
                            const double                      tol       = 1e-10,
                            const typename CrsType::size_type max_iters = 5,
                            const bool ensure_fortran_index = true) {
  using index_type = typename CrsType::index_type;
  static const struct {
    inline constexpr index_type operator[](const index_type i) const {
      return i;
    }
  } eye_p;  // dummy permutation
  iterative_scale_p<IsSymm>(eye_p, A, rs, cs, tol, max_iters,
                            ensure_fortran_index);
}

/*!
 * @}
 */

}  // namespace hilucsi

#endif  // _HILUCSI_PRE_APRIORISCALING_HPP