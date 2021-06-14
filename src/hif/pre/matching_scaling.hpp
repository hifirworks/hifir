///////////////////////////////////////////////////////////////////////////////
//                  This file is part of the HIFIR library                   //
///////////////////////////////////////////////////////////////////////////////

/*!
 * \file hif/pre/matching_scaling.hpp
 * \brief calling interface for matching and scaling
 * \author Qiao Chen

\verbatim
Copyright (C) 2021 NumGeom Group at Stony Brook University

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

#ifndef _HIF_PRE_MATCHINGSCALING_HPP
#define _HIF_PRE_MATCHINGSCALING_HPP

#include <algorithm>
#include <cmath>

#include "hif/Options.h"
#include "hif/ds/CompressedStorage.hpp"
#include "hif/pre/EqlDriver.hpp"
#include "hif/utils/Timer.hpp"
#include "hif/utils/common.hpp"

namespace hif {
namespace internal {

/*!
 * \addtogroup pre
 * @{
 */

/// \brief fix poorly scaled row and column weights due to MC64
/// \tparam PermType permutation matrix, see \ref BiPermMatrix
/// \tparam ScalingArray scaling array type, see \ref Array
/// \param[in] m0 leading block size
/// \param[in] level current factorization level
/// \param[in] p row permutation matrix
/// \param[in] q column permutation matrix
/// \param[in,out] s row scaling
/// \param[in,out] t column scaling
/// \param[in] beta (optional) threshold for determining bad scaling factors
template <class PermType, class ScalingArray>
inline void fix_poor_scaling(const typename ScalingArray::size_type m0,
                             const typename ScalingArray::size_type level,
                             const PermType &p, const PermType &q,
                             ScalingArray &s, ScalingArray &t,
                             const double beta = 1e3) {
  using size_type = typename ScalingArray::size_type;

  const double beta0 = beta < 0.0 ? 1e3 : beta;
  // Fix poorly scaled row and column scaling due to MC64
  // We consider a combo of row (s_i) and column (t_i) scaling is bad if
  // beta*min(s_i,t_i)<max(s_i,t_i). We fix it by setting s_i=t_i=sqrt(s_i*t_i)
  // TODO: we need to fine tune beta. 100 for now.
  if (level > 1u && beta0 > 1.0)  // starting from 2nd level only
    for (size_type i(0); i < m0; ++i)
      if (std::min(s[p[i]], t[q[i]]) * beta0 < std::max(s[p[i]], t[q[i]]))
        s[p[i]] = t[q[i]] = std::sqrt(s[p[i]] * t[q[i]]);
}

/// \brief defer any zero diags to the end
/// \tparam IsSymm if \a true, then assume a symmetric leading block
/// \tparam CcsType ccs storage for input matrix
/// \tparam CrsType crs storage for the input matrix
/// \tparam PermType permutation matrix, see \ref BiPermMatrix
/// \tparam BufType buffer used to stored deferred entries
/// \param[in] A input matrix
/// \param[in] A_crs input matrix of CRS storage
/// \param[in] m0 initial leading block size
/// \param[in,out] p row permutation matrix
/// \param[in,out] q column permutation matrix
/// \param[out] work_p workspace
/// \param[out] work_q workspace
/// \return actual leading block with no zero entries, <= \a m0
///
/// For asymmetric cases, where the whole matrix is stored, thus we need binary
/// search to locate the diagonal entries. For symmetric cases, on the other
/// side, since only the \b lower part is stored, we just need to test the first
/// entry of each column.
template <bool IsSymm, class CcsType, class CrsType, class PermType,
          class BufType>
inline typename CcsType::size_type defer_tiny_diags(
    const CcsType &A, const CrsType &A_crs,
    const typename CcsType::size_type m0, PermType &p, PermType &q,
    BufType &work_p, BufType &work_q) {
  using value_type  = typename CcsType::value_type;
  using size_type   = typename CcsType::size_type;
  using scalar_type = typename ValueTypeTrait<value_type>::value_type;
  constexpr static scalar_type EPS = Const<scalar_type>::EPS;

  // lambda for computing local maximum magnitude for a given row *and* column
  const auto compute_max_mag = [&](const size_type i) -> scalar_type {
    scalar_type v(0);
    if (A_crs.nnz_in_primary(p[i]))
      v = std::max(v, std::abs(*std::max_element(
                          A_crs.val_cbegin(p[i]), A_crs.val_cend(p[i]),
                          [](const value_type a, const value_type b) -> bool {
                            return std::abs(a) < std::abs(b);
                          })));
    if (A.nnz_in_primary(q[i]))
      v = std::max(v, std::abs(*std::max_element(
                          A.val_cbegin(q[i]), A.val_cend(q[i]),
                          [](const value_type a, const value_type b) -> bool {
                            return std::abs(a) < std::abs(b);
                          })));
    if (v == Const<scalar_type>::ZERO) v = scalar_type(1);
    return v;
  };

  // lambda for determining small diagonals and perform statical deferring
  const auto is_good_diag = [&](const size_type col) -> bool {
    const auto q_col = q[col];
    if (IsSymm) {
      // NOTE that since we only store the lower part, and due to the the
      // fact of symmetric pivoting, if the original is a saddle point, then
      // it is still a saddle entry in the permutated system. This makes
      // checking for invalid diagonal efficient for symmetric case
      hif_assert(p[col] == q_col, "fatal");
      auto itr = A.row_ind_cbegin(q_col);
      if (itr == A.row_ind_cend(q_col)) return false;
      if (*itr != q_col) return false;
      if (std::abs(*A.val_cbegin(q_col)) <= compute_max_mag(col) * EPS)
        return false;
    } else {
      const auto p_diag = p[col];
      auto       info =
          find_sorted(A.row_ind_cbegin(q_col), A.row_ind_cend(q_col), p_diag);
      if (!info.first) return false;
      // test numerical value
      if (std::abs(
              *(A.vals().cbegin() + (info.second - A.row_ind().cbegin()))) <=
          compute_max_mag(col) * EPS)
        return false;
    }
    return true;
  };

  size_type m = m0;
  size_type deferrals(0);
  for (size_type i(0); i < m; ++i) {
    // check deferring
    while (!is_good_diag(i + deferrals)) {
      --m;
      work_p[deferrals] = p[i + deferrals];
      work_q[deferrals] = q[i + deferrals];
      ++deferrals;
      if (i + deferrals >= m0) {
        m = i;
        break;
      }
    }
    if (m == i) break;
    // compress
    p[i] = p[i + deferrals];
    q[i] = q[i + deferrals];
  }
  if (deferrals) {
    size_type j(0);
    for (size_type i(m); i < m0; ++i, ++j) {
      p[i] = work_p[j];
      q[i] = work_q[j];
    }
  }

  return m;
}

/// \brief permute the matrix for asymmetric cases
/// \tparam CcsType ccs storage for intermidiate matrix after matching
/// \tparam CrsType crs storage, see \ref CRS
/// \tparam PermType permutation matrix, see \ref BiPermMatrix
/// \param[in] A input matrix after performing matching
/// \param[in] A_crs input matrix with crs type
/// \param[in] m leading block size that must no larger than size(A)
/// \param[in] p row permutation matrix
/// \param[in] q column permutation matrix
/// \param[in] apat if \a true, then compute \f$\mathbf{A}_m^T+\mathbf{A}_m\f$
///             default is \a false
/// \note It's worth noting that \a p must be bi-directional mapping
/// \note If \a apat is \a true, then \a q must be bi-directional mapping
template <class CcsType, class CrsType, class PermType>
inline CcsType compute_leading_block(const CcsType &A, const CrsType &A_crs,
                                     const typename CcsType::size_type m,
                                     const PermType &p, const PermType &q,
                                     const bool apat = false) {
  using size_type  = typename CcsType::size_type;
  using index_type = typename CcsType::index_type;

  CcsType B(m, m);
  auto &  col_start = B.col_start();
  col_start.resize(m + 1);
  hif_error_if(col_start.status() == DATA_UNDEF, "memory allocation failed");
  col_start.front() = 0;  // zero based

  if (!apat) {
    // determine nnz
    for (size_type col = 0u; col < m; ++col) {
      const auto q_col = q[col];
      col_start[col + 1] =
          col_start[col] +
          std::count_if(A.row_ind_cbegin(q_col), A.row_ind_cend(q_col),
                        [&](decltype(q_col) i) {
                          return static_cast<size_type>(p.inv(i)) < m;
                        });
    }
    // for (size_type i = 0u; i < m; ++i) col_start[i + 1] += col_start[i];

    // allocate storage
    auto &row_ind = B.row_ind();
    // NOTE we only indices for reordering step
    row_ind.resize(col_start[m]);
    hif_error_if(row_ind.status() == DATA_UNDEF, "memory allocation failed");
    auto itr = row_ind.begin();

    // assemble nnz arrays
    for (size_type col = 0u; col < m; ++col) {
      const auto q_col   = q[col];
      auto       A_v_itr = A.val_cbegin(q_col);
      for (auto A_itr = A.row_ind_cbegin(q_col), last = A.row_ind_cend(q_col);
           A_itr != last; ++A_itr, ++A_v_itr) {
        const size_type p_inv = p.inv(*A_itr);
        if (p_inv < m) *itr++ = p_inv;
      }
      // sort indices
      std::sort(B.row_ind_begin(col), itr);
    }
  } else {
    // form nonzero pattern of A+A^T

    // create mask
    std::vector<bool> masks(m, false);
    // first determine nnz
    do {
      typename CcsType::iarray_type buf(m);
      hif_error_if(buf.status() == DATA_UNDEF, "memory allocation failed");
      for (size_type i(0); i < m; ++i) {
        size_type nnz(0);
        // first loop through col
        const auto col = q[i];
        for (auto itr = A.row_ind_cbegin(col), last = A.row_ind_cend(col);
             itr != last; ++itr) {
          const size_type ind = p.inv(*itr);
          if (ind < m) {
            masks[ind] = true;
            buf[nnz++] = ind;
          }
        }
        // then loop through row, note that we only count indices are not been
        // counted yet
        const auto row = p[i];
        for (auto itr  = A_crs.col_ind_cbegin(row),
                  last = A_crs.col_ind_cend(row);
             itr != last; ++itr) {
          const size_type ind = q.inv(*itr);
          if (!masks[ind] && ind < m) {
            masks[ind] = true;
            buf[nnz++] = ind;
          }
        }
        col_start[i + 1] = col_start[i] + nnz;
        // reset flags
        std::for_each(buf.cbegin(), buf.cbegin() + nnz,
                      [&](const index_type j) { masks[j] = false; });
      }
    } while (false);  // free buf here

    // then, form indices
    // allocate storage
    auto &row_ind = B.row_ind();
    // NOTE we only indices for reordering step
    row_ind.resize(col_start[m]);
    hif_error_if(row_ind.status() == DATA_UNDEF, "memory allocation failed");

    auto i_itr = row_ind.begin();
    for (size_type i(0); i < m; ++i) {
      const auto col = q[i];
      for (auto itr = A.row_ind_cbegin(col), last = A.row_ind_cend(col);
           itr != last; ++itr) {
        const size_type ind = p.inv(*itr);
        if (ind < m) {
          *i_itr++   = ind;
          masks[ind] = true;
        }
      }
      const auto row = p[i];
      for (auto itr = A_crs.col_ind_cbegin(row), last = A_crs.col_ind_cend(row);
           itr != last; ++itr) {
        const size_type ind = q.inv(*itr);
        if (!masks[ind] && ind < m) {
          masks[ind] = true;
          *i_itr++   = ind;
        }
      }
      // sort indices
      // TODO do we need this in rcm?
      std::sort(B.row_ind_begin(i), i_itr);
      // reset flags
      std::for_each(B.row_ind_cbegin(i), B.row_ind_cend(i),
                    [&](const index_type j) { masks[j] = false; });
    }
  }

  return B;
}

/*!
 * @}
 */ // group pre

}  // namespace internal

/// \brief compute the matching for preprocessing
/// \tparam IsSymm if \a true, then assume a symmetric leading block
/// \tparam CcsType ccs storage for intermidiate matrix after matching
/// \tparam ScalingArray scaling array for row and column, see \ref Array
/// \tparam PermType permutation matrix, see \ref BiPermMatrix
/// \param[in] A input matrix in \ref CCS order
/// \param[in] A_crs the \ref CRS version of \a A
/// \param[in] m0 leading block size
/// \param[in] level current factorization level
/// \param[in] opts control parameters, see \ref Options
/// \param[out] s row scaling vector
/// \param[out] t column scaling vector
/// \param[out] p row permutation vector
/// \param[out] q column permutation vector
/// \return A \a pair of \ref CCS matrix in \b C-index and the actual leading
///         block size, which is no larger than \a m0.
/// \ingroup pre
template <bool IsSymm, class CcsType, class CrsType, class ScalingArray,
          class PermType>
inline std::pair<
    CCS<typename CcsType::value_type, typename CcsType::index_type>,
    typename CcsType::size_type>
do_maching(const CcsType &A, const CrsType &A_crs,
           const typename CcsType::size_type m0,
           const typename CcsType::size_type level, const Options &opts,
           ScalingArray &s, ScalingArray &t, PermType &p, PermType &q) {
  static_assert(!CcsType::ROW_MAJOR, "input must be CCS type");
  static_assert(CrsType::ROW_MAJOR, "input A_crs must be CRS type");
  using value_type  = typename CcsType::value_type;
  using index_type  = typename CcsType::index_type;
  using return_type = CcsType;
  using size_type   = typename CcsType::size_type;

  const size_type M = A.nrows(), N = A.ncols();
  p.resize(M);
  hif_error_if(p().status() == DATA_UNDEF || p.inv().status() == DATA_UNDEF,
               "memory allocation failed for p");
  q.resize(N);
  hif_error_if(q().status() == DATA_UNDEF || q.inv().status() == DATA_UNDEF,
               "memory allocation failed for q");
  s.resize(M);
  hif_error_if(s.status() == DATA_UNDEF, "memory allocation failed for s");
  t.resize(N);
  hif_error_if(s.status() == DATA_UNDEF, "memory allocation failed for t");

  const bool timing       = hif_verbose(PRE_TIME, opts);
  const bool compute_perm = opts.reorder != REORDER_OFF;

  CrsType B;
  if (m0 == M) {
    // NOTE, A_crs is the input, if C index order, then the indices will be
    // temporarily shifted to Fortran order!
    if (!opts.pre_scale)
      B = CrsType(A_crs);  // shallow!
    else {
      B.resize(M, M);
      B.row_start() = A_crs.row_start();
      B.col_ind()   = A_crs.col_ind();
      B.vals().resize(A_crs.nnz());
      hif_error_if(B.vals().status() == DATA_UNDEF, "memory allocation failed");
      std::copy(A_crs.vals().cbegin(), A_crs.vals().cend(), B.vals().begin());
    }
  } else
    B = A_crs.extract_leading(m0);  // for explicit leading block do copy

  using mc64_kernel = EqlDriver<value_type, index_type>;
  do {
    DefaultTimer timer;
    timer.start();
    mc64_kernel::template do_matching<IsSymm>(opts.verbose, B, p(), q(), s, t,
                                              opts.pre_scale);
    timer.finish();
    if (timing) hif_info("Equilibrator took %gs.", (double)timer.time());
  } while (false);
  // fill identity mapping and add one to scaling vectors for offsets, if any
  for (size_type i = m0; i < M; ++i) {
    p[i] = i;
    s[i] = 1;
  }
  for (size_type i = m0; i < N; ++i) {
    q[i] = i;
    t[i] = 1;
  }

  // revert indices
  if (m0 == M) {
    for (auto &v : B.row_start()) --v;
    for (auto &v : B.col_ind()) --v;
  }

  // fix potentially poorly scaled row and column weights
  internal::fix_poor_scaling(m0, level, p, q, s, t, opts.beta);

  // then determine tiny diags
  // using the inverse mappings are buffers since we don't need them for now
  auto &          p_buf = p.inv(), &q_buf = q.inv();
  const size_type m =
      internal::defer_tiny_diags<false>(A, A_crs, m0, p, q, p_buf, q_buf);
  return_type BB;
  if (compute_perm && m) {
    p.build_inv();
    // to see if we need to build A^T+A pattern
    const bool build_apat = !IsSymm && opts.reorder == REORDER_RCM;
    if (build_apat) q.build_inv();
    BB = internal::compute_leading_block(A, A_crs, m, p, q, build_apat);
  }
  return std::make_pair(BB, m);
}

}  // namespace hif

#endif  // _HIF_PRE_MATCHINGSCALING_HPP