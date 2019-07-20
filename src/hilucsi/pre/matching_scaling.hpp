//@HEADER
//----------------------------------------------------------------------------
//                Copyright (C) 2019 The HILUCSI AUTHORS
//----------------------------------------------------------------------------
//@HEADER

/// \file hilucsi/pre/matching_scaling.hpp
/// \brief calling interface for matching and scaling
/// \authors Qiao,

#ifndef _HILUCSI_PRE_MATCHINGSCALING_HPP
#define _HILUCSI_PRE_MATCHINGSCALING_HPP

#include "hilucsi/Options.h"
#include "hilucsi/ds/Array.hpp"
#include "hilucsi/ds/CompressedStorage.hpp"
#include "hilucsi/pre/MC64.hpp"
#include "hilucsi/utils/Timer.hpp"
#include "hilucsi/utils/common.hpp"

namespace hilucsi {
namespace internal {

/*!
 * \addtogroup pre
 * @{
 */

/// \brief defer any zero diags to the end
/// \tparam IsSymm if \a true, then assume a symmetric leading block
/// \tparam CcsType ccs storage for intermidiate matrix after matching
/// \tparam PermType permutation matrix, see \ref BiPermMatrix
/// \tparam BufType buffer used to stored deferred entries
/// \param[in] A input matrix after calling matching
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
template <bool IsSymm, class CcsType, class PermType, class BufType>
inline typename CcsType::size_type defer_zero_diags(
    const CcsType &A, const typename CcsType::size_type m0, PermType &p,
    PermType &q, BufType &work_p, BufType &work_q) {
  using value_type                 = typename CcsType::value_type;
  using size_type                  = typename CcsType::size_type;
  constexpr static value_type ZERO = Const<value_type>::ZERO;

  // kernel
  const auto is_valid_entry = [&](const size_type col) -> bool {
    const auto q_col = q[col];
    if (IsSymm) {
      // NOTE that since we only store the lower part, and due to the the
      // fact of symmetric pivoting, if the original is a saddle point, then
      // it is still a saddle entry in the permutated system. This makes
      // checking for invalid diagonal efficient for symmetric case
      hilucsi_assert(p[col] == q_col, "fatal");
      auto itr = A.row_ind_cbegin(q_col);
      if (itr == A.row_ind_cend(q_col)) return false;
      if (*itr != q_col) return false;
      if (*A.val_cbegin(q_col) == ZERO) return false;
    } else {
      const auto p_diag = p[col];
      auto       info =
          find_sorted(A.row_ind_cbegin(q_col), A.row_ind_cend(q_col), p_diag);
      if (!info.first) return false;
      // test numerical value
      if (*(A.vals().cbegin() + (info.second - A.row_ind().cbegin())) == ZERO)
        return false;
    }
    return true;
  };
  size_type m = m0;

  size_type deferrals(0);
  for (size_type i(0); i < m; ++i) {
    while (!is_valid_entry(i + deferrals)) {
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
/// \tparam PermType permutation matrix, see \ref BiPermMatrix
/// \param[in] A input matrix after performing matching
/// \param[in] m leading block size that must no larger than size(A)
/// \param[in] p row permutation matrix
/// \param[in] q column permutation matrix
/// \note It's worth noting that \a q must be bi-directional mapping
template <class CcsType, class CrsType, class PermType>
inline CcsType compute_perm_leading_block(const CcsType &A, const CrsType &,
                                          const typename CcsType::size_type m,
                                          const PermType &                  p,
                                          const PermType &                  q) {
  using size_type = typename CcsType::size_type;

  CcsType B(m, m);
  auto &  col_start = B.col_start();
  col_start.resize(m + 1);
  hilucsi_error_if(col_start.status() == DATA_UNDEF,
                   "memory allocation failed");
  col_start.front() = 0;  // zero based

  // determine nnz
  for (size_type col = 0u; col < m; ++col) {
    const auto q_col   = q[col];
    col_start[col + 1] = std::count_if(
        A.row_ind_cbegin(q_col), A.row_ind_cend(q_col), [&](decltype(q_col) i) {
          return static_cast<size_type>(p.inv(i)) < m;
        });
  }
  for (size_type i = 0u; i < m; ++i) col_start[i + 1] += col_start[i];

  // allocate storage
  // B.reserve(col_start[m]);
  auto &row_ind = B.row_ind();
  // NOTE we only indices for reordering step
  row_ind.resize(col_start[m]);
  hilucsi_error_if(row_ind.status() == DATA_UNDEF, "memory allocation failed");
  auto itr = row_ind.begin();

  // assemble nnz arrays
  for (size_type col = 0u; col < m; ++col) {
    const auto q_col   = q[col];
    auto       A_v_itr = A.val_cbegin(q_col);
    for (auto A_itr = A.row_ind_cbegin(q_col), last = A.row_ind_cend(q_col);
         A_itr != last; ++A_itr, ++A_v_itr) {
      const size_type p_inv = p.inv(*A_itr);
      if (p_inv < m) {
        *itr++ = p_inv;
      }
    }
    // sort indices
    std::sort(B.row_ind_begin(col), itr);
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
/// \param[in] verbose message verbose flag from \ref Options
/// \param[out] s row scaling vector
/// \param[out] t column scaling vector
/// \param[out] p row permutation vector
/// \param[out] q column permutation vector
/// \param[in] opts control parameters
/// \param[in] hdl_zero_diags if \a false (default), the routine won't handle
///            zero diagonal entries.
/// \return A \a pair of \ref CCS matrix in \b C-index and the actual leading
///         block size, which is no larger than \a m0.
/// \ingroup pre
template <bool IsSymm, class CcsType, class CrsType, class ScalingArray,
          class PermType>
inline std::pair<
    CCS<typename CcsType::value_type, typename CcsType::index_type>,
    typename CcsType::size_type>
do_maching(const CcsType &A, const CrsType &A_crs,
           const typename CcsType::size_type m0, const int verbose,
           ScalingArray &s, ScalingArray &t, PermType &p, PermType &q,
           const Options &opts, const bool hdl_zero_diags = false) {
  static_assert(!CcsType::ROW_MAJOR, "input must be CCS type");
  static_assert(CrsType::ROW_MAJOR, "input A_crs must be CRS type");
  using value_type                = typename CcsType::value_type;
  using index_type                = typename CcsType::index_type;
  using return_type               = CcsType;
  using size_type                 = typename CcsType::size_type;
  constexpr static value_type ONE = Const<value_type>::ONE;

  const size_type M = A.nrows(), N = A.ncols();
  p.resize(M);
  hilucsi_error_if(p().status() == DATA_UNDEF || p.inv().status() == DATA_UNDEF,
                   "memory allocation failed for p");
  q.resize(N);
  hilucsi_error_if(q().status() == DATA_UNDEF || q.inv().status() == DATA_UNDEF,
                   "memory allocation failed for q");
  s.resize(M);
  hilucsi_error_if(s.status() == DATA_UNDEF, "memory allocation failed for s");
  t.resize(N);
  hilucsi_error_if(s.status() == DATA_UNDEF, "memory allocation failed for t");

  const bool timing       = hilucsi_verbose(PRE_TIME, opts);
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
      hilucsi_error_if(B.vals().status() == DATA_UNDEF,
                       "memory allocation failed");
      std::copy(A_crs.vals().cbegin(), A_crs.vals().cend(), B.vals().begin());
    }
  } else
    B = A_crs.extract_leading(m0);  // for explicit leading block do copy

  using mc64_kernel = MC64<value_type, index_type>;
  do {
    DefaultTimer timer;
    timer.start();
    mc64_kernel::template do_matching<IsSymm>(verbose, B, p(), q(), s, t,
                                              opts.pre_scale);
    timer.finish();
    if (timing) hilucsi_info("MC64 matching took %gs.", (double)timer.time());
  } while (false);
  // fill identity mapping and add one to scaling vectors for offsets, if any
  for (size_type i = m0; i < M; ++i) {
    p[i] = i;
    s[i] = ONE;
  }
  for (size_type i = m0; i < N; ++i) {
    q[i] = i;
    t[i] = ONE;
  }

  // revert indices
  if (m0 == M) {
    for (auto &v : B.row_start()) --v;
    for (auto &v : B.col_ind()) --v;
  }

  // then determine zero diags
  // using the inverse mappings are buffers since we don't need them for now
  const size_type m = !hdl_zero_diags ? m0
                                      : internal::defer_zero_diags<false>(
                                            A, m0, p, q, p.inv(), q.inv());
  return_type BB;
  if (compute_perm) {
    p.build_inv();
    BB = internal::compute_perm_leading_block(A, A_crs, m, p, q);
  }
  return std::make_pair(BB, m);
}

}  // namespace hilucsi

#endif  // _HILUCSI_PRE_MATCHINGSCALING_HPP