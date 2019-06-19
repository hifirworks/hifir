//@HEADER
//----------------------------------------------------------------------------
//                Copyright (C) 2019 The PSMILU AUTHORS
//----------------------------------------------------------------------------
//@HEADER

/// \file psmilu_matching/driver.hpp
/// \brief Matching driver interface
/// \authors Qiao,

#ifndef _PSMILU_MATCHING_DRIVER_HPP
#define _PSMILU_MATCHING_DRIVER_HPP

#include <type_traits>
#include <utility>

#include "psmilu_Array.hpp"
#include "psmilu_CompressedStorage.hpp"
#include "psmilu_Options.h"
#include "psmilu_Timer.hpp"
#include "psmilu_log.hpp"
#include "psmilu_utils.hpp"

#include "psmilu_matching/MUMPS.hpp"

#ifdef PSMILU_ENABLE_MC64
#  include "psmilu_matching/MC64.hpp"
#endif  // PSMILU_ENABLE_MC64

namespace psmilu {
namespace internal {

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
/// \ingroup pre
///
/// For asymmetric cases, where the whole matrix is stored, thus we need binary
/// search to locate the diagonal entries. For symmetric cases, on the other
/// side, since only the \b lower part is stored, we just need to test the first
/// entry of each column.
template <bool IsSymm, class CcsType, class PermType, class BufType>
inline typename CcsType::size_type defer_zero_diags(
    const CcsType &A, const typename CcsType::size_type m0, PermType &p,
    PermType &q, BufType &work_p, BufType &work_q) {
  using value_type                      = typename CcsType::value_type;
  using size_type                       = typename CcsType::size_type;
  constexpr static value_type ZERO      = Const<value_type>::ZERO;
  constexpr static bool       ONE_BASED = CcsType::ONE_BASED;
  const auto                  ori_idx   = [](const size_type i) {
    return to_ori_idx<size_type, ONE_BASED>(i);
  };
  const auto is_valid_entry = [&](const size_type col) -> bool {
    const auto q_col = q[col];
    if (IsSymm) {
      // NOTE that since we only store the lower part, and due to the the
      // fact of symmetric pivoting, if the original is a saddle point, then
      // it is still a saddle entry in the permutated system. This makes
      // checking for invalid diagonal efficient for symmetric case
      psmilu_assert(p[col] == q_col, "fatal");
      auto itr = A.row_ind_cbegin(q_col);
      if (itr == A.row_ind_cend(q_col)) return false;
      if (*itr - ONE_BASED != q_col) return false;
      if (*A.val_cbegin(q_col) == ZERO) return false;
    } else {
      const auto p_diag = p[col];
      auto info = find_sorted(A.row_ind_cbegin(q_col), A.row_ind_cend(q_col),
                              ori_idx(p_diag));
      if (!info.first) return false;
      // test numerical value
      if (*(A.vals().cbegin() + (info.second - A.row_ind().cbegin())) == ZERO)
        return false;
    }
    return true;
  };
  size_type m = m0;

  // for (size_type i = 0u; i < m; ++i)
  //   if (!is_valid_entry(i)) {
  //     for (;;) {
  //       if (i == m) break;
  //       --m;
  //       if (is_valid_entry(m)) break;
  //     }
  //     std::swap(p[i], p[m]);
  //     std::swap(q[i], q[m]);
  //   }
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
/// \tparam IsSymm if \a true, then assume a symmetric leading block
/// \tparam ReturnCcsType ccs storage that must be in C index
/// \tparam CcsType ccs storage for intermidiate matrix after matching
/// \tparam PermType permutation matrix, see \ref BiPermMatrix
/// \param[in] A input matrix after performing matching
/// \param[in] m leading block size that must no larger than size(A)
/// \param[in] p row permutation matrix
/// \param[in] q column permutation matrix
/// \ingroup pre
/// \note It's worth noting that \a q must be bi-directional mapping
template <bool IsSymm, class ReturnCcsType, class CcsType, class CrsType,
          class PermType>
inline typename std::enable_if<!IsSymm, ReturnCcsType>::type
compute_perm_leading_block(const CcsType &                   A, const CrsType &,
                           const typename CcsType::size_type m,
                           const PermType &p, const PermType &q) {
  using size_type                 = typename CcsType::size_type;
  constexpr static bool ONE_BASED = CcsType::ONE_BASED;
  static_assert(!(ONE_BASED ^ CrsType::ONE_BASED), "inconsistent ONE_BASED");
  const auto c_idx = [](const size_type i) {
    return to_c_idx<size_type, ONE_BASED>(i);
  };

  ReturnCcsType B(m, m);
  auto &        col_start = B.col_start();
  col_start.resize(m + 1);
  psmilu_error_if(col_start.status() == DATA_UNDEF, "memory allocation failed");
  col_start.front() = 0;  // zero based

  // determine nnz
  for (size_type col = 0u; col < m; ++col) {
    const auto q_col   = q[col];
    col_start[col + 1] = std::count_if(
        A.row_ind_cbegin(q_col), A.row_ind_cend(q_col), [&](decltype(q_col) i) {
          return static_cast<size_type>(p.inv(c_idx(i))) < m;
        });
  }
  for (size_type i = 0u; i < m; ++i) col_start[i + 1] += col_start[i];

  // allocate storage
  // B.reserve(col_start[m]);
  auto &row_ind = B.row_ind();
  // NOTE we only indices for reordering step
  row_ind.resize(col_start[m]);
  psmilu_error_if(row_ind.status() == DATA_UNDEF, "memory allocation failed");
  auto itr = row_ind.begin();

  // assemble nnz arrays
  for (size_type col = 0u; col < m; ++col) {
    const auto q_col   = q[col];
    auto       A_v_itr = A.val_cbegin(q_col);
    for (auto A_itr = A.row_ind_cbegin(q_col), last = A.row_ind_cend(q_col);
         A_itr != last; ++A_itr, ++A_v_itr) {
      const size_type p_inv = p.inv(c_idx(*A_itr));
      if (p_inv < m) {
        *itr++ = p_inv;
      }
    }
    // sort indices
    std::sort(B.row_ind_begin(col), itr);
  }

  return B;
}

/// \brief permute the matrix for symmetric cases
/// \tparam IsSymm if \a true, then assume a symmetric leading block
/// \tparam ReturnCcsType ccs storage that must be in C index
/// \tparam CcsType ccs storage for intermidiate matrix after matching
/// \tparam PermType permutation matrix, see \ref BiPermMatrix
/// \param[in] A input matrix after performing matching
/// \param[in] m leading block size that must no larger than size(A)
/// \param[in] p row permutation matrix
/// \param[in] q column permutation matrix
/// \ingroup pre
/// \note It's worth noting that \a q must be bi-directional mapping
///
/// Notice that, for symmetric cases, since only the lower part is stored, we
/// need to build the \ref CRS version of \a A, which is nothing but the CCS
/// version of upper part of \a A.
template <bool IsSymm, class ReturnCcsType, class CcsType, class CrsType,
          class PermType>
inline typename std::enable_if<IsSymm, ReturnCcsType>::type
compute_perm_leading_block(const CcsType &A, const CrsType &A_crs,
                           const typename CcsType::size_type m,
                           const PermType &p, const PermType &q) {
  using size_type                 = typename CcsType::size_type;
  constexpr static bool ONE_BASED = CcsType::ONE_BASED;
  static_assert(!(ONE_BASED ^ CrsType::ONE_BASED), "inconsistent ONE_BASED");
  const auto c_idx = [](const size_type i) {
    return to_c_idx<size_type, ONE_BASED>(i);
  };
  // for symmetric case, we need first build a CRS of the lower part, which is
  // the "CCS" of the upper part
  using crs_type = typename CcsType::other_type;

  // const crs_type A_crs = crs_type(A);

  ReturnCcsType B(m, m);
  auto &        col_start = B.col_start();
  col_start.resize(m + 1);
  psmilu_error_if(col_start.status() == DATA_UNDEF, "memory allocation failed");
  col_start.front() = 0;  // zero based

  // buffer for value array
  // Array<typename CcsType::value_type> buf(m);
  // psmilu_error_if(buf.status() == DATA_UNDEF, "memory allocation failed");

  // determine nnz
  for (size_type col = 0u; col < m; ++col) {
    const auto q_col = q[col];
    col_start[col + 1] =
        std::count_if(A.row_ind_cbegin(q_col), A.row_ind_cend(q_col),
                      [&, m](decltype(q_col) i) {
                        const size_type p_idx = p.inv(c_idx(i));
                        return p_idx >= col && p_idx < m;
                      });
    if (A_crs.nnz_in_row(q_col)) {
      // for the upper part
      col_start[col + 1] += std::count_if(
          A_crs.col_ind_cbegin(q_col), A_crs.col_ind_cend(q_col) - 1,
          [&, m](decltype(q_col) i) {
            const size_type p_idx = p.inv(c_idx(i));
            return p_idx >= col && p_idx < m;
          });
    }
  }
  for (size_type i = 0u; i < m; ++i) col_start[i + 1] += col_start[i];

  // allocate storage
  // B.reserve(col_start[m]);
  // NOTE we only need indices for next step, i.e. reordering
  auto &row_ind = B.row_ind();
  row_ind.resize(col_start[m]);
  psmilu_error_if(row_ind.status() == DATA_UNDEF, "memory allocation failed");
  auto itr = row_ind.begin();

  // assemble nnz arrays
  for (size_type col = 0u; col < m; ++col) {
    const auto q_col   = q[col];
    auto       A_v_itr = A.val_cbegin(q_col);
    for (auto A_itr = A.row_ind_cbegin(q_col), last = A.row_ind_cend(q_col);
         A_itr != last; ++A_itr, ++A_v_itr) {
      const size_type p_inv = p.inv(c_idx(*A_itr));
      if (p_inv >= col && p_inv < m) {
        *itr++ = p_inv;
      }
    }
    if (A_crs.nnz_in_row(q_col)) {
      auto A_v_itr = A_crs.val_cbegin(q_col);
      for (auto A_itr = A_crs.col_ind_cbegin(q_col),
                last  = A_crs.col_ind_cend(q_col) - 1;
           A_itr != last; ++A_itr, ++A_v_itr) {
        const size_type p_inv = p.inv(c_idx(*A_itr));
        if (p_inv >= col && p_inv < m) {
          *itr++ = p_inv;
        }
      }
    }
    // sort indices
    std::sort(B.row_ind_begin(col), itr);
  }

  return B;
}

/// \brief extract leading diagonal entries for MC64 routine
/// \tparam IsSymm if \a true, then assume a symmetric leading block
/// \tparam ValueType value type for input, e.g. \a double
/// \tparam IndexType index type for input, e.g. \a int
/// \tparam OneBased index flag
/// \param[in] A input matrix
/// \param[in] m leading block size to extract
/// \ingroup pre
///
/// Notice that for efficiency purpose, we return CCS matrices. Also,
/// be aware that the input only accept \ref CCS matrix, which is fine, because
/// by the time calling this function, we should have both \ref CRS and
/// CCS versions of the input matrix.
///
/// \note This function only allocate memory needed for the leading block. No
///       additional heap memory is needed.
template <bool IsSymm, class ValueType, class IndexType, bool OneBased>
inline CCS<ValueType, IndexType, OneBased> extract_leading_block4matching(
    const CCS<ValueType, IndexType, OneBased> &                   A,
    const typename CCS<ValueType, IndexType, OneBased>::size_type m) {
  using value_type                = ValueType;
  using index_type                = IndexType;
  using return_ccs                = CCS<value_type, index_type, OneBased>;
  using size_type                 = typename return_ccs::size_type;
  constexpr static bool ONE_BASED = OneBased;
  const auto            ori_idx   = [](const size_type i) {
    return to_ori_idx<size_type, ONE_BASED>(i);
  };

  const size_type M = A.nrows(), N = A.ncols();
  psmilu_error_if(
      m > M || m > N,
      "leading block size should not be larger than the matrix sizes");

  // shallow copy if leading block is the same as input and not symmetric
  if (!IsSymm && m == M) return A;

  return_ccs B(m, m);

  // NOTE use col_start to first store the position in A
  auto A_itr = A.row_ind().cbegin();
  B.col_start().resize(m + 1);
  auto &col_start = B.col_start();
  psmilu_error_if(col_start.status() == DATA_UNDEF, "memory allocation failed");
  auto &          row_ind = B.row_ind();
  auto &          vals    = B.vals();
  const size_type tgt     = ori_idx(m);

  if (IsSymm) {
    // for symmetric case, we use col_start to store first position
    // note that two binary searches are needed for each column to determine
    // the nnz. Then, while filling values, only one binary is needed.

    auto      A_v_itr1 = A.vals().cbegin();
    size_type nnz(0u);
    for (size_type i = 0u; i < m; ++i) {
      auto info1 = find_sorted(A.row_ind_cbegin(i), A.row_ind_cend(i), tgt);
      auto info2 = find_sorted(A.row_ind_cbegin(i), info1.second, ori_idx(i));
      // only lower part
      nnz += info1.second - info2.second;
      col_start[i + 1] = info2.second - A_itr;
    }

    // memory allocation
    row_ind.resize(nnz);
    psmilu_error_if(row_ind.status() == DATA_UNDEF, "memory allocation failed");
    vals.resize(nnz);
    psmilu_error_if(vals.status() == DATA_UNDEF, "memory allocation failed");

    auto itr   = row_ind.begin();
    auto v_itr = vals.begin();

    col_start[0] = ONE_BASED;
    for (size_type i = 0u; i < m; ++i) {
      auto a_itr   = A_itr + col_start[i + 1],
           last    = find_sorted(a_itr, A.row_ind_cend(i), tgt).second;
      auto A_v_itr = A_v_itr1 + col_start[i + 1];
      // NOTE it's safe to modify col_start now
      col_start[i + 1] = col_start[i] + (last - a_itr);
      for (; a_itr != last; ++a_itr, ++itr, ++v_itr, ++A_v_itr) {
        *itr   = *a_itr;
        *v_itr = *A_v_itr;
      }
    }
  } else {
    // for asymmetric case, we use col_start to store the pass-of-end positions
    // it's worth nothing that one a single binary search is needed for
    // determining the nnz.

    size_type nnz(0u);
    for (size_type i = 0u; i < m; ++i) {
      auto info = find_sorted(A.row_ind_cbegin(i), A.row_ind_cend(i), tgt);
      nnz += info.second - A.row_ind_cbegin(i);
      col_start[i + 1] = info.second - A_itr;
    }

    // memory allocation
    row_ind.resize(nnz);
    psmilu_error_if(row_ind.status() == DATA_UNDEF, "memory allocation failed");
    vals.resize(nnz);
    psmilu_error_if(vals.status() == DATA_UNDEF, "memory allocation failed");

    auto itr   = row_ind.begin();
    auto v_itr = vals.begin();

    col_start[0] = ONE_BASED;
    for (size_type i = 0u; i < m; ++i) {
      auto a_itr   = A.row_ind_cbegin(i);
      auto last    = A_itr + col_start[i + 1];
      auto A_v_itr = A.val_cbegin(i);
      // safe to modify col_start
      col_start[i + 1] = col_start[i] + (last - a_itr);
      for (; a_itr != last; ++a_itr, ++itr, ++v_itr, ++A_v_itr) {
        *itr   = *a_itr;
        *v_itr = *A_v_itr;
      }
    }
  }

  return B;
}
}  // namespace internal

/// \brief compute the matching for preprocessing
/// \tparam IsSymm if \a true, then assume a symmetric leading block
/// \tparam CcsType ccs storage for intermidiate matrix after matching
/// \tparam ScalingArray scaling array for row and column, see \ref Array
/// \tparam PermType permutation matrix, see \ref BiPermMatrix
/// \param[in] A input matrix in \ref CCS order
/// \param[in] m0 leading block size
/// \param[in] verbose message verbose flag from \ref Options
/// \param[out] s row scaling vector
/// \param[out] t column scaling vector
/// \param[out] p row permutation vector
/// \param[out] q column permutation vector
/// \param[in] hdl_zero_diags if \a false (default), the routine won't handle
///            zero diagonal entries.
/// \param[in] compute_perm if \a true (default), will perform explicit
///            computation for permutation matrix for further processing
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
  using value_type                = typename CcsType::value_type;
  using index_type                = typename CcsType::index_type;
  constexpr static bool ONE_BASED = CcsType::ONE_BASED;
  using return_type               = CCS<value_type, index_type>;
  using size_type                 = typename CcsType::size_type;
  constexpr static value_type ONE = Const<value_type>::ONE;
  using mumps_kernel              = MUMPS<value_type, index_type, ONE_BASED>;

  const size_type M = A.nrows(), N = A.ncols();
  p.resize(M);
  psmilu_error_if(p().status() == DATA_UNDEF || p.inv().status() == DATA_UNDEF,
                  "memory allocation failed for p");
  q.resize(N);
  psmilu_error_if(q().status() == DATA_UNDEF || q.inv().status() == DATA_UNDEF,
                  "memory allocation failed for q");
  s.resize(M);
  psmilu_error_if(s.status() == DATA_UNDEF, "memory allocation failed for s");
  t.resize(N);
  psmilu_error_if(s.status() == DATA_UNDEF, "memory allocation failed for t");

  const int  matching     = opts.matching;
  const bool timing       = psmilu_verbose(PRE_TIME, opts);
  const bool compute_perm = opts.reorder != REORDER_OFF;

  CrsType B;
  if (m0 == M)
    B = CrsType(A_crs, true);
  else
    B = A_crs.extract_leading(m0);

#ifndef PSMILU_ENABLE_MC64
  if (matching == MATCHING_MC64)
    psmilu_warning("MC64 is not available, skip to use MUMPS");
  do {
    DefaultTimer timer;
    timer.start();
    mumps_kernel::template do_matching<IsSymm>(verbose, B, p(), q(), s, t,
                                               opts.iter_pre_scale);
    timer.finish();
    if (timing) psmilu_info("MUMPS matching took %gs.", (double)timer.time());
  } while (false);
#else
  using mc64_kernel      = MC64<value_type, index_type, ONE_BASED>;
  std::string match_name = "MUMPS";
  do {
    DefaultTimer timer;
    timer.start();
    if (matching != MATCHING_MUMPS) {
      match_name = "MC64";
      mc64_kernel::template do_matching<IsSymm>(verbose, B, p(), q(), s, t,
                                                opts.iter_pre_scale);
    } else
      mumps_kernel::template do_matching<IsSymm>(verbose, B, p(), q(), s, t,
                                                 opts.iter_pre_scale);
    timer.finish();
    if (timing)
      psmilu_info("%s matching took %gs.", match_name.c_str(),
                  (double)timer.time());
  } while (false);
#endif  // PSMILU_ENABLE_MC64
  // fill identity mapping and add one to scaling vectors for offsets, if any
  for (size_type i = m0; i < M; ++i) {
    p[i] = i;
    s[i] = ONE;
  }
  for (size_type i = m0; i < N; ++i) {
    q[i] = i;
    t[i] = ONE;
  }

  // then determine zero diags
  // using the inverse mappings are buffers since we don't need them for now
  const size_type m = !hdl_zero_diags ? m0
                                      : internal::defer_zero_diags<false>(
                                            A, m0, p, q, p.inv(), q.inv());
  return_type BB;
  if (compute_perm) {
    p.build_inv();
    BB = internal::compute_perm_leading_block<false, return_type>(A, A_crs, m,
                                                                  p, q);
  }
  return std::make_pair(BB, m);
}
}  // namespace psmilu

#endif  // _PSMILU_MATCHING_DRIVER_HPP
