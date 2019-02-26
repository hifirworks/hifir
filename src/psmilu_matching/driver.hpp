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

#ifndef mc64_matching
extern "C" {
#  include "hsl_mc64d.h"
}
#endif

#include <type_traits>
#include <utility>

#include "psmilu_Array.hpp"
#include "psmilu_CompressedStorage.hpp"
#include "psmilu_log.hpp"
#include "psmilu_utils.hpp"

#ifdef mc64_matching

#  include "HSL_MC64.hpp"

#else
#  error "PSMILU requires HSL_MC64, for now..."
#endif

namespace psmilu {
namespace internal {

/// \brief defer any zero diags to the end
/// \tparam IsSymm if \a true, then assume a symmetric leading block
/// \tparam CcsType ccs storage for intermidiate matrix after matching
/// \tparam PermType permutation matrix, see \ref BiPermMatrix
/// \param[in] A input matrix after calling matching
/// \param[in] m0 initial leading block size
/// \param[in,out] p row permutation matrix
/// \param[in,out] q column permutation matrix
/// \return actual leading block with no zero entries, <= \a m0
/// \ingroup pre
///
/// For asymmetric cases, where the whole matrix is stored, thus we need binary
/// search to locate the diagonal entries. For symmetric cases, on the other
/// side, since only the \b lower part is stored, we just need to test the first
/// entry of each column.
template <bool IsSymm, class CcsType, class PermType>
inline typename CcsType::size_type defer_zero_diags(
    const CcsType &A, const typename CcsType::size_type m0, PermType &p,
    PermType &q) {
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
      psmilu_assert(p[col] == q_col, "fatal");
      auto itr = A.row_ind_cbegin(q_col);
      if (itr == A.row_ind_cend(q_col)) return false;
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

  for (size_type i = 0u; i < m; ++i)
    if (!is_valid_entry(i)) {
      for (;;) {
        if (i == m) break;
        --m;
        if (is_valid_entry(m)) break;
      }
      std::swap(p[i], p[m]);
      std::swap(q[i], q[m]);
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
template <bool IsSymm, class ReturnCcsType, class CcsType, class PermType>
inline typename std::enable_if<!IsSymm, ReturnCcsType>::type
compute_perm_leading_block(const CcsType &                   A,
                           const typename CcsType::size_type m,
                           const PermType &p, const PermType &q) {
  using size_type                 = typename CcsType::size_type;
  constexpr static bool ONE_BASED = CcsType::ONE_BASED;
  const auto            c_idx     = [](const size_type i) {
    return to_c_idx<size_type, ONE_BASED>(i);
  };

  ReturnCcsType B(m, m);
  auto &        col_start = B.col_start();
  col_start.resize(m + 1);
  psmilu_error_if(col_start.status() == DATA_UNDEF, "memory allocation failed");
  col_start.front() = 0;  // zero based

  // buffer for value array
  Array<typename CcsType::value_type> buf(m);
  psmilu_error_if(buf.status() == DATA_UNDEF, "memory allocation failed");

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
  B.reserve(col_start[m]);
  auto &row_ind = B.row_ind();
  psmilu_error_if(row_ind.status() == DATA_UNDEF, "memory allocation failed");
  auto &vals = B.vals();
  psmilu_error_if(vals.status() == DATA_UNDEF, "memory allocation failed");
  row_ind.resize(col_start[m]);
  auto itr = row_ind.begin();
  vals.resize(col_start[m]);
  auto v_itr = vals.begin();

  // assemble nnz arrays
  for (size_type col = 0u; col < m; ++col) {
    const auto q_col   = q[col];
    auto       A_v_itr = A.val_cbegin(q_col);
    for (auto A_itr = A.row_ind_cbegin(q_col), last = A.row_ind_cend(q_col);
         A_itr != last; ++A_itr, ++A_v_itr) {
      const size_type p_inv = p.inv(c_idx(*A_itr));
      if (p_inv < m) {
        *itr++     = p_inv;
        buf[p_inv] = *A_v_itr;
      }
    }
    // sort indices
    std::sort(B.row_ind_begin(col), itr);
    for (auto i = B.row_ind_cbegin(col), last = B.row_ind_cend(col); i != last;
         ++i, ++v_itr)
      *v_itr = buf[*i];
  }

  psmilu_assert(v_itr == vals.end(), "fatal");

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
template <bool IsSymm, class ReturnCcsType, class CcsType, class PermType>
inline typename std::enable_if<IsSymm, ReturnCcsType>::type
compute_perm_leading_block(const CcsType &                   A,
                           const typename CcsType::size_type m,
                           const PermType &p, const PermType &q) {
  using size_type                 = typename CcsType::size_type;
  constexpr static bool ONE_BASED = CcsType::ONE_BASED;
  const auto            c_idx     = [](const size_type i) {
    return to_c_idx<size_type, ONE_BASED>(i);
  };
  // for symmetric case, we need first build a CRS of the lower part, which is
  // the "CCS" of the upper part
  using crs_type = typename CcsType::other_type;

  const crs_type A_crs = crs_type(A);

  ReturnCcsType B(m, m);
  auto &        col_start = B.col_start();
  col_start.resize(m + 1);
  psmilu_error_if(col_start.status() == DATA_UNDEF, "memory allocation failed");
  col_start.front() = 0;  // zero based

  // buffer for value array
  Array<typename CcsType::value_type> buf(m);
  psmilu_error_if(buf.status() == DATA_UNDEF, "memory allocation failed");

  // determine nnz
  for (size_type col = 0u; col < m; ++col) {
    const auto q_col   = q[col];
    col_start[col + 1] = std::count_if(
        A.row_ind_cbegin(q_col), A.row_ind_cend(q_col), [&](decltype(q_col) i) {
          return static_cast<size_type>(p.inv(c_idx(i))) >= col;
        });
    if (A_crs.nnz_in_row(q_col)) {
      // for the upper part
      col_start[col + 1] +=
          std::count_if(A_crs.col_ind_cbegin(q_col),
                        A_crs.col_ind_cend(q_col) - 1, [&](decltype(q_col) i) {
                          return static_cast<size_type>(p.inv(c_idx(i))) >= col;
                        });
    }
  }
  for (size_type i = 0u; i < m; ++i) col_start[i + 1] += col_start[i];

  // allocate storage
  B.reserve(col_start[m]);
  auto &row_ind = B.row_ind();
  psmilu_error_if(row_ind.status() == DATA_UNDEF, "memory allocation failed");
  auto &vals = B.vals();
  psmilu_error_if(vals.status() == DATA_UNDEF, "memory allocation failed");
  row_ind.resize(col_start[m]);
  auto itr = row_ind.begin();
  vals.resize(col_start[m]);
  auto v_itr = vals.begin();

  // assemble nnz arrays
  for (size_type col = 0u; col < m; ++col) {
    const auto q_col   = q[col];
    auto       A_v_itr = A.val_cbegin(q_col);
    for (auto A_itr = A.row_ind_cbegin(q_col), last = A.row_ind_cend(q_col);
         A_itr != last; ++A_itr, ++A_v_itr) {
      const size_type p_inv = p.inv(c_idx(*A_itr));
      if (p_inv >= col) {
        *itr++     = p_inv;
        buf[p_inv] = *A_v_itr;
      }
    }
    if (A_crs.nnz_in_row(q_col)) {
      auto A_v_itr = A_crs.val_cbegin(q_col);
      for (auto A_itr = A_crs.col_ind_cbegin(q_col),
                last  = A_crs.col_ind_cend(q_col) - 1;
           A_itr != last; ++A_itr, ++A_v_itr) {
        const size_type p_inv = p.inv(c_idx(*A_itr));
        if (p_inv >= col) {
          *itr++     = p_inv;
          buf[p_inv] = *A_v_itr;
        }
      }
    }
    // sort indices
    std::sort(B.row_ind_begin(col), itr);
    for (auto i = B.row_ind_cbegin(col), last = B.row_ind_cend(col); i != last;
         ++i, ++v_itr)
      *v_itr = buf[*i];
  }

  psmilu_assert(v_itr == vals.end(), "fatal");

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
/// \return A \a pair of \ref CCS matrix in \b C-index and the actual leading
///         block size, which is no larger than \a m0.
/// \ingroup pre
template <bool IsSymm, class CcsType, class ScalingArray, class PermType>
inline std::pair<
    CCS<typename CcsType::value_type, typename CcsType::index_type>,
    typename CcsType::size_type>
do_maching(const CcsType &A, const typename CcsType::size_type m0,
           const int verbose, ScalingArray &s, ScalingArray &t, PermType &p,
           PermType &q, const bool hdl_zero_diags = false) {
  static_assert(!CcsType::ROW_MAJOR, "input must be CCS type");
  using value_type                = typename CcsType::value_type;
  using index_type                = typename CcsType::index_type;
  constexpr static bool ONE_BASED = CcsType::ONE_BASED;
  using match_driver = MatchingDriver<value_type, index_type, ONE_BASED>;
  using input_type   = typename match_driver::input_type;
  constexpr static bool INPUT_ONE_BASED = input_type::ONE_BASED;
  using return_type                     = CCS<value_type, index_type>;
  using size_type                       = typename CcsType::size_type;
  constexpr static value_type ONE       = Const<value_type>::ONE;

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
  matching_control_type control;  // create control parameters for matching
  return_type           B;
  set_default_control(verbose, control, ONE_BASED);
  // first extract matching
  input_type B1 = internal::extract_leading_block4matching<IsSymm>(A, m0);
  // then compute matching
  match_driver::template do_matching<IsSymm>(B1, control, p(), q(), s, t);
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
  const size_type m =
      !hdl_zero_diags ? m0 : internal::defer_zero_diags<IsSymm>(B1, m0, p, q);
  const bool is_eye_perm = p.is_eye() && q.is_eye();
  if (!is_eye_perm) {
    // we need inverse mapping for row permutation
    p.build_inv();
    B = internal::compute_perm_leading_block<IsSymm, return_type>(B1, m, p, q);
  } else {
    psmilu_assert(m0 == m,
                  "if no permutation occurred from matching, then the leading "
                  "sizes, at least, should match!!??");
    if (!INPUT_ONE_BASED)
      B = return_type(B1);  // shallow
    else {
      // if the input is Fortran index system
      B.resize(B1.nrows(), B1.ncols());
      auto &col_start = B.col_start();
      col_start.resize(B1.ncols() + 1);
      psmilu_error_if(col_start.status() == DATA_UNDEF,
                      "memory callocation failed");
      std::transform(B1.col_start().cbegin(), B1.col_start().cend(),
                     col_start.begin(),
                     [](const index_type i) { return i - 1; });
      auto &row_ind = B.row_ind();
      row_ind.resize(B1.nnz());
      psmilu_error_if(row_ind.status() == DATA_UNDEF,
                      "memory allocation failed");
      std::transform(B1.row_ind().cbegin(), B1.row_ind().cend(),
                     row_ind.begin(), [](const index_type i) { return i - 1; });
      // shallow copy
      B.vals() = B1.vals();
    }
  }
  return std::make_pair(B, m);
}
}  // namespace psmilu

#endif  // _PSMILU_MATCHING_DRIVER_HPP
