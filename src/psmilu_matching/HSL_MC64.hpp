//@HEADER
//----------------------------------------------------------------------------
//                Copyright (C) 2019 The PSMILU AUTHORS
//----------------------------------------------------------------------------
//@HEADER

/// \file psmilu_matching/HSL_MC64.hpp
/// \brief MC64 Wrapper
/// \authors Qiao,
/// \note Only available when the user includes hsl_mc64?.h beforehand

#ifndef _PSMILU_MATCHING_HSL_MC64_HPP
#define _PSMILU_MATCHING_HSL_MC64_HPP

#ifndef PSMILU_UNIT_TESTING

// dev purpose
#  if 1
#    include "hsl_mc64d.h"
#  endif

#  ifndef mc64_matching
#    error "This header should be included only after HSL_MC64 C interface!"
#  endif

#endif  // PSMILU_UNIT_TESTING

#include <algorithm>
#include <cmath>
#include <limits>
#include <new>
#include <type_traits>

#include "psmilu_Array.hpp"
#include "psmilu_CompressedStorage.hpp"
#include "psmilu_log.hpp"
#include "psmilu_utils.hpp"

namespace psmilu {

#ifndef PSMILU_UNIT_TESTING

/*!
 * \addtogroup pre
 * @{
 */

/// \class HSL_MC64
/// \brief interface for calling MC64
/// \tparam ValueType value type used, e.g. \a double
/// \tparam IndexType index type used, e.g. \a int
/// \tparam OneBased (ignored) always set to \a true
/// \warning If \a IndexType is not \a int, then bad things may happen!
template <class ValueType, class IndexType, bool OneBased>
class HSL_MC64 {
 public:
  typedef ValueType                             value_type;  ///< value type
  typedef IndexType                             index_type;  ///< index type
  typedef CCS<value_type, index_type, OneBased> ccs_type;    ///< input matrix
  typedef typename ccs_type::size_type          size_type;   ///< size

  constexpr static bool ONE_BASED = OneBased;  ///< index flag

#  ifdef HSL_MC64D_H
  static_assert(std::is_same<value_type, double>::value, "must be double!");
#  elif defined(HSL_MC64S_H)
  static_assert(std::is_same<value_type, float>::value, "must be float!");
#  else
#    error "Unsupported data type (complex is not supported yet in PSMILU)!"
#  endif

  /// \brief perform matching
  /// \tparam IsSymm is \a true, then assume \a A is symmetric
  /// \warning For MC64, symmetric matrices should only contain the lower part
  /// \param[in] A input matrix
  /// \param[in] control control parameters
  /// \param[out] p row permutation vector
  /// \param[out] q column permutation vector
  /// \param[out] s row scaling vector
  /// \param[out] t column scaling vector
  /// \param[out] info output information
  template <bool IsSymm>
  inline void do_matching(const ccs_type &A, const struct mc64_control &control,
                          Array<index_type> &p, Array<index_type> &q,
                          Array<value_type> &s, Array<value_type> &t,
                          struct mc64_info &info) {
    constexpr static size_type int_max =
        static_cast<size_type>(std::numeric_limits<int>::max());

    constexpr static int matrix_type = IsSymm ? 3 : 0, matching_job = 5;

    constexpr static bool consist_int = sizeof(index_type) == sizeof(int);

    static bool warned = false;

    const size_type m = A.nrows(), n = A.ncols(), nnz = A.nnz();
    psmilu_assert(p.size() >= m, "invalid P permutation size");
    psmilu_assert(q.size() >= n, "invalid Q permutation size");
    psmilu_assert(s.size() >= m, "invalid S row scaling size");
    psmilu_assert(t.size() >= n, "invalid T column scaling size");
    psmilu_assert(IsSymm && m == n || !IsSymm && m >= n,
                  "invalid matrix shape");

    // check overflow, since MC aggressively use int, we must ensure it's not
    // overflow
    psmilu_error_if(m > int_max || n > int_max,
                    "matrix sizes (%zdx%zd) exceed INT_MAX, which is beyond "
                    "MC64\'s support!",
                    m, n);
    // since index start array is also int, so nnz must also be bounded by
    // int_max
    psmilu_error_if(
        nnz > int_max,
        "nnz of %zd exceeds INT_MAX, which is beyond MC64\'s support!", nnz);

    psmilu_error_if(ONE_BASED ^ control.f_arrays,
                    "inconsistent index base setup");

    Array<int> pq(m + n);
    if (pq.status() == DATA_UNDEF) psmilu_error("memory allocation failed");
    Array<value_type> st(m + n);
    if (st.status() == DATA_UNDEF) psmilu_error("memory allocation failed");

    int *ptr(nullptr), *row(nullptr);

    if (consist_int) {
      ptr = (int *)A.col_start().data();
      row = (int *)A.row_ind().data();
    } else {
      // we must copy to integer, overflow is already checked though
      if (!warned) {
        psmilu_warning(
            "Inconsistent integer types detected! Deep copy col_start and "
            "row_ind!!");
        warned = true;
      }
      ptr = new (std::nothrow) int[A.col_start().size()];
      psmilu_error_if(!ptr, "memory allocation failed");
      std::copy(A.col_start().cbegin(), A.col_start().cend(), ptr);
      row = new (std::nothrow) int[A.row_ind().size()];
      psmilu_error_if(!row, "memory allocation failed");
      std::copy(A.row_ind().cbegin(), A.row_ind().cend(), row);
    }

    // call matching routine here
    mc64_matching(matching_job, matrix_type, (int)m, (int)n, ptr, row,
                  A.vals().data(), &control, &info, pq.data(), st.data());

    // error handling
    if (info.flag < 0) {
      if (!consist_int) {
        if (ptr) delete[] ptr;
        if (row) delete[] row;
      }
      psmilu_error("MC64 matching returned negative %d flag!", info.flag);
    }

    // compute scaling
    for (size_type i = 0u; i < m; ++i) s[i] = std::exp(st[i]);
    if (IsSymm)
      std::copy(s.cbegin(), s.cend(), t.begin());
    else
      for (size_type i = 0u; i < n; ++i) t[i] = std::exp(st[i + m]);

    // copy permutation
    auto pq_itr = pq.cbegin();
    if (IsSymm)
      psmilu_assert(std::equal(pq_itr, pq_itr + m, pq_itr + m),
                    "symmetric case should have identical row/column "
                    "permutation vectors!");
    std::transform(pq_itr, pq_itr + m, p.begin(),
                   [&](const int i) { return std::abs(i); });
    pq_itr += m;
    std::transform(pq_itr, pq.cend(), q.begin(),
                   [&](const int i) { return std::abs(i); });

    if (!consist_int) {
      if (ptr) delete[] ptr;
      if (row) delete[] row;
    }
  }
};

/// \typedef MatchingDriver
/// \brief unified interface around MC64 backend routine
/// \tparam ValueType value type used
/// \tparam IndexType index type used
/// \tparam OneBased this is ignored for MC64, for API purpose
/// \warning Regardingless OneBased, we always use Fortran index for efficiency
template <class ValueType, class IndexType, bool OneBased = false>
using MatchingDriver = HSL_MC64<ValueType, IndexType, true>;

typedef struct mc64_control matching_control_type;  ///< control parameters
typedef struct mc64_info    matching_info_type;     ///< return info

/// \brief set default control parameters
/// \control[out] control paramteres
/// \note We explicitly assume the index is one based, which is achieved by
///       calling \ref internal::extract_leading_block4matching
inline void set_default_control(matching_control_type &control) {
  mc64_default_control(&control);
  control.f_arrays = 1;
}

#endif  // PSMILU_UNIT_TESTING

namespace internal {

template <bool IsSymm, class ValueType, class IndexType, bool OneBased>
inline CCS<ValueType, IndexType, true> extract_leading_block4matching(
    const CCS<ValueType, IndexType, OneBased> &                   A,
    const typename CCS<ValueType, IndexType, OneBased>::size_type m) {
  using value_type                 = ValueType;
  using index_type                 = IndexType;
  using return_ccs                 = CCS<value_type, index_type, true>;
  using size_type                  = typename return_ccs::size_type;
  constexpr static bool ONE_BASED  = OneBased;
  constexpr static bool CONV_INDEX = !ONE_BASED;
  const auto f_idx = CONV_INDEX ? [](const size_type i) { return i + 1; }
                                : [](const size_type i) { return i; };

  const size_type M = A.nrows(), N = A.ncols();
  psmilu_error_if(
      m >= M || m >= N,
      "leading block size should be strictly smaller than the matrix sizes");

  return_ccs B(m, m);

  // NOTE use col_start to first store the position in A
  auto A_itr = A.row_ind().cbegin();
  B.col_start().resize(m + 1);
  auto &col_start = B.col_start();
  psmilu_error_if(col_start.status() == DATA_UNDEF, "memory allocation failed");
  auto &row_ind = B.row_ind();
  auto &vals    = B.vals();

  if (IsSymm) {
    auto      A_v_itr  = A.vals().cbegin();
    size_type cur_diag = ONE_BASED;
    size_type nnz(0u);
    for (size_type i = 0u; i < m; ++i, ++cur_diag) {
      auto info = find_sorted(A.row_ind_cbegin(i), A.row_ind_cend(i), cur_diag);
      // only lower part
      nnz += A.row_ind_cend(i) - info.second;
      col_start[i + 1] = info.second - A_itr;
    }

    // memory allocation
    row_ind.resize(nnz);
    psmilu_error_if(row_ind.status() == DATA_UNDEF, "memory allocation failed");
    vals.resize(nnz);
    psmilu_error_if(vals.status() == DATA_UNDEF, "memory allocation failed");

    auto itr   = row_ind.begin();
    auto v_itr = vals.begin();

    col_start[0] = 1;
    for (size_type i = 0u; i < m; ++i) {
      auto first = A_itr + col_start[i + 1], last = A.row_ind_cend(i);
      auto v_itr       = A_v_itr + col_start[i + 1];
      col_start[i + 1] = col_start[i] + (last - first);
      for (auto a_itr = first; a_itr != last;
           ++a_itr, ++itr, ++v_itr, ++A_v_itr) {
        *itr   = f_idx(*a_itr);
        *v_itr = *A_v_itr;
      }
    }
  } else {
    const size_type tgt = m + ONE_BASED;
    size_type       nnz(0u);
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

    col_start[0] = 1;
    for (size_type i = 0u; i < m; ++i) {
      auto first       = A.row_ind_cbegin(i);
      auto last        = A_itr + col_start[i + 1];
      auto A_v_itr     = A.val_cbegin(i);
      col_start[i + 1] = col_start[i] + (last - first);
      for (auto a_itr = first; a_itr != last;
           ++a_itr, ++itr, ++v_itr, ++A_v_itr) {
        *itr   = f_idx(*a_itr);
        *v_itr = *A_v_itr;
      }
    }
  }

  return B;
}
}  // namespace internal

/*!
 * @}
 */

}  // namespace psmilu

#endif  // _PSMILU_MATCHING_HSL_MC64_HPP
