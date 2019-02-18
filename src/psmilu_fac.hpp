//@HEADER
//----------------------------------------------------------------------------
//                Copyright (C) 2019 The PSMILU AUTHORS
//----------------------------------------------------------------------------
//@HEADER

/// \file psmilu_fac.hpp
/// \brief Implementation of incomplete multilevel factorization
/// \authors Qiao,

#ifndef _PSMILU_FAC_HPP
#define _PSMILU_FAC_HPP

#include <algorithm>
#include <cmath>
#include <type_traits>
#include <utility>

#include "psmilu_Array.hpp"
#include "psmilu_AugmentedStorage.hpp"
#include "psmilu_Crout.hpp"
#include "psmilu_DenseMatrix.hpp"
#include "psmilu_Options.h"
#include "psmilu_PermMatrix.hpp"
#include "psmilu_Schur.hpp"
#include "psmilu_SparseVec.hpp"
#include "psmilu_diag_pivot.hpp"
#include "psmilu_inv_thres.hpp"
#include "psmilu_log.hpp"
#include "psmilu_utils.hpp"

namespace psmilu {
namespace internal {

/// \ingroup fac
template <class LeftDiagType, class CcsType, class RightDiagType,
          class PermType>
inline Array<typename CcsType::value_type> extract_perm_diag(
    const LeftDiagType &s, const CcsType A, const RightDiagType &t,
    const typename CcsType::size_type m, const PermType &p, const PermType &q) {
  using value_type                = typename CcsType::value_type;
  using size_type                 = typename CcsType::size_type;
  using array_type                = Array<value_type>;
  constexpr static bool ONE_BASED = CcsType::ONE_BASED;
  const auto            ori_idx   = [](const size_type i) {
    return to_ori_idx<size_type, ONE_BASED>(i);
  };

  psmilu_error_if(m > A.nrows() || m > A.ncols(),
                  "invalid leading block size %zd", m);

  array_type diag(m);
  psmilu_error_if(diag.status() == DATA_UNDEF, "memory allocation failed");

  auto v_begin = A.vals().cbegin();
  auto i_begin = A.row_ind().cbegin();
  for (size_type i = 0u; i < m; ++i) {
    psmilu_assert((size_type)q[i] < A.ncols(),
                  "permutated index %zd exceeds the col bound",
                  (size_type)q[i]);
    psmilu_assert((size_type)p[i] < A.nrows(),
                  "permutated index %zd exceeds the row bound",
                  (size_type)p[i]);

    auto info = find_sorted(A.row_ind_cbegin(q[i]), A.row_ind_cend(q[i]),
                            ori_idx(p[i]));
    if (info.first)
      diag[i] = s[p[i]] * *(v_begin + (info.second - i_begin)) * t[q[i]];
    else {
      psmilu_warning("zero diagonal entry %zd detected!", i);
      diag[i] = 0;
    }
  }
  return diag;
}

/// \ingroup fac
template <class AugCcsType, class StartArray>
inline typename AugCcsType::ccs_type extract_L_B(
    const AugCcsType &L, const typename AugCcsType::size_type m,
    const StartArray &L_start) {
  static_assert(!AugCcsType::ROW_MAJOR, "L must be AugCCS!");
  using ccs_type                  = typename AugCcsType::ccs_type;
  using size_type                 = typename ccs_type::size_type;
  constexpr static bool ONE_BASED = ccs_type::ONE_BASED;
#ifndef NDEBUG
  const auto c_idx = [](const size_type i) {
    return to_c_idx<size_type, ONE_BASED>(i);
  };
#endif

#ifndef PSMILU_UNIT_TESTING
  psmilu_assert(m == L.ncols(),
                "column of L should have the size of leading block %zd", m);
#endif  // PSMILU_UNIT_TESTING
  psmilu_assert(m <= L.nrows(), "invalid row size");

  ccs_type L_B(m, m);
  auto &   col_start = L_B.col_start();
  col_start.resize(m + 1);
  psmilu_error_if(col_start.status() == DATA_UNDEF, "memory allocation failed");
  col_start.front() = ONE_BASED;

  auto L_i_begin = L.row_ind().cbegin();
  for (size_type i = 0u; i < m; ++i) {
    auto last = L_i_begin + L_start[i];
    psmilu_assert((last != L.row_ind_cend(i) && c_idx(*last) >= m) ||
                      last == L.row_ind_cend(i),
                  "L_start should not point anywhere inside (1:m,:) region");
    col_start[i + 1] = col_start[i] + (last - L.row_ind_cbegin(i));
  }

  if (!(col_start[m] - ONE_BASED)) {
    psmilu_warning(
        "exactly zero L_B, this most likely is a bug! Continue anyway...");
    return L_B;
  }

  L_B.reserve(col_start[m] - ONE_BASED);
  auto &row_ind = L_B.row_ind();
  auto &vals    = L_B.vals();
  psmilu_error_if(row_ind.status() == DATA_UNDEF || vals.status() == DATA_UNDEF,
                  "memory allocation failed");

  row_ind.resize(col_start[m] - ONE_BASED);
  vals.resize(col_start[m] - ONE_BASED);

  auto itr   = row_ind.begin();
  auto v_itr = vals.begin();

  auto L_v_begin = L.vals().cbegin();

  for (size_type i = 0u; i < m; ++i) {
    itr   = std::copy(L.row_ind_cbegin(i), L_i_begin + L_start[i], itr);
    v_itr = std::copy(L.val_cbegin(i), L_v_begin + L_start[i], v_itr);
  }

  psmilu_assert(itr == row_ind.end(), "fatal issue!");

  return L_B;
}

/// \ingroup fac
template <class AugCrsType, class StartArray>
inline typename AugCrsType::ccs_type extract_U_B(
    const AugCrsType &U, const typename AugCrsType::size_type m,
    const StartArray &U_start) {
  static_assert(AugCrsType::ROW_MAJOR, "U must be AugCRS!");
  using ccs_type                  = typename AugCrsType::ccs_type;
  using size_type                 = typename ccs_type::size_type;
  using index_type                = typename ccs_type::index_type;
  constexpr static bool ONE_BASED = ccs_type::ONE_BASED;
  const auto            c_idx     = [](const size_type i) {
    return to_c_idx<size_type, ONE_BASED>(i);
  };

#ifndef PSMILU_UNIT_TESTING
  psmilu_assert(m == U.nrows(),
                "row size of U should have the size of leading block %zd", m);
#endif  // PSMILU_UNIT_TESTING
  psmilu_assert(m <= U.ncols(), "invalid col size");

  ccs_type U_B(m, m);
  auto &   col_start = U_B.col_start();
  col_start.resize(m + 1);
  psmilu_error_if(col_start.status() == DATA_UNDEF, "memory allocation failed");
  // fill everything with zero
  std::fill(col_start.begin(), col_start.end(), 0);

  auto U_i_begin = U.col_ind().cbegin();
  for (size_type i = 0u; i < m; ++i) {
    auto last = U_i_begin + U_start[i];
    psmilu_assert((last != U.col_ind_cend(i) && c_idx(*last) >= m) ||
                      last == U.col_ind_cend(i),
                  "U_start should not point anywhere inside (:,1:m) region");
    std::for_each(U.col_ind_cbegin(i), last,
                  [&](const index_type i) { ++col_start[c_idx(i) + 1]; });
  }

  // compute nnz
  for (size_type i = 0u; i < m; ++i) col_start[i + 1] += col_start[i];

  if (!col_start[m]) {
    psmilu_warning(
        "exactly zero L_B, this most likely is a bug! Continue anyway...");
    return U_B;
  }

  U_B.reserve(col_start[m]);
  auto &row_ind = U_B.row_ind();
  auto &vals    = U_B.vals();
  psmilu_error_if(row_ind.status() == DATA_UNDEF || vals.status() == DATA_UNDEF,
                  "memory allocation failed");
  row_ind.resize(col_start[m]);
  vals.resize(col_start[m]);

  for (size_type i = 0u; i < m; ++i) {
    auto last = U_i_begin + U_start[i];
    auto itr  = U.col_ind_cbegin(i);
    for (auto v_itr = U.val_cbegin(i); itr != last; ++itr, ++v_itr) {
      const auto j          = c_idx(*itr);
      row_ind[col_start[j]] = i + ONE_BASED;
      vals[col_start[j]]    = *v_itr;
      ++col_start[j];
    }
  }

  psmilu_assert(col_start[m] == col_start[m - 1], "fatal issue!");

  // revert col_start
  index_type tmp(0);
  for (size_type i = 0u; i < m; ++i) std::swap(col_start[i], tmp);

  if (ONE_BASED)
    std::for_each(col_start.begin(), col_start.end(),
                  [](index_type &i) { ++i; });

  return U_B;
}

/// \ingroup fac
template <class LeftDiagType, class CrsType, class RightdiagType,
          class PermType>
inline typename CrsType::other_type extract_E(
    const LeftDiagType &s, const CrsType &A, const RightdiagType &t,
    const typename CrsType::size_type m, const PermType &p, const PermType &q) {
  // it's efficient to extract E from CRS
  static_assert(CrsType::ROW_MAJOR, "input A must be CRS!");
  using ccs_type   = typename CrsType::other_type;
  using size_type  = typename CrsType::size_type;
  using index_type = typename CrsType::index_type;
  const auto c_idx = [](const size_type i) {
    return to_c_idx<size_type, CrsType::ONE_BASED>(i);
  };

  const size_type n = A.nrows();

  psmilu_error_if(m > n || m > A.ncols(),
                  "leading block size %zd should not exceed matrix size", m);
  const size_type N = n - m;
  ccs_type        E(N, m);
  if (!N) {
    psmilu_warning("empty E matrix detected!");
    return E;
  }

  auto &col_start = E.col_start();
  col_start.resize(m + 1);
  psmilu_error_if(col_start.status() == DATA_UNDEF, "memory allocation failed");
  std::fill(col_start.begin(), col_start.end(), 0);

  for (size_type i = m; i < n; ++i) {
    for (auto itr = A.col_ind_cbegin(p[i]), last = A.col_ind_cend(p[i]);
         itr != last; ++itr) {
      const size_type qi = q.inv(c_idx(*itr));
      if (qi < m) ++col_start[qi + 1];
    }
  }

  // accumulate for nnz
  for (size_type i = 0u; i < m; ++i) col_start[i + 1] += col_start[i];

  if (!col_start[m]) {
    psmilu_warning(
        "exactly zero E, this most likely is a bug! Continue anyway...");
    return E;
  }

  E.reserve(col_start[m]);
  auto &row_ind = E.row_ind();
  auto &vals    = E.vals();

  psmilu_error_if(row_ind.status() == DATA_UNDEF || vals.status() == DATA_UNDEF,
                  "memory allocation failed");

  row_ind.resize(col_start[m]);
  vals.resize(col_start[m]);

  for (size_type i = m; i < n; ++i) {
    const size_type pi  = p[i];
    auto            itr = A.col_ind_cbegin(pi), last = A.col_ind_cend(pi);
    auto            v_itr = A.val_cbegin(pi);
    const auto      si    = s[pi];
    for (; itr != last; ++itr, ++v_itr) {
      const size_type j  = c_idx(*itr);
      const size_type qi = q.inv(j);
      if (qi < m) {
        row_ind[col_start[qi]] = i - m + CrsType::ONE_BASED;
        vals[col_start[qi]]    = si * *v_itr * t[j];
        ++col_start[qi];
      }
    }
  }

  // revert
  index_type tmp(0);
  for (size_type i = 0u; i < m; ++i) std::swap(col_start[i], tmp);

  if (CrsType::ONE_BASED)
    std::for_each(col_start.begin(), col_start.end(),
                  [](index_type &i) { ++i; });

  return E;
}

/// \ingroup fac
template <class LeftDiagType, class CcsType, class RightDiagType,
          class PermType, class BufferType>
inline CcsType extract_F(const LeftDiagType &s, const CcsType &A,
                         const RightDiagType &             t,
                         const typename CcsType::size_type m, const PermType &p,
                         const PermType &q, BufferType &buf) {
  static_assert(!CcsType::ROW_MAJOR, "input A must be CCS!");
  using size_type                 = typename CcsType::size_type;
  using index_type                = typename CcsType::index_type;
  constexpr static bool ONE_BASED = CcsType::ONE_BASED;
  const auto            c_idx     = [](const size_type i) {
    return to_c_idx<size_type, ONE_BASED>(i);
  };

  const size_type n = A.ncols();

  psmilu_error_if(m > n || m > A.nrows(),
                  "leading block size %zd should not exceed matrix size", m);

  const size_type N = n - m;
  CcsType         F(m, N);
  if (!N) {
    psmilu_warning("empty F matrix detected!");
    return F;
  }

  auto &col_start = F.col_start();
  col_start.resize(N + 1);
  psmilu_error_if(col_start.status() == DATA_UNDEF, "memory allocation failed");
  col_start.front() = ONE_BASED;

  for (size_type i = 0u; i < N; ++i) {
    const size_type qi = q[i + m];
    size_type       nnz(0);
    std::for_each(A.row_ind_cbegin(qi), A.row_ind_cend(qi),
                  [&](const index_type i) {
                    if (static_cast<size_type>(p.inv(c_idx(i))) < m) ++nnz;
                  });
    col_start[i + 1] = col_start[i] + nnz;
  }

  if (!(col_start[N] - ONE_BASED)) {
    psmilu_warning(
        "exactly zero F, this most likely is a bug! Continue anyway...");
    return F;
  }

  F.reserve(col_start[N] - ONE_BASED);
  auto &row_ind = F.row_ind();
  auto &vals    = F.vals();
  psmilu_error_if(row_ind.status() == DATA_UNDEF || vals.status() == DATA_UNDEF,
                  "memory allocation failed");

  row_ind.resize(col_start[N] - ONE_BASED);
  vals.resize(col_start[N] - ONE_BASED);

  auto v_itr = vals.begin();

  for (size_type i = 0u; i < N; ++i) {
    const size_type qi      = q[i + m];
    auto            itr     = F.row_ind_begin(i);
    auto            A_itr   = A.row_ind_cbegin(qi);
    auto            A_v_itr = A.val_cbegin(qi);
    const auto      ti      = t[qi];
    for (auto A_last = A.row_ind_cend(qi); A_itr != A_last;
         ++A_itr, ++A_v_itr) {
      const size_type j  = c_idx(*A_itr);
      const size_type pi = p.inv(j);
      if (pi < m) {
        *itr = pi;
        ++itr;
        buf[pi] = s[j] * *A_v_itr * ti;
      }
    }
    std::sort(F.row_ind_begin(i), itr);
    if (!ONE_BASED)
      for (auto itr = F.row_ind_cbegin(i), last = F.row_ind_cend(i);
           itr != last; ++itr, ++v_itr)
        *v_itr = buf[*itr];
    else {
      auto last = F.row_ind_end(i);
      for (itr = F.row_ind_begin(i); itr != last; ++itr, ++v_itr) {
        *v_itr = buf[*itr];
        ++*itr;
      }
    }
  }
  return F;
}

/// \class CompressedTypeTrait
/// \ingroup fac
template <class CsType1, class CsType2>
class CompressedTypeTrait {
  static_assert(CsType1::ROW_MAJOR ^ CsType2::ROW_MAJOR,
                "cannot have same type");
  constexpr static bool _1_IS_CRS = CsType1::ROW_MAJOR;

 public:
  typedef typename std::conditional<_1_IS_CRS, CsType1, CsType2>::type crs_type;
  ///< \ref CRS type
  typedef typename crs_type::other_type ccs_type;  ///< \ref CCS type

  /// \brief choose crs type from two inputs
  /// \param[in] A crs type
  /// \return reference to \a A
  /// \sa crs_type, select_ccs
  template <class T = void>
  inline static const crs_type &select_crs(
      const CsType1 &A, const CsType2 & /* B */,
      const typename std::enable_if<_1_IS_CRS, T>::type * = nullptr) {
    return A;
  }

  // dual version
  template <class T = void>
  inline static const crs_type &select_crs(
      const CsType1 & /* A */, const CsType2 &B,
      const typename std::enable_if<!_1_IS_CRS, T>::type * = nullptr) {
    return B;
  }

  /// \brief choose ccs type from two inputs
  /// \param[in] B ccs type
  /// \return reference to \a B
  /// \sa ccs_type, select_crs
  template <class T = void>
  inline static const ccs_type &select_ccs(
      const CsType1 & /* A */, const CsType2 &B,
      const typename std::enable_if<_1_IS_CRS, T>::type * = nullptr) {
    return B;
  }

  // dual version
  template <class T = void>
  inline static const ccs_type &select_ccs(
      const CsType1 &A, const CsType2 & /* B */,
      const typename std::enable_if<!_1_IS_CRS, T>::type * = nullptr) {
    return A;
  }
};

}  // namespace internal

/// \ingroup fac
template <bool IsSymm, class LeftDiagType, class CsType, class RightDiagType,
          class PermType, class PrecsType>
inline CsType iludp_factor(LeftDiagType &s, const CsType &A, RightDiagType &t,
                           const typename CsType::size_type m0,
                           const typename CsType::size_type N,
                           const Options &opts, PermType &p, PermType &q,
                           PrecsType &precs) {
  typedef CsType                      input_type;
  typedef typename CsType::other_type other_type;
  using cs_trait = internal::CompressedTypeTrait<input_type, other_type>;
  typedef typename cs_trait::crs_type crs_type;
  typedef typename cs_trait::ccs_type ccs_type;
  typedef AugCRS<crs_type>            aug_crs_type;
  typedef AugCCS<ccs_type>            aug_ccs_type;
  typedef typename CsType::index_type index_type;
  typedef typename CsType::size_type  size_type;
  typedef typename CsType::value_type value_type;
  typedef DenseMatrix<value_type>     dense_type;
  constexpr static bool               ONE_BASED = CsType::ONE_BASED;

  psmilu_assert(A.nrows() == s.size(),
                "row scaling vector size should match the row size of A");
  psmilu_assert(A.ncols() == t.size(),
                "column scaling vector size should match A\'s column size");
  psmilu_assert(A.nrows() == p.size(),
                "row permutation vector size should match A\'s row size");
  psmilu_assert(A.ncols() == q.size(),
                "column permutation vector size should match A\'s column size");
  psmilu_assert(m0 <= std::min(A.nrows(), A.ncols()),
                "leading size should be smaller than size of A");
  const size_type cur_level = precs.size() + 1;
#ifndef NDEBUG
  if (IsSymm)
    psmilu_error_if(cur_level != 1u,
                    "symmetric must be applied to first level!");
#endif

  // build counterpart type
  const other_type A_counterpart(A);
  const crs_type & A_crs = cs_trait::select_crs(A, A_counterpart);
  const ccs_type & A_ccs = cs_trait::select_ccs(A, A_counterpart);

  // extract diagonal
  auto d = internal::extract_perm_diag(s, A_ccs, t, m0, p, q);

  // create U storage
  aug_crs_type U(m0, A.ncols());
  psmilu_error_if(U.row_start().status() == DATA_UNDEF,
                  "memory allocation failed for U:row_start at level %zd.",
                  cur_level);
  U.reserve(A.nnz() * opts.alpha_U);
  psmilu_error_if(
      U.col_ind().status() == DATA_UNDEF || U.vals().status() == DATA_UNDEF,
      "memory allocation failed for U-nnz arrays at level %zd.", cur_level);

  // create L storage
  aug_ccs_type L(A.nrows(), m0);
  psmilu_error_if(L.col_start().status() == DATA_UNDEF,
                  "memory allocation failed for L:col_start at level %zd.",
                  cur_level);
  L.reserve(A.nnz() * opts.alpha_L);
  psmilu_error_if(
      L.row_ind().status() == DATA_UNDEF || L.vals().status() == DATA_UNDEF,
      "memory allocation failed for L-nnz arrays at level %zd.", cur_level);

  // create l and ut buffer
  SparseVector<value_type, index_type, ONE_BASED> l(A.nrows()), ut(A.ncols());

  // create buffer for L and U start
  Array<index_type> L_start(m0), U_start(m0);
  psmilu_error_if(
      L_start.status() == DATA_UNDEF || U_start.status() == DATA_UNDEF,
      "memory allocation failed for L_start and/or U_start at level %zd.",
      cur_level);

  // create storage for kappa's
  Array<value_type> kappa_l(m0), kappa_ut(m0);
  psmilu_error_if(
      kappa_l.status() == DATA_UNDEF || kappa_ut.status() == DATA_UNDEF,
      "memory allocation failed for kappa_l and/or kappa_ut at level %zd.",
      cur_level);

  // initialize m
  size_type m(m0);

  U.begin_assemble_rows();
  L.begin_assemble_cols();

  // localize parameters
  const auto tau_d = opts.tau_d, tau_kappa = opts.tau_kappa, tau_U = opts.tau_U,
             tau_L   = opts.tau_L;
  const auto alpha_L = opts.alpha_L, alpha_U = opts.alpha_U;

  for (Crout step; step < m; ++step) {
    // first check diagonal
    bool            pvt    = std::abs(1. / d[step]) > tau_d;
    const size_type m_prev = m;

    // inf loop
    for (;;) {
      if (pvt) {
        // test m value before plugin m-1 to array accessing
        while (m > step && std::abs(1. / d[m - 1]) > tau_d) --m;
        if (m == step) break;
        // defer bad column to the end for U
        U.interchange_cols(step, m - 1);
        // defer bad row to the end for L
        L.interchange_rows(step, m - 1);
        // udpate p and q; be aware that the inverse mappings are also updated
        p.interchange(step, m - 1);
        q.interchange(step, m - 1);
        // update diagonal since we maintain a permutated version of it
        std::swap(d[step], d[m - 1]);
        --m;
      }
      // compute kappa ut
      update_kappa_ut(step, U, kappa_ut);
      // then compute kappa for l
      update_kappa_l<IsSymm>(step, L, kappa_ut, kappa_l);
      const auto k_ut = std::abs(kappa_ut[step]), k_l = std::abs(kappa_l[step]);
      // check pivoting
      pvt = k_ut > tau_kappa || k_l > tau_kappa;
      if (pvt) continue;
      // update U
      step.update_U_start(U, U_start);
      // then update L
      const bool no_change = m == m_prev;
      step.update_L_start<IsSymm>(L, m, L_start, no_change);
      // compute Uk'
      ut.reset_counter();
      step.compute_ut(s, A_crs, t, p[step], q, L, d, U, U_start, ut);
      // compute Lk
      l.reset_counter();
      step.compute_l<IsSymm>(s, A_ccs, t, p, q[step], m, L, L_start, d, U, l);
      // update diagonal entries
#ifndef NDEBUG
      const bool u_is_nonsingular =
#endif
          step.scale_inv_diag(d, ut);
      psmilu_assert(!u_is_nonsingular, "u is singular at level %zd step %zd",
                    cur_level, step);
#ifndef NDEBUG
      const bool l_is_nonsingular =
#endif
          step.scale_inv_diag(d, l);
      psmilu_assert(!l_is_nonsingular, "l is singular at level %zd step %zd",
                    cur_level, step);
      // apply drop for U
      apply_dropping_and_sort(tau_U, k_ut, A_crs.nnz_in_row(p[step]), alpha_U,
                              ut);
      // apply drop for L, for symmetric case, we will pass in the ut size
      // so that the alg will only drop the remaining part
      apply_dropping_and_sort(tau_L, k_l, A_ccs.nnz_in_col(q[step]), alpha_L, l,
                              IsSymm ? ut.size() : 0u);
      // push back rows to U
      U.push_back_row(step, ut.inds().cbegin(), ut.inds().cbegin() + ut.size(),
                      ut.vals());
      // then push back to L
      if (IsSymm) {
        auto info = find_sorted(ut.inds().cbegin(),
                                ut.inds().cbegin() + ut.size(), m + ONE_BASED);
        L.push_back_col(step, ut.inds().cbegin(), info.second, ut.vals(),
                        l.inds().cbegin(), l.inds().cbegin() + l.size(),
                        l.vals());
      } else
        L.push_back_col(step, l.inds().cbegin(), l.inds().cbegin() + l.size(),
                        l.vals());
      break;
    }  // inf loop
  }

  U.end_assemble_rows();
  L.end_assemble_cols();

  // finalize start positions
  U_start[m - 1] = U.row_start()[m - 1];
  L_start[m - 1] = L.col_start()[m - 1];

  // now we are done

  // compute C version of Schur complement
  crs_type S_tmp;
  compute_Schur_C(s, A_crs, t, p, q, m, A.nrows(), L, d, U, U_start, S_tmp);
  const input_type S(input_type::ROW_MAJOR ? std::move(S_tmp)
                                           : input_type(S_tmp));

  // compute L_B and U_B
  auto L_B = internal::extract_L_B(L, m, L_start);
  auto U_B = internal::extract_U_B(U, m, U_start);

  // test H version
  const size_type nm     = A.nrows() - m;
  const auto      cbrt_N = std::cbrt(N);
  dense_type      S_D;
  psmilu_assert(S_D.empty(), "fatal!");
  if (S.nnz() >= static_cast<size_type>(opts.rho * nm * nm) ||
      nm <= static_cast<size_type>(opts.c_d * cbrt_N)) {
    S_D = dense_type::from_sparse(S);
    if (m <= static_cast<size_type>(opts.c_h * cbrt_N)) {
#ifdef PSMILU_UNIT_TESTING
      ccs_type T_E, T_F;
#endif
      compute_Schur_H(L, L_start, L_B, s, A_ccs, t, p, q, d, U_B, U, S_D
#ifdef PSMILU_UNIT_TESTING
                      ,
                      T_E, T_F
#endif
      );
    }  // H version check
  }

  precs.emplace_back(m, A.nrows(), std::move(L_B), std::move(d), std::move(U_B),
                     internal::extract_E(s, A_crs, t, m, p, q),
                     internal::extract_F(s, A_ccs, t, m, p, q, ut.vals()),
                     std::move(s), std::move(t), std::move(p()),
                     std::move(q.inv()));

  // if dense is not empty, then push it back
  if (!S_D.empty()) {
    auto &last_level = precs.back().dense_solver;
    last_level.set_matrix(std::move(S_D));
    last_level.factorize();
  }
#ifndef NDEBUG
  else
    psmilu_error_if(!precs.back().dense_solver.empty(), "should be empty!");
#endif

  return S;
}

}  // namespace psmilu

#endif  // _PSMILU_FAC_HPP
