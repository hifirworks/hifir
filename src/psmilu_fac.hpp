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
#include <type_traits>

#include "psmilu_Array.hpp"
#include "psmilu_log.hpp"
#include "psmilu_utils.hpp"

namespace psmilu {
namespace internal {

/// \ingroup fac
template <class CcsType, class PermType>
inline Array<typename CcsType::value_type> extract_perm_diag(
    const CcsType A, const typename CcsType::size_type m, const PermType &p,
    const PermType &q) {
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
      diag[i] = *(v_begin + (info.second - i_begin));
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
}  // namespace psmilu

#endif  // _PSMILU_FAC_HPP
