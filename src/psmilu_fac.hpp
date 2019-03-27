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
#include "psmilu_Timer.hpp"
#include "psmilu_diag_pivot.hpp"
#include "psmilu_inv_thres.hpp"
#include "psmilu_log.hpp"
#include "psmilu_pre.hpp"
#include "psmilu_utils.hpp"

namespace psmilu {
namespace internal {

/*!
 * \addtogroup fac
 * @{
 */

/// \brief extract permutated diagonal
/// \tparam LeftDiagType left scaling vector type, see \ref Array
/// \tparam CcsType input ccs matrix, see \ref CCS
/// \tparam RightDiagType right scaling vector type, see \ref Array
/// \tparam PermType permutation matrix type, see \ref BiPermMatrix
/// \param[in] s row scaling vector
/// \param[in] A input matrix in CCS format
/// \param[in] t column scaling vector
/// \param[in] m leading block size
/// \param[in] p row permutation vector
/// \param[in] q column permutation vector
/// \return permutated diagonal of \a A
///
/// This routine, essentially, is to compute:
///
/// \f[
///   \boldsymbol{D}=\left(\boldsymbol{SAT}\right)_{\boldsymbol{p}_{1:m},
///     \boldsymbol{q}_{1:m}}
/// \f]
///
/// This routine is used before \ref Crout update to extract the initial
/// diagonal entries.
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
    // using binary search
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

#if 0
template <class AugCcsType, class StartArray>
inline typename AugCcsType::crs_type extract_L_B(
    const AugCcsType &L, const typename AugCcsType::size_type m,
    const StartArray &L_start) {
  static_assert(!AugCcsType::ROW_MAJOR, "L must be AugCCS!");
  using crs_type                  = typename AugCcsType::crs_type;
  using size_type                 = typename crs_type::size_type;
  constexpr static bool ONE_BASED = crs_type::ONE_BASED;

  psmilu_assert(m <= L.nrows(), "invalid row size");

  const auto c_idx = [](const size_type i) {
    return to_c_idx<size_type, ONE_BASED>(i);
  };

  crs_type L_B(m, m);
  auto &   row_start = L_B.row_start();
  row_start.resize(m + 1);
  psmilu_error_if(row_start.status() == DATA_UNDEF, "memory allocation failed");
  std::fill(row_start.begin(), row_start.end(), 0);
  // row_start.front() = ONE_BASED;

  auto L_i_begin = L.row_ind().cbegin();
  for (size_type i = 0u; i < m; ++i) {
    auto last = L_i_begin + L_start[i];
    for (auto itr = L.row_ind_cbegin(i); itr != last; ++itr) {
      const auto j = c_idx(*itr);
      psmilu_assert(j < m, "%zd exceeds the L_B bound %zd", j, m);
      ++row_start[j + 1];
    }
  }

  // accumulate nnz
  for (size_type i = 0; i < m; ++i) row_start[i + 1] += row_start[i];
  const size_type nnz = row_start.back();
  if (!nnz) {
    if (ONE_BASED)
      for (auto &v : row_start) ++v;
    return L_B;
  }

  L_B.reserve(nnz);
  auto &col_ind = L_B.col_ind();
  auto &vals    = L_B.vals();
  psmilu_error_if(col_ind.status() == DATA_UNDEF || vals.status() == DATA_UNDEF,
                  "memory allocation failed");

  col_ind.resize(nnz);
  vals.resize(nnz);

  for (size_type i = 0; i < m; ++i) {
    auto last    = L_i_begin + L_start[i];
    auto L_v_itr = L.val_cbegin(i);
    for (auto itr = L.row_ind_cbegin(i); itr != last; ++itr, ++L_v_itr) {
      const auto j          = c_idx(*itr);
      col_ind[row_start[j]] = i + ONE_BASED;
      vals[row_start[j]]    = *L_v_itr;
      ++row_start[j];
    }
  }

  if (ONE_BASED)
    for (auto &v : row_start) ++v;

  return L_B;
}

#else

/// \brief extract the lower part of leading block
/// \tparam AugCcsType augmented ccs storage, see \ref AugCCS
/// \tparam StartArray starting position array, see \ref Array
/// \param[in] L augmented L after \ref Crout update
/// \param[in] m final leading block size
/// \param[in] L_start final starting positions
/// \return the exact lower part in \ref CCS storage
/// \sa extract_U_B
template <class AugCcsType, class StartArray>
inline typename AugCcsType::ccs_type extract_L_B(
    const AugCcsType &L, const typename AugCcsType::size_type m,
    const StartArray &L_start) {
  static_assert(!AugCcsType::ROW_MAJOR, "L must be AugCCS!");
  using ccs_type                  = typename AugCcsType::ccs_type;
  using size_type                 = typename ccs_type::size_type;
  constexpr static bool ONE_BASED = ccs_type::ONE_BASED;
#  ifndef NDEBUG
  const auto            c_idx     = [](const size_type i) {
    return to_c_idx<size_type, ONE_BASED>(i);
  };
#  endif

#  ifndef PSMILU_UNIT_TESTING
  psmilu_assert(m == L.ncols(),
                "column of L(%zd) should have the size of leading block %zd",
                L.ncols(), m);
#  endif  // PSMILU_UNIT_TESTING
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
    // psmilu_warning(
    //     "exactly zero L_B, this most likely is a bug! Continue anyway...");
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

#endif

#if 0
template <class AugCrsType, class StartArray>
inline typename AugCrsType::crs_type extract_U_B(
    const AugCrsType &U, const typename AugCrsType::size_type m,
    const StartArray &U_start) {
  static_assert(AugCrsType::ROW_MAJOR, "U must be AugCRS!");
  using crs_type                  = typename AugCrsType::crs_type;
  using size_type                 = typename crs_type::size_type;
  using index_type                = typename crs_type::index_type;
  constexpr static bool ONE_BASED = crs_type::ONE_BASED;
  const auto            c_idx     = [](const size_type i) {
    return to_c_idx<size_type, ONE_BASED>(i);
  };

  psmilu_assert(m <= U.ncols(), "invalid col size");

  crs_type U_B(m, m);
  auto &   row_start = U_B.row_start();
  row_start.resize(m + 1);
  psmilu_error_if(row_start.status() == DATA_UNDEF, "memory allocation failed");
  row_start.front() = ONE_BASED;

  auto U_i_begin = U.col_ind().cbegin();
  for (size_type i = 0u; i < m; ++i) {
    auto last        = U_i_begin + U_start[i];
    row_start[i + 1] = row_start[i] + (last - U.col_ind_cbegin(i));
  }

  const size_type nnz = row_start[m] - ONE_BASED;

  if (!nnz) return U_B;

  U_B.reserve(nnz);
  auto &col_ind = U_B.col_ind();
  auto &vals    = U_B.vals();
  psmilu_error_if(col_ind.status() == DATA_UNDEF || vals.status() == DATA_UNDEF,
                  "memory allocation failed");

  col_ind.resize(nnz);
  vals.resize(nnz);

  auto itr   = col_ind.begin();
  auto v_itr = vals.begin();

  auto U_v_begin = U.vals().cbegin();

  for (size_type i = 0u; i < m; ++i) {
    itr   = std::copy(U.col_ind_cbegin(i), U_i_begin + U_start[i], itr);
    v_itr = std::copy(U.val_cbegin(i), U_v_begin + U_start[i], v_itr);
  }

  psmilu_assert(itr == col_ind.end(), "fatal!");
  return U_B;
}

#else

/// \brief extract the upper part
/// \tparam AugCrsType augmented crs storage, see \ref AugCRS
/// \tparam StartArray array type for staring positions, see \ref Array
/// \param[in] U upper part in augmented storage after \ref Crout update
/// \param[in] m final leading block size
/// \param[in] U_start final starting positions for \a U
/// \return the exact upper part in \ref CCS format of \a U
/// \sa extract_L_B
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

#  ifndef PSMILU_UNIT_TESTING
  psmilu_assert(m == U.nrows(),
                "row size of U should have the size of leading block %zd", m);
#  endif  // PSMILU_UNIT_TESTING
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
    // psmilu_warning(
    //     "exactly zero L_B, this most likely is a bug! Continue anyway...");
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

#endif

/// \brief extract the \a E part
/// \tparam LeftDiagType left scaling vector type, see \ref Array
/// \tparam CrsType input crs matrix, see \ref CRS
/// \tparam RightDiagType right scaling vector type, see \ref Array
/// \tparam PermType permutation matrix type, see \ref BiPermMatrix
/// \param[in] s row scaling vector
/// \param[in] A input matrix in crs format
/// \param[in] t column scaling vector
/// \param[in] m leading block size (final)
/// \param[in] p row permutation vector
/// \param[in] q column permutation vector
/// \return The \a E part in \ref CCS format
/// \sa extract_F
///
/// This routine is to extract the \a E part \b after \ref Crout update.
/// Essentially, this routine is to perform:
///
/// \f[
///   \boldsymbol{E}=\left(\boldsymbol{SAT}\right)_{\boldsymbol{p}_{m+1:n},
///     \boldsymbol{q}_{1:m}}
/// \f]
template <class LeftDiagType, class CrsType, class RightDiagType,
          class PermType>
inline typename CrsType::other_type extract_E(
    const LeftDiagType &s, const CrsType &A, const RightDiagType &t,
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
    // psmilu_warning("empty E matrix detected!");
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
    // psmilu_warning(
    //     "exactly zero E, this most likely is a bug! Continue anyway...");
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

/// \brief extract the \a F part
/// \tparam LeftDiagType left scaling vector type, see \ref Array
/// \tparam CcsType input ccs matrix, see \ref CCS
/// \tparam RightDiagType right scaling vector type, see \ref Array
/// \tparam PermType permutation matrix type, see \ref BiPermMatrix
/// \tparam BufferType work space array, can be \ref Array or \a std::vector
/// \param[in] s row scaling vector
/// \param[in] A input matrix in CCS format
/// \param[in] t column scaling vector
/// \param[in] m leading block size
/// \param[in] p row permutation vector
/// \param[in] q column permutation vector
/// \param buf work space
/// \return The \a F part in ccs storage
/// \sa extract_E
///
/// Note that unlike extracting the \a E part, this routine takes \ref CCS as
/// input, and with the permutation vectors, we need a value buffer space as
/// an intermidiate storage to store the values so that is will make sorting
/// much easier. The buffer space is a dense array, and can be passed in from
/// that of \a l or \a ut (if squared systems).
///
/// This routine, essentially, is to compute:
///
/// \f[
///   \boldsymbol{F}=\left(\boldsymbol{SAT}\right)_{\boldsymbol{p}_{1:m},
///     \boldsymbol{q}_{m+1:n}}
/// \f]
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
    // psmilu_warning("empty F matrix detected!");
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
    // psmilu_warning(
    //     "exactly zero F, this most likely is a bug! Continue anyway...");
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
/// \brief Core component to filter CCS and CRS types
/// \tparam CsType1 compressed type I
/// \tparam CsType2 compressed type II
///
/// Since we allow user to arbitrarily use either \ref CCS or \ref CRS as input,
/// we need to build the counterpart and, most importantly, determine the
/// \ref CCS and \ref CRS from the input and its counterpart. This helper trait
/// is to accomplish this task.
template <class CsType1, class CsType2>
class CompressedTypeTrait {
  static_assert(CsType1::ROW_MAJOR ^ CsType2::ROW_MAJOR,
                "cannot have same type");
  constexpr static bool _1_IS_CRS = CsType1::ROW_MAJOR;
  ///< flag of type I is \ref CRS

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

/// \brief update the L start positions for symmetric cases
/// \tparam L_AugCcsType augmented ccs type, see \ref AugCCS
/// \tparam L_StartType starting position array type, see \ref Array
/// \param[in] L augmented ccs for L
/// \param[in] row row index for updating
/// \param[out] L_start starting position of L
///
/// The reason we need this routine is because, for symmetric cases, we just
/// want \a L_start pointing to the asymmetric part, i.e. m+1:n region. However,
/// the problem is that \a m decreases whenever we interchange the rows in L.
/// As a result, we need to update \a L_start after each row interchanges. It's
/// worth noting that \a U_start is treated identically due to the higher
/// priority (comparing to L).
///
/// The complexity is \f$\mathcal{O}(\textrm{nnz}(\boldsymbol{L}_{row,:}))\f$
template <class L_AugCcsType, class L_StartType>
inline void update_L_start_symm(const L_AugCcsType &                   L,
                                const typename L_AugCcsType::size_type row,
                                L_StartType &L_start) {
  using index_type  = typename L_AugCcsType::index_type;
  index_type aug_id = L.start_row_id(row);
  while (!L.is_nil(aug_id)) {
    const auto col_idx = L.col_idx(aug_id);
    L_start[col_idx]   = L.val_pos_idx(aug_id);
    aug_id             = L.next_row_id(aug_id);
  }
}

/*!
 * @}
 */ // group fac

}  // namespace internal

/// \brief perform incomplete LU diagonal pivoting factorization for a level
/// \tparam IsSymm if \a true, then assume a symmetric leading block
/// \tparam CsType input compressed storage, either \ref CRS or \ref CCS
/// \tparam CroutStreamer information streamer for \ref Crout update
/// \tparam PrecsType multilevel preconditioner type, \ref Precs and \ref Prec
/// \param[in] A input for this level
/// \param[in] m0 initial leading block size
/// \param[in] N reference \b global size for determining Schur sparsity
/// \param[in] opts control parameters
/// \param[in] Crout_info information streamer, API same as \ref psmilu_info
/// \param[in,out] precs list of preconditioner, newly computed components will
///                      be pushed back to the list.
/// \return Schur complement for next level (if needed), in the same type as
///         that of the input, i.e. \a CsType
/// \ingroup fac
///
/// This is the core algorithm, which has been demonstrated in the paper, i.e.
/// the algorithm 2, \b ilu_dp_factor. There are two modifications: 1) we put
/// the preprocessing inside this routine, and 2) we embed post-processing (
/// computing Schur complement and updating preconditioner components for the
/// current level) as well. The reasons are, for 1), the preprocessing requires
/// the input type to be \ref CCS, which can only be determined inside this
/// routine (we want to keep this routine as clean as possible, we can, of
/// course, extract the preprocessing out and make this routine takes inputs
/// of both CCS and CRS of \a A.); for 2), it's efficient to compute the Schur
/// complement with \a L_start and \a U_start as well as augmented \a L and \a
/// U. Therefore, instead of sorely factorization, this routine is more or less
/// like a level problem solver.
template <bool IsSymm, class CsType, class CroutStreamer, class PrecsType>
inline CsType iludp_factor(const CsType &A, const typename CsType::size_type m0,
                           const typename CsType::size_type N,
                           const Options &opts, const CroutStreamer &Crout_info,
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

  psmilu_assert(m0 <= std::min(A.nrows(), A.ncols()),
                "leading size should be smaller than size of A");
  const size_type cur_level = precs.size() + 1;
#ifndef NDEBUG
  if (IsSymm)
    psmilu_error_if(cur_level != 1u,
                    "symmetric must be applied to first level!");
#endif

  if (psmilu_verbose(INFO, opts))
    psmilu_info("\nenter level %zd.\n", cur_level);

  DefaultTimer timer;

  // build counterpart type
  const other_type A_counterpart(A);

  // now use our trait and its static methods to precisely determine the ccs
  // and crs components.
  const crs_type &A_crs = cs_trait::select_crs(A, A_counterpart);
  const ccs_type &A_ccs = cs_trait::select_ccs(A, A_counterpart);

  if (psmilu_verbose(INFO, opts)) psmilu_info("performing preprocessing...");

  // preprocessing
  timer.start();
  Array<value_type>        s, t;
  BiPermMatrix<index_type> p, q;
  size_type m = do_preprocessing<IsSymm>(A_ccs, m0, opts, s, t, p, q);
  timer.finish();  // prefile pre-processing

  if (psmilu_verbose(INFO, opts)) psmilu_info("time: %gs", timer.time());

#if 0
  std::fill(s.begin(), s.end(), value_type(1));
  std::fill(t.begin(), t.end(), value_type(1));
  p.make_eye();
  q.make_eye();
  m = m0;
#endif

  if (psmilu_verbose(INFO, opts)) psmilu_info("preparing data variables...");

  timer.start();

  // extract diagonal
  auto d = internal::extract_perm_diag(s, A_ccs, t, m, p, q);

  // create U storage
  aug_crs_type U(m, A.ncols());
  psmilu_error_if(U.row_start().status() == DATA_UNDEF,
                  "memory allocation failed for U:row_start at level %zd.",
                  cur_level);
  U.reserve(A.nnz() * opts.alpha_U);
  psmilu_error_if(
      U.col_ind().status() == DATA_UNDEF || U.vals().status() == DATA_UNDEF,
      "memory allocation failed for U-nnz arrays at level %zd.", cur_level);

  // create L storage
  aug_ccs_type L(A.nrows(), m);
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
  Array<index_type> L_start(m), U_start(m);
  psmilu_error_if(
      L_start.status() == DATA_UNDEF || U_start.status() == DATA_UNDEF,
      "memory allocation failed for L_start and/or U_start at level %zd.",
      cur_level);

  // create storage for kappa's
  Array<value_type> kappa_l(m), kappa_ut(m);
  psmilu_error_if(
      kappa_l.status() == DATA_UNDEF || kappa_ut.status() == DATA_UNDEF,
      "memory allocation failed for kappa_l and/or kappa_ut at level %zd.",
      cur_level);

  U.begin_assemble_rows();
  L.begin_assemble_cols();

  // localize parameters
  const auto tau_d = opts.tau_d, tau_kappa = opts.tau_kappa, tau_U = opts.tau_U,
             tau_L   = opts.tau_L;
  const auto alpha_L = opts.alpha_L, alpha_U = opts.alpha_U;

  size_type       interchanges(0);
  const size_type m_in(m);

  if (psmilu_verbose(INFO, opts)) psmilu_info("start Crout update...");

  for (Crout step; step < m; ++step) {
    // first check diagonal
    bool            pvt    = std::abs(1. / d[step]) > tau_d;
    const size_type m_prev = m;

    Crout_info(" Crout step %zd, leading block size %zd", step, m_prev);

    size_type local_interchanges(0);

    // inf loop
    for (;;) {
      //----------------
      // pivoting
      //---------------

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
        ++local_interchanges;
        if (IsSymm) internal::update_L_start_symm(L, m, L_start);
      }

      //----------------
      // inverse thres
      //----------------

      // compute kappa ut
      update_kappa_ut(step, U, kappa_ut);
      // then compute kappa for l
      update_kappa_l<IsSymm>(step, L, kappa_ut, kappa_l);
      const auto k_ut = std::abs(kappa_ut[step]), k_l = std::abs(kappa_l[step]);
      // check pivoting
      pvt = k_ut > tau_kappa || k_l > tau_kappa;

      Crout_info("  kappa_ut=%g, kappa_l=%g, pvt=%s", (double)k_ut, (double)k_l,
                 (pvt ? "yes" : "no"));

      if (pvt) continue;

      Crout_info(
          "  previous/current leading block sizes %zd/%zd, interchanges=%zd",
          m_prev, m, local_interchanges);

      interchanges += local_interchanges;  // accumulate global interchanges

      //------------------------
      // update start positions
      //------------------------

      Crout_info("  updating L_start/U_start and performing Crout update");

      // update U
      step.update_U_start(U, U_start);
      // then update L
      // const bool no_change = m == m_prev;
      step.update_L_start<IsSymm>(L, m, L_start);

      //----------------------
      // compute Crout updates
      //----------------------

      // compute Uk'
      step.compute_ut(s, A_crs, t, p[step], q, L, d, U, U_start, ut);
      // compute Lk
      step.compute_l<IsSymm>(s, A_ccs, t, p, q[step], m, L, L_start, d, U, l);
      // update diagonal entries
#ifndef NDEBUG
      const bool u_is_nonsingular =
#else
      (void)
#endif
          step.scale_inv_diag(d, ut);
      psmilu_assert(!u_is_nonsingular, "u is singular at level %zd step %zd",
                    cur_level, step);
#ifndef NDEBUG
      const bool l_is_nonsingular =
#else
      (void)
#endif
          step.scale_inv_diag(d, l);
      psmilu_assert(!l_is_nonsingular, "l is singular at level %zd step %zd",
                    cur_level, step);
      // update diagonals b4 dropping
      step.update_B_diag<IsSymm>(l, ut, m, d);

      //---------------
      // drop and sort
      //---------------

      const size_type ori_ut_size = ut.size(), ori_l_size = l.size();

      // apply drop for U
      apply_dropping_and_sort(tau_U, k_ut, A_crs.nnz_in_row(p[step]), alpha_U,
                              ut);
      // push back rows to U
      U.push_back_row(step, ut.inds().cbegin(), ut.inds().cbegin() + ut.size(),
                      ut.vals());

      Crout_info("  ut sizes before/after dropping %zd/%zd, drops=%zd",
                 ori_ut_size, ut.size(), ori_ut_size - ut.size());

      if (IsSymm) {
        // for symmetric cases, we need first find the leading block size
        auto info = find_sorted(ut.inds().cbegin(),
                                ut.inds().cbegin() + ut.size(), m + ONE_BASED);
        apply_dropping_and_sort(tau_L, k_l, A_ccs.nnz_in_col(q[step]), alpha_L,
                                l, info.second - ut.inds().cbegin());

#ifndef NDEBUG
        if (info.second != ut.inds().cbegin() &&
            info.second != ut.inds().cbegin() + ut.size() && l.size())
          psmilu_error_if(*(info.second - 1) >= *l.inds().cbegin() ||
                              *(info.second - 1) - ONE_BASED >= m,
                          "l contains symm part (%zd,%zd,%zd)",
                          (size_type)(*(info.second - 1)),
                          (size_type)*l.inds().cbegin(), m);
#endif

        Crout_info(
            "  l sizes (asymm parts) before/after dropping %zd/%zd, drops=%zd",
            ori_l_size, l.size(), ori_l_size - l.size());

        // push back symmetric entries and offsets
        L.push_back_col(step, ut.inds().cbegin(), info.second, ut.vals(),
                        l.inds().cbegin(), l.inds().cbegin() + l.size(),
                        l.vals());
      } else {
        // for asymmetric cases, just do exactly the same things as ut
        apply_dropping_and_sort(tau_L, k_l, A_ccs.nnz_in_col(q[step]), alpha_L,
                                l);

        Crout_info("  l sizes before/after dropping %zd/%zd, drops=%zd",
                   ori_l_size, l.size(), ori_l_size - l.size());

        L.push_back_col(step, l.inds().cbegin(), l.inds().cbegin() + l.size(),
                        l.vals());
      }
      break;
    }  // inf loop

    Crout_info(" Crout step %zd done!", step);
  }

  U.end_assemble_rows();
  L.end_assemble_cols();

  // finalize start positions
  U_start[m - 1] = U.row_start()[m - 1];
  L_start[m - 1] = L.col_start()[m - 1];

  timer.finish();  // profile Crout update

  // now we are done
  if (psmilu_verbose(INFO, opts)) {
    psmilu_info(
        "finish Crout update...\n"
        "\ttotal interchanges=%zd\n"
        "\tleading block size in=%zd\n"
        "\tleading block size out=%zd\n"
        "\tdiff=%zd",
        interchanges, m_in, m, m_in - m);
    psmilu_info("time: %gs", timer.time());
  }

  if (psmilu_verbose(INFO, opts))
    psmilu_info("computing Schur complement (C)...");

  timer.start();

  // compute C version of Schur complement
  crs_type S_tmp;
  compute_Schur_C(s, A_crs, t, p, q, m, A.nrows(), L, d, U, U_start, S_tmp);
  const input_type S(S_tmp);  // if input==crs, then wrap, ow copy

  // compute L_B and U_B
  auto L_B = internal::extract_L_B(L, m, L_start);
  auto U_B = internal::extract_U_B(U, m, U_start);

  if (psmilu_verbose(INFO, opts))
    psmilu_info("nnz(S_C)=%zd, nnz(L_B)=%zd, nnz(U_B)=%zd...", S.nnz(),
                L_B.nnz(), U_B.nnz());

  // test H version
  const size_type nm     = A.nrows() - m;
  const auto      cbrt_N = std::cbrt(N);
  dense_type      S_D;
  psmilu_assert(S_D.empty(), "fatal!");
  if (S.nnz() >= static_cast<size_type>(opts.rho * nm * nm) ||
      nm <= static_cast<size_type>(opts.c_d * cbrt_N)) {
    bool use_h_ver = false;
    S_D            = dense_type::from_sparse(S);
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
      use_h_ver = true;
    }  // H version check
    if (psmilu_verbose(INFO, opts))
      psmilu_info("converted Schur complement (%s) to dense for last level...",
                  (use_h_ver ? "H" : "C"));
  }

  // NOTE that L_B/U_B are CCS, we need CRS, we can save computation with
  // symmetric case
  crs_type L_B2, U_B2;
  if (IsSymm) {
    L_B2.resize(m, m);
    U_B2.resize(m, m);
    L_B2.row_start() = std::move(U_B.col_start());
    U_B2.row_start() = std::move(L_B.col_start());
    L_B2.col_ind()   = std::move(U_B.row_ind());
    U_B2.col_ind()   = std::move(L_B.row_ind());
    L_B2.vals()      = std::move(U_B.vals());
    U_B2.vals()      = std::move(L_B.vals());
  } else {
    L_B2 = crs_type(L_B);
    U_B2 = crs_type(U_B);
  }
  precs.emplace_back(
      m, A.nrows(), std::move(L_B2), std::move(d), std::move(U_B2),
      crs_type(internal::extract_E(s, A_crs, t, m, p, q)),
      crs_type(internal::extract_F(s, A_ccs, t, m, p, q, ut.vals())),
      std::move(s), std::move(t), std::move(p()), std::move(q.inv()));

  // if dense is not empty, then push it back
  if (!S_D.empty()) {
    auto &last_level = precs.back().dense_solver;
    last_level.set_matrix(std::move(S_D));
    last_level.factorize();
    if (psmilu_verbose(INFO, opts))
      psmilu_info("successfully factorized the dense complement...");
  }
#ifndef NDEBUG
  else
    psmilu_error_if(!precs.back().dense_solver.empty(), "should be empty!");
#endif

  timer.finish();  // profile post-processing

  if (psmilu_verbose(INFO, opts)) psmilu_info("time: %gs", timer.time());

  if (psmilu_verbose(INFO, opts)) psmilu_info("\nfinish level %zd.", cur_level);

  return S;
}

}  // namespace psmilu

#endif  // _PSMILU_FAC_HPP
