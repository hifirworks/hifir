//@HEADER
//----------------------------------------------------------------------------
//                Copyright (C) 2019 The PSMILU AUTHORS
//----------------------------------------------------------------------------
//@HEADER

/// \file psmilu_MT/tri_slv.hpp
/// \brief triangular solve
/// \authors Qiao,

#ifndef _PSMILU_MT_TRISLV_HPP
#define _PSMILU_MT_TRISLV_HPP

#include <cstddef>

#include "psmilu_utils.hpp"
#include "utils.hpp"

namespace psmilu {
/// \brief solve a row, kernel level function
/// \tparam CrsType matrix type, see \ref CRS
/// \tparam SolType solution array, see \ref Array
/// \param[in] A input matrix
/// \param[in] row row number
/// \param[in,out] x rhs/solution
/// \ingroup mt
/// \note For unit diagonal only
template <class CrsType, class SolType>
inline void solve_row_unit(const CrsType &A, const std::size_t row,
                           SolType &x) {
  static_assert(CrsType::ROW_MAJOR, "must be CRS");
  constexpr static bool ONE_BASED = CrsType::ONE_BASED;
  using size_type                 = typename CrsType::size_type;
  using value_type                = typename CrsType::value_type;

  const auto c_idx = [](const size_type i) {
    return to_c_idx<size_type, ONE_BASED>(i);
  };

  auto       v_itr = A.val_cbegin(row);
  value_type tmp(0);
  for (auto itr = A.col_ind_cbegin(row), last = A.col_ind_cend(row);
       itr != last; ++itr, ++v_itr)
    tmp += *v_itr * x[c_idx(*itr)];
  x[row] -= tmp;
}

namespace mt {

template <class CrsType, class LS_Type, class SolType>
inline void solve_level_unit(const CrsType &A, const LS_Type &ls,
                             const std::size_t lvl, const int thread,
                             const int threads, SolType &x) {
  static_assert(CrsType::ROW_MAJOR, "must be CRS");
  using size_type = typename CrsType::size_type;

  // get set start
  auto            start = ls.node_cbegin(lvl);
  const size_type nodes = ls.nodes_in_level(lvl);
  const size_type m     = nodes / threads * threads;
  for (size_type i = 0; i < m; i += threads) {
    const size_type row = *(start + i + thread);
    solve_row_unit(A, row, x);
  }
  const int offsets = nodes - m;
  if (thread < offsets) solve_row_unit(A, *(start + m + thread), x);
}

template <class CrsType, class LS_Type, class SolType>
inline void solve_unit(const CrsType &A, const LS_Type &ls, const int thread,
                       const int threads, SolType &x) {
  static_assert(CrsType::ROW_MAJOR, "must be CRS");
  using size_type   = typename CrsType::size_type;
  const auto levels = ls.levels();
  for (size_type level(0); level < levels; ++level) {
    solve_level_unit(A, ls, level, thread, threads, x);
#pragma omp barrier
  }
}

template <class U_Type, class DiagType, class L_Type, class LS_Type,
          class SolType>
inline void solve_UDL(const U_Type &U, const DiagType &D, const L_Type &L,
                      const LS_Type &ls_U, const LS_Type &ls_L,
                      const UniformParts &parts_D, const int thread,
                      SolType &x) {
  // NOTE we are dealing with inv(LDU)*x, thus all three parts must be
  // solved separately
  using size_type = typename U_Type::size_type;

  const int threads = parts_D.size();
  // assume sync in
  // solve for inv(L)*x
  solve_unit(L, ls_L, thread, threads, x);
#pragma omp barrier
  // solve for inv(D)*x
  const auto &part = parts_D[thread];
  const auto  n    = part.istart + part.len;
  for (size_type i = part.istart; i < n; ++i) x[i] /= D[i];
#pragma omp barrier
  // solve for inv(U)*x
  solve_unit(U, ls_U, thread, threads, x);

  // not sync!
}
}  // namespace mt

}  // namespace psmilu

#endif  // _PSMILU_MT_TRISLV_HPP