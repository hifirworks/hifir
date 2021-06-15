///////////////////////////////////////////////////////////////////////////////
//                  This file is part of the HIFIR library                   //
///////////////////////////////////////////////////////////////////////////////

/*!
 * \file hif/arch/ls_trsv.hpp
 * \brief Levelset-based triangular solver
 * \author Qiao Chen

\verbatim
Copyright (C) 2021 NumGeom Group at Stony Brook University

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
\endverbatim

 */

#ifndef _HIF_ARCH_LSTRSV_HPP
#define _HIF_ARCH_LSTRSV_HPP

#include <algorithm>
#include <cmath>
#include <iterator>
#include <type_traits>

#include "hif/ds/Array.hpp"
#include "hif/ds/CompressedStorage.hpp"
#include "hif/utils/mt.hpp"

namespace hif {

/// \class LevelSets
/// \tparam IndexType index type, e.g. \a int
/// \brief Object representing the level sets for data dependency DAG
/// \ingroup mt
/// \note See reference:
///     Dr. Saad, Iterative Methods for Sparse Linear Systems, Chp 11.6
template <class IndexType>
class LevelSets {
 public:
  typedef IndexType                       index_type;   ///< index type
  typedef Array<index_type>               iarray_type;  ///< integer array type
  typedef typename iarray_type::size_type size_type;    ///< size type
  typedef typename iarray_type::const_iterator const_iterator;
  ///< iterator

  /// \brief default constructor
  LevelSets() = default;

  /// \brief deep- or shallow copy constructor
  /// \param[in] other another level set
  /// \param[in] clone if \a false (default), do shallow copy
  LevelSets(const LevelSets &other, bool clone = false)
      : _level_start(other._level_start, clone),
        _level_nodes(other._level_nodes, clone) {}

  /// \brief default move
  LevelSets(LevelSets &&) = default;

  /// \brief check if empty
  inline bool empty() const { return _level_start.empty(); }

  /// \brief get levels
  inline size_type levels() const {
    return empty() ? 0u : _level_start.size() - 1;
  }

  /// \brief get the number of nodes in a specific level
  /// \param[in] lvl level number
  inline size_type nodes_in_level(const size_type lvl) const {
    hif_assert(lvl < levels(), "%zd exceeds the level number %zd", lvl,
               levels());
    return _level_start[lvl + 1] - _level_start[lvl];
  }

  /// \brief get the node list start
  /// \param[in] lvl level number
  inline const_iterator node_begin(const size_type lvl) const {
    hif_assert(lvl < levels(), "%zd exceeds the level number %zd", lvl,
               levels());
    return _level_nodes.cbegin() + _level_start[lvl];
  }

  /// \brief get the node list end (pass-of-end)
  /// \param[in] lvl level number
  inline const_iterator node_end(const size_type lvl) const {
    hif_assert(lvl < levels(), "%zd exceeds the level number %zd", lvl,
               levels());
    return _level_nodes.cbegin() + _level_start[lvl + 1];
  }

  // aliases
  inline const_iterator node_cbegin(const size_type lvl) const {
    return node_begin(lvl);
  }
  inline const_iterator node_cend(const size_type lvl) const {
    return node_end(lvl);
  }

  /// \brief determine level sets
  /// \tparam IsUpper flag indicating lower or upper systems
  /// \tparam CrsType triangular matrix type, see \ref CRS
  template <bool IsUpper, class CrsType, typename T = void>
  inline typename std::enable_if<!IsUpper, T>::type determine_levels(
      const CrsType &L) {
    static_assert(CrsType::ROW_MAJOR, "only support row major!");

    const size_type n = L.nrows();
    hif_error_if(n != L.ncols(), "must be squared matrix");

    iarray_type lvls(n, 0);
    hif_error_if(lvls.status() == DATA_UNDEF, "memory allocation failed");

    // build the levels
    for (size_type i = 0; i < n; ++i) {
      index_type tmp(0);
      for (auto itr = L.col_ind_cbegin(i), last = L.col_ind_cend(i);
           itr != last; ++itr)
        tmp = std::max(tmp, lvls[*itr]);
      lvls[i] = 1 + tmp;
    }
    _determine_core(L, lvls);
  }

  /// \tparam IsUpper flag indicating lower or upper systems
  /// \tparam CrsType triangular matrix type, see \ref CRS
  template <bool IsUpper, class CrsType, typename T = void>
  inline typename std::enable_if<IsUpper, T>::type determine_levels(
      const CrsType &U) {
    static_assert(CrsType::ROW_MAJOR, "only support row major!");
    using iterator = std::reverse_iterator<const_iterator>;

    const size_type n = U.nrows();
    hif_error_if(n != U.ncols(), "must be squared matrix");

    iarray_type lvls(n, 0);
    hif_error_if(lvls.status() == DATA_UNDEF, "memory allocation failed");

    // build the levels in reversed order
    for (size_type i(n); i != 0u; --i) {
      const auto i1 = i - 1;
      index_type tmp(0);
      auto       last = iterator(U.col_ind_cbegin(i1));
      for (auto itr = iterator(U.col_ind_cend(i1)); itr != last; ++itr)
        tmp = std::max(tmp, lvls[*itr]);
      lvls[i1] = 1 + tmp;
    }
    _determine_core(U, lvls);
  }

 protected:
  /// \brief core for building levelsets
  /// \tparam CrsType triangular matrix type, see \ref CRS
  /// \param[in] A input triangular matrix
  /// \param[in] lvls prebuilt levels
  template <class CrsType>
  inline void _determine_core(const CrsType &A, const iarray_type &lvls) {
    const size_type n = A.nrows();
    // Note that levels range from [1,max_levels]
    const size_type max_levels = *std::max_element(lvls.cbegin(), lvls.cend());
    _level_start.resize(max_levels + 1);
    hif_error_if(_level_start.status() == DATA_UNDEF,
                 "memory allocation failed");
    std::fill(_level_start.begin(), _level_start.end(), index_type(0));
    // again, lvls starts from 1
    for (const auto lvl : lvls) ++_level_start[lvl];
    for (size_type i = 0; i < max_levels; ++i)
      _level_start[i + 1] += _level_start[i];
    hif_assert((size_type)_level_start[max_levels] == n, "fatal!");
    _level_nodes.resize(n);
    hif_error_if(_level_nodes.status() == DATA_UNDEF,
                 "memory allocation failed");
    for (size_type i = 0; i < n; ++i) {
      const auto lvl                    = lvls[i] - 1;
      _level_nodes[_level_start[lvl]++] = i;
    }
    // revert level start
    index_type tmp(0);
    for (size_type i = 0; i < max_levels; ++i) std::swap(tmp, _level_start[i]);
  }

 protected:
  iarray_type _level_start;  ///< array for starting positions
  iarray_type _level_nodes;  ///< array for nodes for each set
};

/// \class LsSpTrSolver
/// \brief Levelset sparse triangular solver
/// \tparam ValueType value type, either \a double or \a float
/// \tparam IndexType index type, e.g. \a int
/// \ingroup mt
template <class ValueType, class IndexType>
class LsSpTrSolver {
 public:
  using levelset_type = LevelSets<IndexType>;             ///< levelset type
  using index_type    = IndexType;                        ///< index type
  using value_type    = ValueType;                        ///< value type
  using index_array   = Array<index_type>;                ///< array type
  using value_array   = Array<value_type>;                ///< value array
  using size_type     = typename index_array::size_type;  ///< size

  /// \brief default constructor
  LsSpTrSolver() = default;

  /// \brief default copy
  LsSpTrSolver(const LsSpTrSolver &) = default;

  /// \brief default move
  LsSpTrSolver(LsSpTrSolver &&) = default;

  /// \brief check emptyness
  inline bool empty() const {
    if (!_A_ref) return true;
    return _sets.empty();
  }

  inline size_type nrows() const { return empty() ? 0u : _A_ref->nrows(); }
  inline size_type ncols() const { return nrows(); }

  /// \brief setup
  inline void setup(const CRS<value_type, index_type> &A) { _A_ref = &A; }

  /// \brief optimize
  /// \tparam IsUpper upper/lower flag
  template <bool IsUpper>
  inline void optimize() {
    hif_assert(_A_ref, "upset matrix");
#ifdef _OPENMP
    _sets.template determine_levels<IsUpper>(*_A_ref);
#endif
  }

  /// \brief determine parallelism
  inline size_type compute_parallelism() const {
    if (empty()) return 0;
    return (size_type)std::ceil(_A_ref->nrows() / (double)_sets.levels());
  }

  /// \brief solve as strict lower
  /// \tparam RhsType right-hand size type
  /// \param[in,out] y input and output of right-hand size and solution, resp
  template <class RhsType>
  inline void solve_as_strict_lower(RhsType &y) const {
    int nthreads = mt::get_nthreads();
    if (nthreads == 0) nthreads = mt::get_nthreads(-1);  // query
    if (nthreads == 1 || compute_parallelism() <= 20u)
      return _A_ref->solve_as_strict_lower(y);
    _solve(*_A_ref, _sets, nthreads, y);
  }

  /// \brief solve as strict upper
  /// \tparam RhsType right-hand size type
  /// \param[in,out] y input and output of right-hand size and solution, resp
  template <class RhsType>
  inline void solve_as_strict_upper(RhsType &y) const {
    int nthreads = mt::get_nthreads();
    if (nthreads == 0) nthreads = mt::get_nthreads(-1);  // query
    if (nthreads == 1 || compute_parallelism() <= 20u)
      return _A_ref->solve_as_strict_upper(y);
    _solve(*_A_ref, _sets, nthreads, y);
  }

 protected:
  /// \brief helper function
  template <class RhsType>
  static void _solve(const CRS<value_type, index_type> &A,
                     const levelset_type &sets, const int nthreads,
                     RhsType &x) {
    const auto nlvls = sets.levels();
#ifdef _OPENMP
#  pragma omp parallel num_threads(nthreads)
#else
    (void)nthreads;
#endif
    do {
      const int thread = mt::get_thread();
      for (size_type lvl(0); lvl < nlvls; ++lvl) {
        auto       first = sets.node_cbegin(lvl);
        const auto part =
            mt::uniform_partition(sets.nodes_in_level(lvl), nthreads, thread);
        auto last = first + part.second;
        for (auto iter = first + part.first; iter != last; ++iter) {
          const auto row   = *iter;
          auto       v_itr = A.val_cbegin(row);
          value_type tmp(0);
          auto       last2 = A.col_ind_cend(row);
          for (auto itr = A.col_ind_cbegin(row); itr != last2; ++itr, ++v_itr)
            tmp += *v_itr * x[*itr];
          x[row] -= tmp;
        }
#pragma omp barrier
      }
    } while (false);  // parallel region
  }

 protected:
  levelset_type                      _sets;   ///< levelsets
  const CRS<value_type, index_type> *_A_ref;  ///< reference to triangular
};

}  // namespace hif

#endif  // _HIF_ARCH_LSTRSV_HPP
