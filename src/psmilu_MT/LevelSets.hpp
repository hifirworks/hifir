//@HEADER
//----------------------------------------------------------------------------
//                Copyright (C) 2019 The PSMILU AUTHORS
//----------------------------------------------------------------------------
//@HEADER

/// \file psmilu_MT/LevelSets.hpp
/// \brief Level sets for MT triangular solvers
/// \authors Qiao,
///
/// Reference: Dr. Saad, Iterative Methods for Sparse Linear Systems, Chp 11.6

#ifndef _PSMILU_MT_LEVELSETS_HPP
#define _PSMILU_MT_LEVELSETS_HPP

#include <algorithm>
#include <type_traits>

#include "psmilu_Array.hpp"
#include "psmilu_log.hpp"
#include "psmilu_utils.hpp"

namespace psmilu {

/// \class LevelSets
/// \tparam IndexType index type, e.g. \a int
/// \brief Object representing the level sets for data dependency DAG
/// \ingroup mt
template <class IndexType>
class LevelSets {
 public:
  typedef IndexType                       index_type;   ///< index type
  typedef Array<index_type>               iarray_type;  ///< integer array type
  typedef typename iarray_type::size_type size_type;    ///< size type
  typedef typename iarray_type::const_iterator const_iterator;
  ///< iterator

  /// \brief create level sets
  /// \tparam IsL is lower
  /// \tparam CrsType triangular matrix type, see \ref CRS
  /// \param[in] A either lower or upper matrix
  template <bool IsL, class CrsType>
  inline static LevelSets create_from(const CrsType &A) {
    LevelSets LS;
    LS.determine_levels<IsL>(A);
    return LS;
  }

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
    psmilu_assert(lvl < levels(), "%zd exceeds the level number %zd", lvl,
                  levels());
    return _level_start[lvl + 1] - _level_start[lvl];
  }

  /// \brief get the node list start
  /// \param[in] lvl level number
  inline const_iterator node_begin(const size_type lvl) const {
    psmilu_assert(lvl < levels(), "%zd exceeds the level number %zd", lvl,
                  levels());
    return _level_nodes.cbegin() + _level_start[lvl];
  }

  /// \brief get the node list end (pass-of-end)
  /// \param[in] lvl level number
  inline const_iterator node_end(const size_type lvl) const {
    psmilu_assert(lvl < levels(), "%zd exceeds the level number %zd", lvl,
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

  template <bool IsL, class CrsType, typename T = void>
  inline typename std::enable_if<IsL, T>::type determine_levels(
      const CrsType &L) {
    static_assert(CrsType::ROW_MAJOR, "only support row major!");
    constexpr static bool ONE_BASED = CrsType::ONE_BASED;

    const auto c_idx = [](const size_type i) {
      return to_c_idx<size_type, ONE_BASED>(i);
    };

    const size_type n = L.nrows();
    psmilu_error_if(n != L.ncols(), "must be squared matrix");

    iarray_type lvls(n, 0);
    psmilu_error_if(lvls.status() == DATA_UNDEF, "memory allocation failed");

    // build the levels
    for (size_type i = 0; i < n; ++i) {
      index_type tmp(0);
      for (auto itr = L.col_ind_cbegin(i), last = L.col_ind_cend(i);
           itr != last; ++itr)
        tmp = std::max(tmp, lvls[c_idx(*itr)]);
      lvls[i] = 1 + tmp;
    }
    _determine_core(L, lvls);
  }

  template <bool IsL, class CrsType, typename T = void>
  inline typename std::enable_if<!IsL, T>::type determine_levels(
      const CrsType &U) {
    static_assert(CrsType::ROW_MAJOR, "only support row major!");
    constexpr static bool ONE_BASED = CrsType::ONE_BASED;

    const auto c_idx = [](const size_type i) {
      return to_c_idx<size_type, ONE_BASED>(i);
    };

    const size_type n = U.nrows();
    psmilu_error_if(n != U.ncols(), "must be squared matrix");

    iarray_type lvls(n, 0);
    psmilu_error_if(lvls.status() == DATA_UNDEF, "memory allocation failed");

    // build the levels in reversed order
    for (size_type i(n); i != 0u; --i) {
      const auto i1 = i - 1;
      index_type tmp(0);
      for (auto itr = U.col_ind_cbegin(i1), last = U.col_ind_cend(i1);
           itr != last; ++itr)
        tmp = std::max(tmp, lvls[c_idx(*itr)]);
      lvls[i1] = 1 + tmp;
    }
    _determine_core(U, lvls);
  }

 protected:
  iarray_type _level_start;  ///< array for starting positions
  iarray_type _level_nodes;  ///< array for nodes for each set

 protected:
  template <class CrsType>
  inline void _determine_core(const CrsType &A, const iarray_type &lvls) {
    const size_type n = A.nrows();
    // Note that levels range from [1,max_levels]
    const size_type max_levels = *std::max_element(lvls.cbegin(), lvls.cend());
    _level_start.resize(max_levels + 1);
    psmilu_error_if(_level_start.status() == DATA_UNDEF,
                    "memory allocation failed");
    std::fill(_level_start.begin(), _level_start.end(), index_type(0));
    // again, lvls starts from 1
    for (const auto lvl : lvls) ++_level_start[lvl];
    for (size_type i = 0; i < max_levels; ++i)
      _level_start[i + 1] += _level_start[i];
    psmilu_assert((size_type)_level_start[max_levels] == n, "fatal!");
    _level_nodes.resize(_level_start[max_levels]);
    psmilu_error_if(_level_nodes.status() == DATA_UNDEF,
                    "memory allocation failed");
    for (size_type i = 0; i < n; ++i) {
      const auto lvl                    = lvls[i] - 1;
      _level_nodes[_level_start[lvl]++] = i;
    }
    // revert level start
    index_type tmp(0);
    for (size_type i = 0; i < max_levels; ++i) std::swap(tmp, _level_start[i]);
  }
};

}  // namespace psmilu

#endif  // _PSMILU_MT_LEVELSETS_HPP
