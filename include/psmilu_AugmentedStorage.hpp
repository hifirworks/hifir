//@HEADER
//----------------------------------------------------------------------------
//                Copyright (C) 2019 The PSMILU AUTHORS
//----------------------------------------------------------------------------
//@HEADER

/// \file psmilu_AugmentedStorage.hpp
/// \brief Implementation of augmented data structure for compressed storage
/// \authors Qiao,
///
/// The key of augmented data structure is to maintain a forward link list.
/// However, using a traditional pointer based link list can list to
/// inefficient memory usage or/and computational cost, thus we utilize an
/// array-based link list implementation. Another shining point of using
/// array-based implementation is the compatibility, i.e. one can implement
/// the augmented extension on all kinds of programming languages.

#ifndef _PSMILU_AUGMENTEDSTORAGE_HPP
#define _PSMILU_AUGMENTEDSTORAGE_HPP

#include <numeric>

#include "psmilu_CompressedStorage.hpp"

namespace psmilu {
namespace internal {

/// \class AugmentedCore
/// \brief A group of forward link lists for CompressedStorage
/// \tparam IndexType index type
/// \sa CompressedStorage
template <class IndexType>
class AugmentedCore {
 public:
  typedef IndexType                       index_type;   ///< index
  typedef Array<index_type>               iarray_type;  ///< container
  typedef typename iarray_type::size_type size_type;    ///< size
  typedef AugmentedCore                   this_type;

 private:
  constexpr static index_type _EMPTY = std::numeric_limits<index_type>::max();

 public:
  AugmentedCore() = default;

  inline constexpr static bool empty(const size_type i) { return i == _EMPTY; }

  inline size_type val_pos_idx(const size_type nid) const {
    psmilu_assert(nid != _EMPTY && nid < _val_pos.size(), "invalid nid %zd",
                  nid);
    return _val_pos[nid];
  }

  inline iarray_type &      val_pos() { return _val_pos; }
  inline const iarray_type &val_pos() const { return _val_pos; }

 protected:
  template <class Iter, bool OneBased>
  inline void _push_back_nodes(const size_type entry, Iter first, Iter last) {
    const auto      nnz   = std::distance(first, last);
    const size_type start = _node_inds.size();
    _node_inds.resize(start + nnz);
    // fill in node ids
    std::fill_n(_node_inds.begin() + start, nnz, entry);
    // resize value positions
    _val_pos.resize(_node_inds.size());
    _node_next.resize(_val_pos.size());
    // NOTE that first:last store the indices in compressed storage
    Iter            itr = first;
    const size_type n   = _node_inds.size();
    for (size_type i = start; i < n; ++i, ++itr) {
      // first update value positions
      _val_pos[i]       = i;
      const size_type j = to_c_idx<size_type, OneBased>(*itr);
      psmilu_assert(j < _node_start.size(),
                    "%zd exceeds the bound in node_start", j);
      // get current ending position
      const size_type jend = _node_end[j] != _EMPTY ? _node_end[j] : i;
      psmilu_assert(_node_next[jend] == _EMPTY,
                    "ending position %zd next is not empty", jend);
      // update next list
      _node_next[jend] = i;
      // finally update _node_end
      _node_end[j] = i;
      // need to check start index as well
      if (_node_start[j] == _EMPTY) _node_start[j] = i;
    }
  }

  inline size_type _start_node(const size_type lid) const {
    psmilu_assert(lid < _node_start.size(), "%zd exceeds the node start array",
                  lid);
    return _node_start[lid];
  }

  inline size_type _next_node(const size_type nid) const {
    psmilu_assert(nid != _EMPTY && nid < _node_inds.size(), "invalid nid %zd",
                  nid);
    return _node_inds[nid];
  }

  inline size_type _node_ind(const size_type nid) const {
    psmilu_assert(nid != _EMPTY && nid < _node_inds.size(), "invalid nid %zd",
                  nid);
    return _node_inds[nid];
  }

 protected:
  iarray_type _node_inds;   ///< indices (values) of each node
  iarray_type _node_start;  ///< head nodes
  iarray_type _node_next;   ///< next nodes
  iarray_type _node_end;    ///< ending positions
  iarray_type _val_pos;     ///< value positions

  // TODO do we need to use size_type for storing the positions? This will
  // almost double the memory usage for if index_type is just 32bit,
  // assuming that size_t is 64bit, which is common.
};
}  // namespace internal

template <class CrsType>
class AugCRS : public internal::AugmentedCore<typename CrsType::index_type> {
  typedef internal::AugmentedCore<typename CrsType::index_type> _base;

 public:
  typedef CrsType                      crs_type;   ///< crs
  typedef typename crs_type::size_type size_type;  ///< size
  constexpr static bool                ONE_BASED = crs_type::ONE_BASED;
  typedef typename _base::iarray_type  iarray_type;

  inline size_type start_col_id(const size_type col) const {
    return _base::_start_node(to_c_idx<size_type, ONE_BASED>(col));
  }

  inline size_type next_col_id(const size_type col_id) const {
    return _base::_next_node(col_id);
  }

  inline size_type row_idx(const size_type col_id) const {
    return _base::_node_ind(col_id);
  }

  // utilities
  inline crs_type &         crs() { return *_crs; }
  inline const crs_type &   crs() const { return *_crs; }
  inline iarray_type &      row_inds() { return _base::_node_inds; }
  inline const iarray_type &row_inds() const { return _base::_node_inds; }
  inline iarray_type &      col_start() { return _base::_node_start; }
  inline const iarray_type &col_start() const { return _base::_node_start; }
  inline iarray_type &      col_end() { return _base::_node_end; }
  inline const iarray_type &col_end() const { return _base::_node_end; }

  template <class Iter>
  inline void push_back_row(const size_type row, Iter first, Iter last) {
    _base::template _push_back_nodes<Iter, ONE_BASED>(row, first, last);
  }

 protected:
  crs_type &_crs;
};

template <class CcsType>
class AugCCS;

}  // namespace psmilu

#endif  // _PSMILU_AUGMENTEDSTORAGE_HPP
