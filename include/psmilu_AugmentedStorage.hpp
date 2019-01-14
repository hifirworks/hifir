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

#include <algorithm>
#include <iterator>
#include <numeric>

#include "psmilu_CompressedStorage.hpp"
#include "psmilu_utils.hpp"

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
  typedef AugmentedCore                   this_type;    ///< this
  typedef typename iarray_type::iterator  iterator;     ///< iterator type
  typedef std::reverse_iterator<iterator> reverse_iterator;

 private:
  constexpr static index_type _EMPTY = std::numeric_limits<index_type>::max();

 public:
  AugmentedCore() = default;

  /// \brief check if a given node handle (size_type) is empty or not
  /// \return if \a true, then the node is empty
  inline constexpr static bool empty(const size_type i) { return i == _EMPTY; }

  /// \brief given a node handle, query its value position (index)
  /// \note The index is in C-based system
  inline size_type val_pos_idx(const size_type nid) const {
    psmilu_assert(nid != _EMPTY && nid < _val_pos.size(), "invalid nid %zd",
                  nid);
    return _val_pos[nid];
  }

  inline iarray_type &      val_pos() { return _val_pos; }
  inline const iarray_type &val_pos() const { return _val_pos; }

 protected:
  /// \brief update linked lists given new nodes from back
  /// \tparam Iter iterator type
  /// \tparam OneBased if \a true, then values in first:last is Fortran indices
  /// \param[in] entry row or column index, in C-based system
  /// \param[in] first starting iterator position of index range
  /// \param[in] last pass-of-end iterator position of index range
  /// \note Complexity: \f$\mathcal{O}(n}\f$, where n is size from first to last
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

  /// \brief get starting node ID
  /// \param[in] lid linked list ID (C-based)
  inline size_type _start_node(const size_type lid) const {
    psmilu_assert(lid < _node_start.size(), "%zd exceeds the node start array",
                  lid);
    return _node_start[lid];
  }

  /// \brief get the next node ID
  /// \param[in] nid node ID
  inline size_type _next_node(const size_type nid) const {
    psmilu_assert(nid != _EMPTY && nid < _node_inds.size(), "invalid nid %zd",
                  nid);
    return _node_inds[nid];
  }

  /// \brief get node item ({col-,row-}index in aug-{ccx,crs}, correspondingly)
  /// \param[in] nid node ID
  inline size_type _node_ind(const size_type nid) const {
    psmilu_assert(nid != _EMPTY && nid < _node_inds.size(), "invalid nid %zd",
                  nid);
    return _node_inds[nid];
  }

  /// \brief given two existing indices, swap them explicitly
  /// \param[in] p1 position 1
  /// \param[in] p2 position 2
  /// \note Complexity: \f$\mathcal{O}(1)\f$
  inline void _swap_val_pos_exp(const size_type p1, const size_type p2) {
    psmilu_assert(p1 < _val_pos.size(), "%zd exceeds val_pos size %zd", p1,
                  _val_pos.size());
    psmilu_assert(p2 < _val_pos.size(), "%zd exceeds val_pos size %zd", p2,
                  _val_pos.size());
    std::swap(_val_pos[p1], _val_pos[p2]);
  }

  /// \brief rotate value positions toward left
  /// \param[in] n **local** size, i.e. how many elements to shift
  /// \param[in] src original position in **global** index respect to _val_pos
  /// \note Complexity: \f$\mathcal{O}(n)\f$
  /// \sa _rotate_val_pos_right
  inline void _rotate_val_pos_left(const size_type n, const size_type src) {
    psmilu_assert(src < _val_pos.size(), "%zd exceeds val_pos size %zd", src,
                  _val_pos.size());
    psmilu_assert(src + n <= _val_pos.size(), "%zd exceeds the length",
                  src + n);
    rotate_left(n, src, _val_pos);
  }

  /// \brief rotate value positions toward right
  /// \param[in] n **local** size, i.e. how many elements to shift
  /// \param[in] src original position in **global** index respect to _val_pos
  /// \note Complexity: \f$\mathcal{O}(n)\f$
  /// \sa _rotate_val_pos_left
  inline void _rotate_val_pos_right(const size_type n, const size_type src) {
    psmilu_assert(src < _val_pos.size(), "%zd exceeds val_pos size %zd", src,
                  _val_pos.size());
    psmilu_assert(src + 1u - n >= 0u, "invalid right rotation");
    rotate_right(n, src, _val_pos);
  }

  /// \brief interface the head and tail nodes given list i and k
  /// \param[in] i i-th linked list
  /// \param[in] k k-th linked list
  inline void _interchange_ik_head_tail(const size_type i, const size_type k) {
    psmilu_assert(i < _node_start.size(), "%zd exceeds node_start size %zd", i,
                  _node_start.size());
    psmilu_assert(k < _node_start.size(), "%zd exceeds node_start_ size %zd", k,
                  _node_start.size());
    std::swap(_node_start[i], _node_start[k]);
    std::swap(_node_end[i], _node_end[k]);
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

/// \class AugCRS
/// \brief Augmented CRS data structure
/// \tparam CrsType A CRS type instantiation
/// \sa AugCCS
template <class CrsType>
class AugCRS : public internal::AugmentedCore<typename CrsType::index_type> {
  typedef internal::AugmentedCore<typename CrsType::index_type> _base;

 public:
  typedef CrsType                          crs_type;   ///< crs
  typedef typename crs_type::size_type     size_type;  ///< size
  constexpr static bool                    ONE_BASED = crs_type::ONE_BASED;
  typedef typename _base::iarray_type      iarray_type;  ///< index array
  typedef typename iarray_type::value_type index_type;   ///< index type

  /// \brief given column index, get the starting column handle
  /// \param[in] col column index
  /// \return first column handle in the augmented data structure
  inline size_type start_col_id(const size_type col) const {
    return _base::_start_node(col);
  }

  /// \brief given a column handle, get its next location
  /// \param[in] col_id column handle/ID
  inline size_type next_col_id(const size_type col_id) const {
    return _base::_next_node(col_id);
  }

  /// \brief given a column handle/ID, get its corresponding row index
  /// \param[in] col_id column handle/ID
  inline size_type row_idx(const size_type col_id) const {
    return _base::_node_ind(col_id);
  }

  using _base::empty;
  using _base::val_pos_idx;

  /// \brief interchange two columns
  /// \note This is the core operation for augmented data structure
  /// \param[in] i i-th column
  /// \param[in] k k-th column
  ///
  /// Mathematically, after calling this routine, the i-th and k-th columns
  /// will be interchanged. The total complexity of this routine is linear with
  /// respect to the total number of nonzeros in columns \a i and \a k.
  void interchange_cols(const size_type i, const size_type k) {
    psmilu_assert(i < _crs.ncols(), "%zd exceeds max ncols", i);
    psmilu_assert(k < _crs.ncols(), "%zd exceeds max ncols", k);
    if (i == k) return;  // fast return if possible
    size_type       i_col_id = start_col_id(i), k_col_id = start_col_id(k);
    const size_type nrows = _crs.nrows();            // current nrows
    const size_type nnz   = _base::_val_pos.size();  // current nnz
    for (;;) {
      const bool i_empty = empty(i_col_id), k_empty = empty(k_col_id);
      if (i_empty && k_empty) break;
      const size_type i_row = !i_empty ? row_idx(i_col_id) : nrows;
      psmilu_assert(i_row <= nrows, "fatal issue");
      const size_type k_row = !k_empty ? row_idx(k_col_id) : nrows;
      psmilu_assert(k_row <= nrows, "fatal issue");
      psmilu_assert(nrows != k_row && i_row != nrows, "fatal issue");
      const size_type i_vp = !i_empty ? val_pos_idx(i_col_id) : nnz;
      const size_type k_vp = !k_empty ? val_pos_idx(k_col_id) : nnz;
      if (i_row == k_row) {
        // both rows exists
        // first, swap value array in CRS
        std::swap(_crs.vals()[i_vp], _crs.vals()[k_vp]);
        // then swap the val position in AugCRS
        _base::_swap_val_pos_exp(i_vp, k_vp);
        // finally, advance to next handles for both i and k
        i_col_id = next_col_id(i_col_id);
        k_col_id = next_col_id(k_col_id);
      } else if (i_row < k_row) {
        // rotate i_row to k_row
        auto i_col_itr_first = _crs.col_ind_begin(i_row),
             i_col_itr_last  = _crs.col_ind_end(i_row),
             i_col_itr_pos   = _crs.col_ind().begin() + i_vp;
        psmilu_assert(std::is_sorted(i_col_itr_first, i_col_itr_last),
                      "%zd is not a sorted row", i_row);
        const index_type k_col = to_ori_idx<index_type, ONE_BASED>(k);
        // O(log n)
        auto srch_info = find_sorted(i_col_itr_first, i_col_itr_last, k_col);
        // we should not find the value, cuz if true, it should to the code
        // block above
        psmilu_debug_code(bool k_col_should_not_exit = !srch_info.first);
        psmilu_assert(k_col_should_not_exit, "see prefix, failed with row %zd",
                      i_row);
        // now, assign new value to i-col
        *i_col_itr_pos         = k_col;
        const bool do_left_rot = i_col_itr_pos < srch_info.second;
        size_type  n           = std::abs(i_col_itr_pos - srch_info.second);
        // since we use lower bound, it may happen that n == 0
        if (n) {
          if (srch_info.second != i_col_itr_last) ++n;
          if (do_left_rot) {
            _base::_rotate_val_pos_left(n, i_vp);  // O(n)
            rotate_left(n, i_vp, _crs.col_ind());  // O(n)
            rotate_left(n, i_vp, _crs.vals());     // O(n)
          } else {
            _base::_rotate_val_pos_right(n, i_vp);  // O(n)
            rotate_right(n, i_vp, _crs.col_ind());  // O(n)
            rotate_right(n, i_vp, _crs.vals());     // O(n)
          }
        }
        // the indices should still be sorted
        psmilu_debug_code(i_col_itr_first = _crs.col_ind_begin(i_row);
                          i_col_itr_last  = _crs.col_ind_end(i_row));
        psmilu_assert(std::is_sorted(i_col_itr_first, i_col_itr_last),
                      "%zd is not a sorted row (after rotation)", i_row);
        // keep k, advance i
        i_col_id = next_col_id(i_col_id);
      } else {
        // rotate k_row to i_row
        auto k_col_itr_first = _crs.col_ind_begin(k_row),
             k_col_itr_last  = _crs.col_ind_begin(k_row),
             k_col_itr_pos   = _crs.col_ind().begin() + k_vp;
        psmilu_assert(std::is_sorted(k_col_itr_first, k_col_itr_last),
                      "%zd is not a sorted row", k_row);
        const index_type i_col = to_ori_idx<index_type, ONE_BASED>(i);
        // O(log n)
        auto srch_info = find_sorted(k_col_itr_first, k_col_itr_last, i_col);
        psmilu_debug_code(bool i_col_should_not_exit = !srch_info.first);
        psmilu_assert(i_col_should_not_exit, "see prefix, failed with row %zd",
                      k_row);
        // assign new column index to k-col
        *k_col_itr_pos         = i_col;
        const bool do_left_rot = k_col_itr_pos < srch_info.second;
        size_type  n           = std::abs(k_col_itr_pos - srch_info.second);
        if (n) {
          if (srch_info.second != k_col_itr_last) ++n;
          if (do_left_rot) {
            _base::_rotate_val_pos_left(n, k_vp);  // O(n)
            rotate_left(n, k_vp, _crs.col_ind());  // O(n)
            rotate_left(n, k_vp, _crs.vals());     // O(n)
          } else {
            _base::_rotate_val_pos_right(n, k_vp);  // O(n)
            rotate_right(n, k_vp, _crs.col_ind());  // O(n)
            rotate_right(n, k_vp, _crs.vals());     // O(n)
          }
        }
        // after rotation, the indices should be sorted as well
        psmilu_debug_code(k_col_itr_first = _crs.col_ind_begin(k_row);
                          k_col_itr_last  = _crs.col_ind_begin(k_row));
        psmilu_assert(std::is_sorted(k_col_itr_first, k_col_itr_last),
                      "%zd is not a sorted row (after rotation)", k_row);
        // keep i, advance k
        k_col_id = next_col_id(k_col_id);
      }
    }
    // finally, swap the head and tail link list
    _base::_interchange_ik_head_tail(i, k);
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

  /// \brief push back a row with column indices
  /// \tparam Iter iterator type
  /// \param[in] row current row index (C-based)
  /// \param[in] first starting iterator
  /// \param[in] last pass-of-end iterator
  template <class Iter>
  inline void push_back_row(const size_type row, Iter first, Iter last) {
    _base::template _push_back_nodes<Iter, ONE_BASED>(row, first, last);
  }

 protected:
  crs_type &_crs;  ///< reference to CRS
};

/// \class AugCCS
/// \brief Augmented CCS data structure
/// \tparam CcsType A CCS type instantiation
/// \sa AugCRS
template <class CcsType>
class AugCCS : public internal::AugmentedCore<typename CcsType::index_type> {
  typedef internal::AugmentedCore<typename CcsType::index_type> _base;

 public:
  typedef CcsType                          ccs_type;   ///< ccs type
  typedef typename ccs_type::size_type     size_type;  ///< size
  constexpr static bool                    ONE_BASED = ccs_type::ONE_BASED;
  typedef typename _base::iarray_type      iarray_type;  ///< index array
  typedef typename iarray_type::value_type index_type;   ///< index type

  /// \brief given row index, get the starting row handle/ID
  /// \param[in] row row index
  /// \return first row handle in the augmented data structure
  inline size_type start_row_id(const size_type row) const {
    return _base::_start_node(row);
  }

  /// \brief given a row handle, get its next location
  /// \param[in] row_id row handle/ID
  inline size_type next_row_id(const size_type row_id) const {
    return _base::_next_node(row_id);
  }

  /// \brief given a row handle/ID, get its corresponding column index
  /// \param[in] row_id row handle/ID
  inline size_type col_idx(const size_type row_id) const {
    return _base::_node_ind(row_id);
  }

  using _base::empty;
  using _base::val_pos_idx;

  /// \brief interchange two rows
  /// \note This is the core operation for augmented data structure
  /// \param[in] i i-th row
  /// \param[in] k k-th row
  ///
  /// Mathematically, after calling this routine, the i-th and k-th rows
  /// will be interchanged. The total complexity of this routine is linear with
  /// respect to the total number of nonzeros in rows \a i and \a k.
  void interchange_rows(const size_type i, const size_type k) {
    psmilu_assert(i < _ccs.nrows(), "%zd exceeds max nrows", i);
    psmilu_assert(k < _ccs.nrows(), "%zd exceeds max nrows", k);
    if (i == k) return;  // fast return
    size_type       i_row_id = start_row_id(i), k_row_id = start_row_id(k);
    const size_type ncols = _ccs.ncols();            // current ncols
    const size_type nnz   = _base::_val_pos.size();  // current nnz
    for (;;) {
      // determine the emptyness
      const bool i_empty = empty(i_row_id), k_empty = empty(k_row_id);
      if (i_empty && k_empty) break;
      const size_type i_col = !i_empty ? col_idx(i_row_id) : ncols;
      const size_type k_col = !k_empty ? col_idx(k_row_id) : ncols;
      psmilu_assert(i_col <= ncols, "fatal issue");
      psmilu_assert(k_col <= ncols, "fatal issue");
      psmilu_assert(k_col != ncols && i_col != ncols, "fatal issue");
      const size_type i_vp = !i_empty ? val_pos_idx(i_row_id) : nnz;
      const size_type k_vp = !k_empty ? val_pos_idx(k_row_id) : nnz;
      if (i_col == k_col) {
        // both columns exist
        // simplest case
        std::swap(_ccs.vals()[i_vp], _ccs.vals()[k_vp]);
        // swap the value positions as well
        _base::_swap_val_pos_exp(i_vp, k_vp);
        // advance handles
        i_row_id = next_row_id(i_row_id);
        k_row_id = next_row_id(k_row_id);
      } else if (i_col < k_col) {
        // rotate i_col to k_col
        auto i_row_itr_first = _ccs.row_ind_begin(i_col),
             i_row_itr_last  = _ccs.row_ind_end(i_col),
             i_row_itr_pos   = _ccs.col_ind().begin() + i_vp;
        psmilu_assert(std::is_sorted(i_row_itr_first, i_row_itr_last),
                      "%zd is not a sorted column", i_col);
        const index_type k_row = to_ori_idx<index_type, ONE_BASED>(k);
        // search O(log n)
        auto srch_info = find_sorted(i_row_itr_first, i_row_itr_last, k_row);
        psmilu_debug_code(bool k_row_should_not_exit = !srch_info.first);
        psmilu_assert(k_row_should_not_exit,
                      "see prefix, failed with column %zd", i_col);
        // assign new row index to i-row
        *i_row_itr_pos         = k_row;
        const bool do_left_rot = i_row_itr_pos < srch_info.second;
        size_type  n           = std::abs(i_row_itr_pos - srch_info.second);
        if (n) {
          if (srch_info.second != i_row_itr_last) ++n;
          if (do_left_rot) {
            _base::_rotate_val_pos_left(n, i_vp);
            rotate_left(n, i_vp, _ccs.row_ind());
            rotate_left(n, i_vp, _ccs.vals());
          } else {
            _base::_rotate_val_pos_right(n, i_vp);
            rotate_right(n, i_vp, _ccs.row_ind());
            rotate_right(n, i_vp, _ccs.vals());
          }
        }
        // NOTE that the indices should maintain sorted after rotation
        psmilu_debug_code(i_row_itr_first = _ccs.row_ind_begin(i_col);
                          i_row_itr_last  = _ccs.row_ind_end(i_col));
        psmilu_assert(std::is_sorted(i_row_itr_first, i_row_itr_last),
                      "%zd is not a sorted column (after rotation)", i_col);
        // advance i
        i_row_id = next_row_id(i_row_id);
      } else {
        // rotate k_col to i_col
        auto k_row_itr_first = _ccs.row_ind_begin(k_col),
             k_row_itr_last  = _ccs.row_ind_end(k_col),
             k_row_itr_pos   = _ccs.col_ind().begin() + k_vp;
        psmilu_assert(std::is_sorted(k_row_itr_first, k_row_itr_last),
                      "%zd is not a sorted column", k_col);
        const index_type i_row = to_ori_idx<index_type, ONE_BASED>(i);
        // search O(log n)
        auto srch_info = find_sorted(k_row_itr_first, k_row_itr_last, i_row);
        psmilu_debug_code(bool i_row_should_not_exit = !srch_info.first);
        psmilu_assert(i_row_should_not_exit,
                      "see prefix, failed with column %zd", k_col);
        // assign new row index to k-row
        *k_row_itr_pos         = i_row;
        const bool do_left_rot = k_row_itr_pos < srch_info.second;
        size_type  n           = std::abs(k_row_itr_pos - srch_info.second);
        if (n) {
          if (srch_info.second != k_row_itr_last) ++n;
          if (do_left_rot) {
            _base::_rotate_val_pos_left(n, k_vp);
            rotate_left(n, k_vp, _ccs.row_ind());
            rotate_left(n, k_vp, _ccs.vals());
          } else {
            _base::_rotate_val_pos_right(n, k_vp);
            rotate_right(n, k_vp, _ccs.row_ind());
            rotate_right(n, k_vp, _ccs.vals());
          }
        }
        // NOTE that the indices should maintain sorted after rotation
        psmilu_debug_code(k_row_itr_first = _ccs.row_ind_begin(k_col);
                          k_row_itr_last  = _ccs.row_ind_end(k_col));
        psmilu_assert(std::is_sorted(k_row_itr_first, k_row_itr_last),
                      "%zd is not a sorted column (after rotation)", k_col);
        // advance k
        k_row_id = next_row_id(k_row_id);
      }
    }
    // finally, swap the head and tail
    _base::_interchange_ik_head_tail(i, k);
  }

  // utilities
  inline ccs_type &         ccs() { return *_ccs; }
  inline const ccs_type &   ccs() const { return *_ccs; }
  inline iarray_type &      col_inds() { return _base::_node_inds; }
  inline const iarray_type &col_inds() const { return _base::_node_inds; }
  inline iarray_type &      row_start() { return _base::_node_start; }
  inline const iarray_type &row_start() const { return _base::_node_start; }
  inline iarray_type &      row_end() { return _base::_node_end; }
  inline const iarray_type *row_end() const { return _base::_node_end; }

  /// \brief push back a column with row indices
  /// \tparam Iter iterator type
  /// \param[in] col current column index (C-based)
  /// \param[in] first starting iterator
  /// \param[in] last pass-of-end iterator
  template <class Iter>
  inline void push_back_col(const size_type col, Iter first, Iter last) {
    _base::template _push_back_nodes<Iter, ONE_BASED>(col, first, last);
  }

 protected:
  ccs_type &_ccs;  ///< reference to CCS
};

}  // namespace psmilu

#endif  // _PSMILU_AUGMENTEDSTORAGE_HPP
