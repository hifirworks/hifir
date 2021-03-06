///////////////////////////////////////////////////////////////////////////////
//                  This file is part of the HIFIR library                   //
///////////////////////////////////////////////////////////////////////////////

/*!
 * \file hif/ds/AugmentedStorage.hpp
 * \brief Core about augmented CRS and CCS storage formats
 * \author Qiao Chen

\verbatim
Copyright (C) 2020 NumGeom Group at Stony Brook University

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

#ifndef _HIF_DS_AUGMENTEDSTORAGE_HPP
#define _HIF_DS_AUGMENTEDSTORAGE_HPP

#include <algorithm>
#include <numeric>

#include "hif/ds/Array.hpp"
#include "hif/utils/common.hpp"

namespace hif {
namespace internal {

/// \class AugmentedCore
/// \brief A group of forward link lists for CompressedStorage
/// \tparam IndexType index type
/// \tparam IndPtrType index_pointer type
/// \sa CompressedStorage
/// \ingroup ds
///
/// This is the version of explicitly maintaining the value positions, i.e.
/// the mapping from linked lists to the compressed storage, and corresponding
/// inverse mapping, i.e. the reversed mapping from compressed storage to the
/// linked lists. Notice that this is a flexible approach, meaning that one
/// can use this data structure to interchange rows/columns in arbitrary orders.
/// However, the drawback is also obvious, we need to mainly two nnz-sized
/// arrays, which are memory inefficient.
///
/// An alternative idea is to take into account of Crout updating, by using
/// the leading positions. This approach is relatively counterintuitive, but
/// can be efficient in memory usage. The drawback is that this will make the
/// augmented data structure only suitable for Crout update, which is less a
/// concern for this project.
template <class IndexType, class IndPtrType = std::ptrdiff_t>
class AugmentedCore {
 public:
  typedef IndexType                       index_type;    ///< index
  typedef Array<index_type>               iarray_type;   ///< container
  typedef IndPtrType                      indptr_type;   ///< index_pointer type
  typedef Array<indptr_type>              iparray_type;  ///< indptr array
  typedef typename iarray_type::size_type size_type;     ///< size
  typedef AugmentedCore                   this_type;     ///< this
  typedef typename iarray_type::iterator  iterator;      ///< iterator type

 protected:
  constexpr static indptr_type _NIL = std::numeric_limits<indptr_type>::max();

 public:
  /// \brief default constructor
  AugmentedCore() = default;

  /// \brief constructor with number of lists
  /// \param[in] nlist number of linked lists
  /// \param[in] nnz total number of nonzeros
  /// \param[in] reserve if \a true (default), then reserve spaces
  explicit AugmentedCore(const size_type nlist, const size_type nnz = 0,
                         bool reserve = true)
      : _node_inds(),
        _node_start(nlist, _NIL),
        _node_next(),
        _node_end(nlist, _NIL),
        _val_pos(),
        _node_counts(nlist, static_cast<index_type>(0)) {
    if (nnz) {
      if (reserve) {
        _node_inds.reserve(nnz);
        _node_next.reserve(nnz);
        _val_pos.reserve(nnz);
        _val_pos_inv.reserve(nnz);
      } else {
        _node_inds.resize(nnz);
        _node_next.resize(nnz);
        _val_pos.resize(nnz);
        _val_pos_inv.reserve(nnz);
      }
    }
  }

  /// \brief shallow copy of clone
  /// \param[in] other another augmented data structure
  /// \param[in] clone if \a false (default), do shallow copy
  AugmentedCore(const this_type &other, bool clone = false)
      : _node_inds(other._node_inds, clone),
        _node_start(other._node_start, clone),
        _node_next(other._node_next, clone),
        _node_end(other._node_end, clone),
        _val_pos(other._val_pos, clone),
        _val_pos_inv(other._val_pos_inv, clone),
        _node_counts(other._node_counts, clone) {}

  // default stuffs
  AugmentedCore(this_type &&) = default;
  this_type &operator=(const this_type &) = default;
  this_type &operator=(this_type &&) = default;

  /// \brief check if a given node handle is NIL or not
  /// \return if \a true, then the node is nil
  inline constexpr static bool is_nil(const indptr_type i) { return i == _NIL; }

  /// \brief given a node handle, query its value position (index)
  /// \note The index is in C-based system
  inline indptr_type val_pos_idx(const size_type nid) const {
    hif_assert(!is_nil((index_type)nid), "NIL node detected");
    hif_assert(nid < _val_pos.size(), "invalid nid %zd", nid);
    return _val_pos[nid];
  }

  inline iparray_type &      val_pos() { return _val_pos; }
  inline const iparray_type &val_pos() const { return _val_pos; }

  /// \brief reserve spaces for nnz arrays
  /// \param[in] nnz total number of nonzeros
  inline void reserve(const size_type nnz) {
    _node_inds.reserve(nnz);
    _node_next.reserve(nnz);
    _val_pos.reserve(nnz);
    _val_pos.reserve(nnz);
    _val_pos_inv.reserve(nnz);
  }

  inline iparray_type &      secondary_start() { return _node_start; }
  inline const iparray_type &secondary_start() const { return _node_start; }
  inline iparray_type &      secondary_end() { return _node_end; }
  inline const iparray_type &secondary_end() const { return _node_end; }
  inline iarray_type &       secondary_counts() { return _node_counts; }
  inline const iarray_type & secondary_counts() const { return _node_counts; }

 protected:
  /// \brief begin assemble nodes
  /// \param[in] nlist number of lists
  inline void _begin_assemble_nodes(const size_type nlist) {
    const static indptr_type nil = _NIL;
    _node_start.resize(nlist);
    _node_end.resize(nlist);
    _node_counts.resize(nlist);
    // resize all nnz arrays to zero, thus reserve them b4hand is the way to go
    _node_inds.resize(0u);
    _node_next.resize(0u);
    _val_pos.resize(0u);
    _val_pos_inv.resize(0u);
    // also for n-sized arrays, set them to nil
    std::fill_n(_node_start.begin(), nlist, nil);
    std::fill_n(_node_end.begin(), nlist, nil);
    std::fill_n(_node_counts.begin(), nlist, index_type(0));
  }

  /// \brief update linked lists given new nodes from back
  /// \tparam Iter iterator type
  /// \param[in] entry row or column index, in C-based system
  /// \param[in] first starting iterator position of index range
  /// \param[in] last pass-of-end iterator position of index range
  /// \note Complexity: \f$\mathcal{O}(n)\f$, where n is size from first to last
  template <class Iter>
  inline void _push_back_nodes(const size_type entry, Iter first, Iter last) {
    const auto      nnz   = std::distance(first, last);
    const size_type start = _node_inds.size();
    _node_inds.resize(start + nnz);
    // fill in node ids
    std::fill_n(_node_inds.begin() + start, nnz, entry);
    // resize value positions
    _val_pos.resize(_node_inds.size());
    _node_next.resize(_val_pos.size());
    _val_pos_inv.resize(_node_next.size());
    // NOTE that first:last store the indices in compressed storage
    Iter            itr = first;
    const size_type n   = _node_inds.size();
    for (size_type i = start; i < n; ++i, ++itr) {
      // for newly added nodes, their next/val-pos/val-pos-inv are same
      _val_pos[i]       = i;
      _val_pos_inv[i]   = i;
      _node_next[i]     = _NIL;
      const size_type j = *itr;
      hif_assert(j < _node_start.size(), "%zd exceeds the bound in node_start",
                 j);
      // if we have _node_end[j] (_node_end is not ready when entry==0)
      if (!is_nil(_node_end[j])) _node_next[_node_end[j]] = i;
      // finally update _node_end
      _node_end[j] = i;
      // need to check start index as well
      if (is_nil(_node_start[j])) _node_start[j] = i;
      // increment the counter
      ++_node_counts[j];
    }
  }

  /// \brief get starting node ID
  /// \param[in] lid linked list ID (C-based)
  inline indptr_type _start_node(const size_type lid) const {
    hif_assert(lid < _node_start.size(), "%zd exceeds the node start array",
               lid);
    return _node_start[lid];
  }

  /// \brief get the next node ID
  /// \param[in] nid node ID
  inline indptr_type _next_node(const size_type nid) const {
    hif_assert(!is_nil((indptr_type)nid), "NIL node detected");
    hif_assert(nid < _node_inds.size(), "invalid nid %zd", nid);
    return _node_next[nid];
  }

  /// \brief get node item ({col-,row-}index in aug-{ccx,crs}, correspondingly)
  /// \param[in] nid node ID
  inline index_type _node_ind(const size_type nid) const {
    hif_assert(!is_nil((indptr_type)nid), "NIL node detected");
    hif_assert(nid < _node_inds.size(), "invalid nid %zd", nid);
    return _node_inds[nid];
  }

  /// \brief given two existing indices, swap them explicitly
  /// \param[in] p1 position 1
  /// \param[in] p2 position 2
  /// \note Complexity: \f$\mathcal{O}(1)\f$
  inline void _swap_val_pos_exp(const size_type p1, const size_type p2) {
    hif_assert(p1 < _val_pos.size(), "%zd exceeds val_pos size %zd", p1,
               _val_pos.size());
    hif_assert(p2 < _val_pos.size(), "%zd exceeds val_pos size %zd", p2,
               _val_pos.size());
    // swap value positions
    std::swap(_val_pos[_val_pos_inv[p1]], _val_pos[_val_pos_inv[p2]]);
    // swap the inverse mapping
    std::swap(_val_pos_inv[p1], _val_pos_inv[p2]);
  }

  /// \brief rotate value positions toward left
  /// \param[in] n **local** size, i.e. how many elements to shift
  /// \param[in] src original position in **global** index
  /// \note Complexity: \f$\mathcal{O}(n)\f$
  /// \sa _rotate_val_pos_right
  ///
  /// Firstly, rotate the inverse mapping, then assign the value positions
  /// accordingly.
  inline void _rotate_val_pos_left(const size_type n, const size_type src) {
    hif_assert(src < _val_pos_inv.size(), "%zd exceeds val_pos_inv size %zd",
               src, _val_pos_inv.size());
    hif_assert(src + n <= _val_pos_inv.size(), "%zd exceeds the length",
               src + n);
    rotate_left(n, src, _val_pos_inv);
    // then assign new positions
    auto itr = _val_pos_inv.cbegin() + src;
    // this is how to build inverse mapping
    for (size_type i = 0u; i < n; ++i, ++itr) {
      hif_assert((size_type)*itr < _val_pos.size(),
                 "inv-val-pos index %zd exceeds value position length",
                 (size_type)*itr);
      _val_pos[*itr] = i + src;
    }
  }

  /// \brief rotate value positions toward right
  /// \param[in] n **local** size, i.e. how many elements to shift
  /// \param[in] src original position in **global** index
  /// \note Complexity: \f$\mathcal{O}(n)\f$
  /// \sa _rotate_val_pos_left
  ///
  /// Firstly, rotate the inverse mapping, then assign the value positions
  /// accordingly.
  inline void _rotate_val_pos_right(const size_type n, const size_type src) {
    hif_assert(src < _val_pos_inv.size(), "%zd exceeds val_pos_inv size %zd",
               src, _val_pos_inv.size());
    // NOTE for clean implementation, we plus 1 before subtract n to avoid
    // ugly integer "overflow" (there is not overflow in unsigned anyway).
    // explian: src-n may become max(size_t), which, to me, seems ugly, even
    // though +1 will make it back to 0...
    const size_type start = src + 1u - n;
    rotate_right(n, src, _val_pos_inv);
    // then assign new positions
    auto itr = _val_pos_inv.cbegin() + start;
    // this is how to build inverse mapping
    for (size_type i = 0u; i < n; ++i, ++itr) {
      hif_assert((size_type)*itr < _val_pos.size(),
                 "inv-val-pos index %zd exceeds value position length",
                 (size_type)*itr);
      _val_pos[*itr] = i + start;
    }
  }

  /// \brief interchange the head and tail nodes given list i and k
  /// \param[in] i i-th linked list
  /// \param[in] k k-th linked list
  /// \note Complexity: \f$\mathcal{O}(1)\f$
  inline void _interchange_ik_head_tail(const size_type i, const size_type k) {
    hif_assert(i < _node_start.size(), "%zd exceeds node_start size %zd", i,
               _node_start.size());
    hif_assert(k < _node_start.size(), "%zd exceeds node_start_ size %zd", k,
               _node_start.size());
    std::swap(_node_start[i], _node_start[k]);
    std::swap(_node_end[i], _node_end[k]);
    std::swap(_node_counts[i], _node_counts[k]);
  }

  /// \brief build augmented cores based on compressed storages
  /// \param[in] nlist number of linked lists
  /// \param[in] ind_start starting indices
  /// \param[in] indices array storing the indices
  /// \warning Advanced usaged
  inline void _build_aug(const size_type nlist, const iparray_type &ind_start,
                         const iarray_type &indices) {
    typedef typename iarray_type::const_iterator const_iterator;
    hif_assert(
        size_type(ind_start.back() - ind_start.front()) == indices.size(),
        "nnz size issue");
    // reserve spaces
    const size_type nnz = indices.size();
    reserve(nnz);                  // reserve space
    _begin_assemble_nodes(nlist);  // resize start/end to nlist
    const size_type n = ind_start.size() - 1u;
    for (size_type i = 0u; i < n; ++i) {
      // hif_assert(j < indices.size(), "%zd exceeds indices size", j);
      const size_type lnnz = ind_start[i + 1] - ind_start[i];
      auto first = indices.cbegin() + ind_start[i], last = first + lnnz;
      // push back nodes
      _push_back_nodes<const_iterator>(i, first, last);
    }
  }

  /// \brief query total number of in a certain list
  inline size_type _nodes_in_list(const size_type i) const {
    hif_assert(i < _node_counts.size(), "%zd exceeds total number of lists %zd",
               i, _node_counts.size());
    return _node_counts[i];
  }

 protected:
  iarray_type  _node_inds;    ///< indices (values) of each node
  iparray_type _node_start;   ///< head nodes
  iparray_type _node_next;    ///< next nodes
  iparray_type _node_end;     ///< ending positions
  iparray_type _val_pos;      ///< value positions
  iparray_type _val_pos_inv;  ///< value position inverse, i.e. inv(_val_pos)
  iarray_type  _node_counts;  ///< number of nodes per list

  // TODO do we need to use size_type for storing the positions? This will
  // almost double the memory usage for if index_type is just 32bit,
  // assuming that size_t is 64bit, which is common.
};
}  // namespace internal

/// \class AugCRS
/// \brief Augmented CRS data structure
/// \tparam CrsType A CRS type instantiation
/// \sa AugCCS
/// \note This class inherits from CRS
/// \ingroup ds
template <class CrsType>
class AugCRS : public CrsType,
               public internal::AugmentedCore<typename CrsType::index_type,
                                              typename CrsType::indptr_type> {
  typedef internal::AugmentedCore<typename CrsType::index_type,
                                  typename CrsType::indptr_type>
      _base;

 public:
  typedef AugCRS                           this_type;         ///< aug crs type
  typedef CrsType                          crs_type;          ///< crs
  typedef typename crs_type::other_type    ccs_type;          ///< ccs
  typedef typename crs_type::size_type     size_type;         ///< size
  typedef typename _base::iarray_type      iarray_type;       ///< index array
  typedef typename iarray_type::value_type index_type;        ///< index type
  typedef typename _base::indptr_type      indptr_type;       ///< index_pointer
  typedef typename _base::iparray_type     iparray_type;      ///< indptr array
  typedef typename crs_type::value_type    value_type;        ///< value
  constexpr static bool                    ROW_MAJOR = true;  ///< crs flag

 private:
  constexpr static indptr_type _NIL = _base::_NIL;

 public:
  /// \brief default constructor
  AugCRS() = default;

  /// \brief constructor for general matrix
  /// \param[in] nrows number of rows
  /// \param[in] ncols number of columns
  /// \param[in] nnz ignored if zero (default)
  /// \param[in] reserve if \a true (default), reserve spaces
  AugCRS(const size_type nrows, const size_type ncols, const size_type nnz = 0u,
         bool reserve = true)
      : crs_type(nrows, ncols, nnz, reserve), _base(ncols, nnz, reserve) {}

  /// \brief constructor to either clone or shallow copy another AugCRS
  /// \param[in] other another augmented CRS
  /// \param[in] clone if \a false (default), then do shallow copy
  AugCRS(const this_type &other, bool clone = false)
      : crs_type(other, clone), _base(other, clone) {}

  // default stuffs
  AugCRS(this_type &&) = default;
  this_type &operator=(const this_type &) = default;
  this_type &operator=(this_type &&) = default;

  /// \brief build augmented data structure based on mature CRS matrices
  /// \param[in] crs mature CRS system
  /// \param[in] clone if \a false (default), \a crs will be shallow copied
  /// \warning Advanced usage routine!
  inline void build_aug(const crs_type &crs, bool clone = false) {
    !clone ? crs_type::operator=(crs)
           : crs_type::operator=(crs_type(crs, true));
    hif_error_if(crs.status() == DATA_UNDEF, "undefined data input matrix");
    hif_error_if(crs.nrows() + 1u != crs.row_start().size(),
                 "row-start array size (%zd) does not agree with nrows (%zd)",
                 crs.row_start().size(), crs.nrows());
    hif_error_if(crs.nnz() != crs.col_ind().size(), "inconsistent nnz");
    _base::_build_aug(crs.ncols(), crs.row_start(), crs.col_ind());
  }

  /// \brief build augmented data structure based on mature CRS
  /// \param[in] crs temp CRS system
  inline void build_aug(crs_type &&crs) {
    crs_type::operator=(crs);
    hif_error_if(crs.status() == DATA_UNDEF, "undefined data input matrix");
    hif_error_if(crs.nrows() + 1u != crs.row_start().size(),
                 "row-start array size (%zd) does not agree with nrows (%zd)",
                 crs.row_start().size(), crs.nrows());
    hif_error_if(crs.nnz() != crs.col_ind().size(), "inconsistent nnz");
    _base::_build_aug(crs.ncols(), crs.row_start(), crs.col_ind());
  }

  /// \brief build augmented data structured based on mature CCS matrices
  /// \param[in] ccs mature CCS system
  /// \warning Advanced usage routine
  inline void build_aug(const ccs_type &ccs) {
    crs_type::operator=(crs_type(ccs));
    hif_error_if(ccs.status() == DATA_UNDEF, "undefined data input matrix");
    hif_error_if(nrows() + 1u != crs_type::row_start().size(),
                 "row-start array size (%zd) does not agree with nrows (%zd)",
                 crs_type::row_start().size(), nrows());
    hif_error_if(crs_type::nnz() != col_ind.size(), "inconsistent nnz");
    _base::_build_aug(ncols(), crs_type::row_start(), col_ind());
  }

  /// \brief given column index, get the starting column handle
  /// \param[in] col column index
  /// \return first column handle in the augmented data structure
  inline indptr_type start_col_id(const size_type col) const {
    return _base::_start_node(col);
  }

  /// \brief unified interface to get starting augmented ID
  inline indptr_type start_aug_id(const size_type col) const {
    return start_col_id(col);
  }

  /// \brief given a column handle, get its next location
  /// \param[in] col_id column handle/ID
  inline indptr_type next_col_id(const size_type col_id) const {
    return _base::_next_node(col_id);
  }

  /// \brief unified interface for advancing augmented ID
  /// \param[in] aug_id augmented ID
  inline indptr_type next_aug_id(const size_type aug_id) const {
    return next_col_id(aug_id);
  }

  /// \brief given a column handle/ID, get its corresponding row index
  /// \param[in] col_id column handle/ID
  inline index_type row_idx(const size_type col_id) const {
    return _base::_node_ind(col_id);
  }

  /// \brief unified interface for getting primary index
  inline index_type primary_idx(const size_type aug_id) const {
    return row_idx(aug_id);
  }

  /// \brief reserve space for nnz arrays
  /// \param[in] nnz total number of nonzeros
  inline void reserve(const size_type nnz) {
    crs_type::reserve(nnz);
    _base::reserve(nnz);
  }

  using _base::is_nil;
  using _base::val_pos_idx;
  using crs_type::col_ind;
  using crs_type::ncols;
  using crs_type::nrows;
  using crs_type::vals;

  /// \brief get value with col id
  /// \param[in] col_id column handle/ID
  inline value_type val_from_col_id(const size_type col_id) const {
    // this is how to get value
    return vals()[val_pos_idx(col_id)];
  }

  /// \brief unified interface for getting value from augmented ID
  /// \param[in] aug_id augmented ID
  inline value_type val_from_aug_id(const size_type aug_id) const {
    return val_from_col_id(aug_id);
  }

  /// \brief get the total number of nonzeros columnwise
  /// \param[in] col column index
  inline size_type nnz_in_col(const size_type col) const {
    return _base::_nodes_in_list(col);
  }

  /// \brief unified interface for getting in secondary (col) direction
  /// \param[in] col column index
  inline size_type nnz_in_secondary(const size_type col) const {
    return nnz_in_col(col);
  }

  /// \brief interchange two columns
  /// \note This is the core operation for augmented data structure
  /// \param[in] i i-th column
  /// \param[in] k k-th column
  ///
  /// Mathematically, after calling this routine, the i-th and k-th columns
  /// will be interchanged. The total complexity of this routine is linear with
  /// respect to the total number of nonzeros in columns \a i and \a k.
  void interchange_cols(const size_type i, const size_type k) {
    hif_assert(i < ncols(), "%zd exceeds max ncols", i);
    hif_assert(k < ncols(), "%zd exceeds max ncols", k);
    if (i == k) return;  // fast return if possible
    // get the starting handles/IDs
    indptr_type i_col_id = start_col_id(i), k_col_id = start_col_id(k);
    for (;;) {
      // determine current statuses
      const bool i_empty = is_nil(i_col_id), k_empty = is_nil(k_col_id);
      // break if both of them are empty/nil (meaning finished)
      if (i_empty && k_empty) break;
      // get the row indices for i and k, as well as the value positions, which
      // are needed to locate the corresponding locations of the vals array
      const indptr_type i_row = !i_empty ? row_idx(i_col_id) : _NIL;
      const indptr_type k_row = !k_empty ? row_idx(k_col_id) : _NIL;
      const indptr_type i_vp  = !i_empty ? val_pos_idx(i_col_id) : _NIL;
      const indptr_type k_vp  = !k_empty ? val_pos_idx(k_col_id) : _NIL;
      if (i_row == k_row) {
        // both rows exists
        // first, swap value array in CRS
        std::swap(vals()[i_vp], vals()[k_vp]);
        // then swap the val position in AugCRS
        _base::_swap_val_pos_exp(i_vp, k_vp);
        // finally, advance to next handles for both i and k
        i_col_id = next_col_id(i_col_id);
        k_col_id = next_col_id(k_col_id);
      } else if (i_row < k_row || is_nil(k_row)) {
        // rotate i_row to k_row
        // get the local row range by using standard CRS API
        // then get the position of i-col by using the value position
        auto i_col_itr_first = crs_type::col_ind_begin(i_row),
             i_col_itr_last  = crs_type::col_ind_end(i_row),
             i_col_itr_pos   = crs_type::col_ind().begin() + i_vp;
        hif_assert(std::is_sorted(i_col_itr_first, i_col_itr_last),
                   "%zd is not a sorted row", (size_type)i_row);
        // since k is missing, we need to find the nearest position of k
        // O(log n)
        // NOTE we use lower-bound operation, thus, given the following example
        //  [1 4 8]
        // if we want to find 0, then index 0 is returned
        // if we want to find 2, then index 1 is returned
        // if we want to find 5, then index 2 is returned
        // if we want to find 9, then index 3 (pass-of-end) is returned
        auto srch_info = find_sorted(i_col_itr_first, i_col_itr_last, k);
        // we should not find the value, cuz if true, it should to the code
        // block above
        hif_debug_code(bool k_col_should_not_exit = !srch_info.first);
        hif_assert(k_col_should_not_exit, "see prefix, failed with row %zd",
                   (size_type)i_row);
        // now, assign new value to i-col
        *i_col_itr_pos              = k;
        const bool      do_left_rot = i_col_itr_pos < srch_info.second;
        const size_type n = std::abs(i_col_itr_pos - srch_info.second);
        // since we use lower bound, it may happen that n == 0
        if (n) {
          if (do_left_rot) {
            _base::_rotate_val_pos_left(n, i_vp);  // O(n)
            rotate_left(n, i_vp, col_ind());       // O(n)
            rotate_left(n, i_vp, vals());          // O(n)
          } else {
            // NOTE that +1 due to using lower_bound
            _base::_rotate_val_pos_right(n + 1, i_vp);  // O(n)
            rotate_right(n + 1, i_vp, col_ind());       // O(n)
            rotate_right(n + 1, i_vp, vals());          // O(n)
          }
        }
        // the indices should still be sorted
        hif_debug_code(i_col_itr_first = crs_type::col_ind_begin(i_row);
                       i_col_itr_last  = crs_type::col_ind_end(i_row));
        hif_assert(std::is_sorted(i_col_itr_first, i_col_itr_last),
                   "%zd is not a sorted row (after rotation)",
                   (size_type)i_row);
        // keep k, advance i
        i_col_id = next_col_id(i_col_id);
      } else {
        // rotate k_row to i_row, this is exactly the same as above, just change
        // i to k
        auto k_col_itr_first = crs_type::col_ind_begin(k_row),
             k_col_itr_last  = crs_type::col_ind_end(k_row),
             k_col_itr_pos   = crs_type::col_ind().begin() + k_vp;
        hif_assert(std::is_sorted(k_col_itr_first, k_col_itr_last),
                   "%zd is not a sorted row", (size_type)k_row);
        // O(log n)
        auto srch_info = find_sorted(k_col_itr_first, k_col_itr_last, i);
        hif_debug_code(bool i_col_should_not_exit = !srch_info.first);
        hif_assert(i_col_should_not_exit, "see prefix, failed with row %zd",
                   (size_type)k_row);
        // assign new column index to k-col
        *k_col_itr_pos              = i;
        const bool      do_left_rot = k_col_itr_pos < srch_info.second;
        const size_type n = std::abs(k_col_itr_pos - srch_info.second);
        if (n) {
          if (do_left_rot) {
            _base::_rotate_val_pos_left(n, k_vp);  // O(n)
            rotate_left(n, k_vp, col_ind());       // O(n)
            rotate_left(n, k_vp, vals());          // O(n)
          } else {
            // NOTE that +1 due to using lower_bound
            _base::_rotate_val_pos_right(n + 1, k_vp);  // O(n)
            rotate_right(n + 1, k_vp, col_ind());       // O(n)
            rotate_right(n + 1, k_vp, vals());          // O(n)
          }
        }
        // after rotation, the indices should be sorted as well
        hif_debug_code(k_col_itr_first = crs_type::col_ind_begin(k_row);
                       k_col_itr_last  = crs_type::col_ind_end(k_row));
        hif_assert(std::is_sorted(k_col_itr_first, k_col_itr_last),
                   "%zd is not a sorted row (after rotation)",
                   (size_type)k_row);
        // keep i, advance k
        k_col_id = next_col_id(k_col_id);
      }
    }
    // finally, swap the head and tail link list
    _base::_interchange_ik_head_tail(i, k);
  }

  /// \brief unified interface for interchanging entries
  /// \sa interchange_cols
  inline void interchange_entries(const size_type i, const size_type k) {
    interchange_cols(i, k);
  }

  /// \brief defer a column to the end
  /// \param[in] from which column to defer
  /// \param[in] to the current pass-of-end column index
  /// \warning the parameter \a to must be current pass-of-last column
  inline void defer_col(const size_type from, const size_type to) {
    hif_assert(from < ncols(), "%zd exceeds max ncols", from);
    hif_assert(to < ncols(), "%zd exceeds max ncols", to);
    hif_assert(from < to, "from (%zd) should strictly less than to (%zd)", from,
               to);
    hif_assert(is_nil(start_col_id(to)), "destination must be empty!");
    indptr_type col_id = start_col_id(from);
    while (!is_nil(col_id)) {
      const auto row = row_idx(col_id);
      const auto vp  = val_pos_idx(col_id);
      auto       itr = col_ind().begin() + vp;
      hif_assert(itr != crs_type::col_ind_end(row), "fatal");
      // assign "defer" column index
      *itr              = to;
      const size_type n = crs_type::col_ind_end(row) - itr;
      rotate_left(n, vp, col_ind());
      rotate_left(n, vp, vals());
      _base::_rotate_val_pos_left(n, vp);
      col_id = next_col_id(col_id);
    }
    // NOTE that the start/end should not been visited, thus we can directly
    // reuse the routine "swap"
    _base::_interchange_ik_head_tail(from, to);
  }

  /// \brief unified interface for deferring
  /// \sa defer_col
  inline void defer_entry(const size_type from, const size_type to) {
    defer_col(from, to);
  }

  // utilities
  inline iarray_type &       row_ind() { return _base::_node_inds; }
  inline const iarray_type & row_ind() const { return _base::_node_inds; }
  inline iparray_type &      col_start() { return _base::_node_start; }
  inline const iparray_type &col_start() const { return _base::_node_start; }
  inline iparray_type &      col_end() { return _base::_node_end; }
  inline const iparray_type &col_end() const { return _base::_node_end; }
  inline iparray_type &      col_next() { return _base::_node_next; }
  inline const iparray_type &col_next() const { return _base::_node_next; }
  inline iarray_type &       col_counts() { return _base::_node_counts; }
  inline const iarray_type & col_counts() const { return _base::_node_counts; }

  /// \brief begin to assemble rows
  inline void begin_assemble_rows() {
    crs_type::begin_assemble_rows();
    _base::_begin_assemble_nodes(ncols());
  }

  /// \brief push back a row with column indices
  /// \tparam Iter iterator type
  /// \tparam ValueArray dense value array
  /// \param[in] row current row index (C-based)
  /// \param[in] first starting iterator
  /// \param[in] last pass-of-end iterator
  /// \param[in] v dense value array, the value can be queried from indices
  template <class Iter, class ValueArray>
  inline void push_back_row(const size_type row, Iter first, Iter last,
                            const ValueArray &v) {
    crs_type::template push_back_row<Iter, ValueArray>(row, first, last, v);
    _base::template _push_back_nodes<Iter>(row, first, last);
  }

  /// \brief push back a row with two lists of column indices
  /// \tparam Iter1 iterator type
  /// \tparam ValueArray1 dense value array
  /// \tparam Iter2 iterator type
  /// \tparam ValueArray2 dense value array
  /// \param[in] row current row index (c-based)
  /// \param[in] first1 starting iterator of list 1
  /// \param[in] last1 pass-of-end iterator of list 1
  /// \param[in] v1 dense value array for list 1
  /// \param[in] first2 starting iterator of list 2
  /// \param[in] last2 pass-of-end iterator of list 2
  /// \param[in] v2 dense value array for list 2
  template <class Iter1, class ValueArray1, class Iter2, class ValueArray2>
  inline void push_back_row(const size_type row, Iter1 first1, Iter1 last1,
                            const ValueArray1 &v1, Iter2 first2, Iter2 last2,
                            const ValueArray2 &v2) {
    crs_type::push_back_row(row, first1, last1, v1, first2, last2, v2);
    _base::template _push_back_nodes<Iter1>(row, first1, last1);
    _base::template _push_back_nodes<Iter2>(row, first2, last2);
  }
};

/// \class AugCCS
/// \brief Augmented CCS data structure
/// \tparam CcsType A CCS type instantiation
/// \sa AugCRS
/// \note This class inherits from CCS
/// \ingroup ds
template <class CcsType>
class AugCCS : public CcsType,
               public internal::AugmentedCore<typename CcsType::index_type,
                                              typename CcsType::indptr_type> {
  typedef internal::AugmentedCore<typename CcsType::index_type,
                                  typename CcsType::indptr_type>
      _base;

 public:
  typedef AugCCS                           this_type;     ///< this
  typedef CcsType                          ccs_type;      ///< ccs type
  typedef typename ccs_type::other_type    crs_type;      ///< crs
  typedef typename ccs_type::size_type     size_type;     ///< size
  typedef typename _base::iarray_type      iarray_type;   ///< index array
  typedef typename iarray_type::value_type index_type;    ///< index type
  typedef typename _base::indptr_type      indptr_type;   ///< index_pointer
  typedef typename _base::iparray_type     iparray_type;  ///< indptr array
  typedef typename ccs_type::value_type    value_type;    ///< value
  constexpr static bool ROW_MAJOR = false;                ///< row major flag

 private:
  constexpr static indptr_type _NIL = _base::_NIL;

 public:
  /// \brief default constructor
  AugCCS() = default;

  /// \brief constructor for general matrices
  /// \param[in] nrows number of rows
  /// \param[in] ncols number of columns
  /// \param[in] nnz ignored if zero (default)
  /// \param[in] reserve if \a true (default), then reserve spaces
  AugCCS(const size_type nrows, const size_type ncols, const size_type nnz = 0u,
         bool reserve = true)
      : ccs_type(nrows, ncols, nnz, reserve), _base(nrows, nnz, reserve) {}

  /// \brief constructor to either clone or shallow copy another AugCCS
  /// \param[in] other another augmented CCS
  /// \param[in] clone if \a false (default), then do shallow copy
  AugCCS(const this_type &other, bool clone = false)
      : ccs_type(other, clone), _base(other, clone) {}

  // default stuffs
  AugCCS(this_type &&) = default;
  this_type &operator=(const this_type &) = default;
  this_type &operator=(this_type &&) = default;

  /// \brief build augmented data structure based on mature CCS matrices
  /// \param[in] ccs mature CCS system
  /// \param[in] clone if \a false (default), \a crs will be shallow copied
  /// \warning Advanced usage routine!
  inline void build_aug(const ccs_type &ccs, bool clone = false) {
    !clone ? ccs_type::operator=(ccs)
           : ccs_type::operator=(ccs_type(ccs, true));
    hif_error_if(ccs.status() == DATA_UNDEF, "undefined data input");
    hif_error_if(
        ccs.ncols() + 1u != ccs.col_start().size(),
        "inconsistent sizes between ncols (%zd) and col-start size (%zd)",
        ccs.ncols(), ccs.col_start().size());
    hif_error_if(ccs.nnz() != ccs.row_ind().size(), "inconsistent nnz");
    _base::_build_aug(ccs.nrows(), ccs.col_start(), ccs.row_ind());
  }

  /// \brief build augmented data structure based on mature CCS
  /// \param[in] ccs temp CCS system
  inline void build_aug(crs_type &&ccs) {
    ccs_type::operator=(ccs);
    hif_error_if(ccs.status() == DATA_UNDEF, "undefined data input");
    hif_error_if(
        ccs.ncols() + 1u != ccs.col_start().size(),
        "inconsistent sizes between ncols (%zd) and col-start size (%zd)",
        ccs.ncols(), ccs.col_start().size());
    hif_error_if(ccs.nnz() != ccs.row_ind().size(), "inconsistent nnz");
    _base::_build_aug(ccs.nrows(), ccs.col_start(), ccs.row_ind());
  }

  /// \brief build augmented data structured based on mature CRS matrices
  /// \param[in] crs mature CRS system
  /// \warning Advanced usage routine
  inline void build_aug(const crs_type &crs) {
    ccs_type::operator=(ccs_type(crs));
    hif_error_if(crs.status() == DATA_UNDEF, "undefined data input");
    hif_error_if(
        ncols() + 1u != ccs_type::col_start().size(),
        "inconsistent sizes between ncols (%zd) and col-start size (%zd)",
        ncols(), ccs_type::col_start().size());
    hif_error_if(ccs_type::nnz() != row_ind().size(), "inconsistent nnz");
    _base::_build_aug(nrows(), ccs_type::col_start(), row_ind());
  }

  /// \brief given row index, get the starting row handle/ID
  /// \param[in] row row index
  /// \return first row handle in the augmented data structure
  inline indptr_type start_row_id(const size_type row) const {
    return _base::_start_node(row);
  }

  /// \brief unified interface for getting starting augmented ID
  inline indptr_type start_aug_id(const size_type row) const {
    return start_row_id(row);
  }

  /// \brief given a row handle, get its next location
  /// \param[in] row_id row handle/ID
  inline indptr_type next_row_id(const size_type row_id) const {
    return _base::_next_node(row_id);
  }

  /// \brief unified interface for advancing to next augmented ID
  /// \param[in] aug_id augmented ID
  inline indptr_type next_aug_id(const size_type aug_id) const {
    return next_row_id(aug_id);
  }

  /// \brief given a row handle/ID, get its corresponding column index
  /// \param[in] row_id row handle/ID
  inline index_type col_idx(const size_type row_id) const {
    return _base::_node_ind(row_id);
  }

  /// \brief unified interface for getting primary (col) index
  /// \param[in] aug_id augmented ID
  inline index_type primary_idx(const size_type aug_id) const {
    return col_idx(aug_id);
  }

  /// \brief reserve space for nnz arrays
  /// \param[in] nnz total number of nonzeros
  inline void reserve(const size_type nnz) {
    ccs_type::reserve(nnz);
    _base::reserve(nnz);
  }

  using _base::is_nil;
  using _base::val_pos_idx;
  using ccs_type::ncols;
  using ccs_type::nrows;
  using ccs_type::row_ind;
  using ccs_type::vals;

  /// \brief get the value from row handle
  /// \param[in] row_id row handle/ID
  inline value_type val_from_row_id(const size_type row_id) const {
    return vals()[val_pos_idx(row_id)];
  }

  /// \brief unified interface for value from augmented ID
  /// \param[in] aug_id row augmented ID
  inline value_type val_from_aug_id(const size_type aug_id) const {
    return val_from_row_id(aug_id);
  }

  /// \brief get the total number of nonzeros in row
  /// \param[in] row row index
  inline size_type nnz_in_row(const size_type row) const {
    return _base::_nodes_in_list(row);
  }

  /// \brief unified interface for nnz in secondary direction
  inline size_type nnz_in_secondary(const size_type row) const {
    return nnz_in_row(row);
  }

  /// \brief interchange two rows
  /// \note This is the core operation for augmented data structure
  /// \param[in] i i-th row
  /// \param[in] k k-th row
  ///
  /// Mathematically, after calling this routine, the i-th and k-th rows
  /// will be interchanged. The total complexity of this routine is linear with
  /// respect to the total number of nonzeros in rows \a i and \a k.
  void interchange_rows(const size_type i, const size_type k) {
    hif_assert(i < nrows(), "%zd exceeds max nrows", i);
    hif_assert(k < nrows(), "%zd exceeds max nrows", k);
    if (i == k) return;  // fast return
    // get the starting handles/IDs
    indptr_type i_row_id = start_row_id(i), k_row_id = start_row_id(k);
    for (;;) {
      // determine the emptyness
      const bool i_empty = is_nil(i_row_id), k_empty = is_nil(k_row_id);
      // break if both are nil/empty, meaning that we have finished swapping
      if (i_empty && k_empty) break;
      // get the column indices of i and k, as well as the value positions,
      // which are needed to locate the corresponding locations of the vals
      // array
      const indptr_type i_col = !i_empty ? col_idx(i_row_id) : _NIL;
      const indptr_type k_col = !k_empty ? col_idx(k_row_id) : _NIL;
      const indptr_type i_vp  = !i_empty ? val_pos_idx(i_row_id) : _NIL;
      const indptr_type k_vp  = !k_empty ? val_pos_idx(k_row_id) : _NIL;
      if (i_col == k_col) {
        // both columns exist
        // simplest case
        std::swap(vals()[i_vp], vals()[k_vp]);
        // swap the value positions as well
        _base::_swap_val_pos_exp(i_vp, k_vp);
        // advance handles
        i_row_id = next_row_id(i_row_id);
        k_row_id = next_row_id(k_row_id);
      } else if (i_col < k_col || is_nil(k_col)) {
        // rotate i_col to k_col
        // get the local range of column by using standard CCS API
        // then get the positions of i-row by using the value position
        auto i_row_itr_first = ccs_type::row_ind_begin(i_col),
             i_row_itr_last  = ccs_type::row_ind_end(i_col),
             i_row_itr_pos   = ccs_type::row_ind().begin() + i_vp;
        hif_assert(std::is_sorted(i_row_itr_first, i_row_itr_last),
                   "%zd is not a sorted column", (size_type)i_col);
        // search O(log n)
        auto srch_info = find_sorted(i_row_itr_first, i_row_itr_last, k);
        hif_debug_code(bool k_row_should_not_exit = !srch_info.first);
        hif_assert(k_row_should_not_exit, "see prefix, failed with column %zd",
                   (size_type)i_col);
        // assign new row index to i-row
        *i_row_itr_pos              = k;
        const bool      do_left_rot = i_row_itr_pos < srch_info.second;
        const size_type n = std::abs(i_row_itr_pos - srch_info.second);
        if (n) {
          if (do_left_rot) {
            _base::_rotate_val_pos_left(n, i_vp);
            rotate_left(n, i_vp, row_ind());
            rotate_left(n, i_vp, vals());
          } else {
            // plus 1 due to using lower_bound
            _base::_rotate_val_pos_right(n + 1, i_vp);
            rotate_right(n + 1, i_vp, row_ind());
            rotate_right(n + 1, i_vp, vals());
          }
        }
        // NOTE that the indices should maintain sorted after rotation
        hif_debug_code(i_row_itr_first = ccs_type::row_ind_begin(i_col);
                       i_row_itr_last  = ccs_type::row_ind_end(i_col));
        hif_assert(std::is_sorted(i_row_itr_first, i_row_itr_last),
                   "%zd is not a sorted column (after rotation)",
                   (size_type)i_col);
        // advance i
        i_row_id = next_row_id(i_row_id);
      } else {
        // rotate k_col to i_col, exactly same as above, just change i to k
        auto k_row_itr_first = ccs_type::row_ind_begin(k_col),
             k_row_itr_last  = ccs_type::row_ind_end(k_col),
             k_row_itr_pos   = ccs_type::row_ind().begin() + k_vp;
        hif_assert(std::is_sorted(k_row_itr_first, k_row_itr_last),
                   "%zd is not a sorted column", (size_type)k_col);
        // search O(log n)
        auto srch_info = find_sorted(k_row_itr_first, k_row_itr_last, i);
        hif_debug_code(bool i_row_should_not_exit = !srch_info.first);
        hif_assert(i_row_should_not_exit, "see prefix, failed with column %zd",
                   (size_type)k_col);
        // assign new row index to k-row
        *k_row_itr_pos              = i;
        const bool      do_left_rot = k_row_itr_pos < srch_info.second;
        const size_type n = std::abs(k_row_itr_pos - srch_info.second);
        if (n) {
          if (do_left_rot) {
            _base::_rotate_val_pos_left(n, k_vp);
            rotate_left(n, k_vp, row_ind());
            rotate_left(n, k_vp, vals());
          } else {
            _base::_rotate_val_pos_right(n + 1, k_vp);
            rotate_right(n + 1, k_vp, row_ind());
            rotate_right(n + 1, k_vp, vals());
          }
        }
        // NOTE that the indices should maintain sorted after rotation
        hif_debug_code(k_row_itr_first = ccs_type::row_ind_begin(k_col);
                       k_row_itr_last  = ccs_type::row_ind_end(k_col));
        hif_assert(std::is_sorted(k_row_itr_first, k_row_itr_last),
                   "%zd is not a sorted column (after rotation)",
                   (size_type)k_col);
        // advance k
        k_row_id = next_row_id(k_row_id);
      }
    }
    // finally, swap the head and tail
    _base::_interchange_ik_head_tail(i, k);
  }

  /// \brief unified interface for interchanging
  /// \sa interchange_rows
  inline void interchange_entries(const size_type i, const size_type k) {
    interchange_rows(i, k);
  }

  /// \brief defer a row to the end
  /// \param[in] from which row to defer
  /// \param[in] to the current pass-of-end row index
  /// \warning the parameter \a to must be current pass-of-last row
  inline void defer_row(const size_type from, const size_type to) {
    hif_assert(from < nrows(), "%zd exceeds max nrows", from);
    hif_assert(to < nrows(), "%zd exceeds max nrows", to);
    hif_assert(from < to, "from (%zd) should strictly less than to (%zd)", from,
               to);
    hif_assert(is_nil(start_row_id(to)), "destination must be empty!");
    indptr_type row_id = start_row_id(from);
    while (!is_nil(row_id)) {
      const auto col = col_idx(row_id);
      const auto vp  = val_pos_idx(row_id);
      auto       itr = row_ind().begin() + vp;
      hif_assert(itr != ccs_type::row_ind_end(col), "fatal");
      // assign "defer" column index
      *itr              = to;
      const size_type n = ccs_type::row_ind_end(col) - itr;
      rotate_left(n, vp, row_ind());
      rotate_left(n, vp, vals());
      _base::_rotate_val_pos_left(n, vp);
      row_id = next_row_id(row_id);
    }
    // NOTE that the start/end should not been visited, thus we can directly
    // reuse the routine "swap"
    _base::_interchange_ik_head_tail(from, to);
  }

  /// \brief unified interface for deferring
  /// \sa defer_row
  inline void defer_entry(const size_type from, const size_type to) {
    defer_row(from, to);
  }

  // utilities
  inline iarray_type &       col_ind() { return _base::_node_inds; }
  inline const iarray_type & col_ind() const { return _base::_node_inds; }
  inline iparray_type &      row_start() { return _base::_node_start; }
  inline const iparray_type &row_start() const { return _base::_node_start; }
  inline iparray_type &      row_end() { return _base::_node_end; }
  inline const iparray_type *row_end() const { return _base::_node_end; }
  inline iparray_type &      row_next() { return _base::_node_next; }
  inline const iparray_type &row_next() const { return _base::_node_next; }
  inline iarray_type &       row_counts() { return _base::_node_counts; }
  inline const iarray_type & row_counts() const { return _base::_node_counts; }

  /// \brief begin to assemble columns
  inline void begin_assemble_cols() {
    ccs_type::begin_assemble_cols();
    _base::_begin_assemble_nodes(nrows());
  }

  /// \brief push back a column with row indices
  /// \tparam Iter iterator type
  /// \tparam ValueArray dense value array
  /// \param[in] col current column index (C-based)
  /// \param[in] first starting iterator
  /// \param[in] last pass-of-end iterator
  /// \param[in] v dense value array, the value can be queried from indices
  template <class Iter, class ValueArray>
  inline void push_back_col(const size_type col, Iter first, Iter last,
                            const ValueArray &v) {
    ccs_type::template push_back_col<Iter, ValueArray>(col, first, last, v);
    _base::template _push_back_nodes<Iter>(col, first, last);
  }

  /// \brief push back a column with two lists of row indices
  /// \tparam Iter1 iterator type
  /// \tparam ValueArray1 dense value array
  /// \tparam Iter2 iterator type
  /// \tparam ValueArray2 dense value array
  /// \param[in] col current column index (c-based)
  /// \param[in] first1 starting iterator for list 1
  /// \param[in] last1 pass-of-end iterator for list 1
  /// \param[in] v1 dense value array for list 1
  /// \param[in] first2 starting iterator for list 2
  /// \param[in] last2 pass-of-end iterator for list 2
  /// \param[in] v2 dense value array for list 2
  template <class Iter1, class ValueArray1, class Iter2, class ValueArray2>
  inline void push_back_col(const size_type col, Iter1 first1, Iter1 last1,
                            const ValueArray1 &v1, Iter2 first2, Iter2 last2,
                            const ValueArray2 &v2) {
    ccs_type::push_back_col(col, first1, last1, v1, first2, last2, v2);
    _base::template _push_back_nodes<Iter1>(col, first1, last1);
    _base::template _push_back_nodes<Iter2>(col, first2, last2);
  }
};

}  // namespace hif

#endif  // _HIF_DS_AUGMENTEDSTORAGE_HPP
