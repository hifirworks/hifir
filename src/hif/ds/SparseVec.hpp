///////////////////////////////////////////////////////////////////////////////
//                  This file is part of HIF project                         //
///////////////////////////////////////////////////////////////////////////////

/*!
 * \file hif/ds/SparseVec.hpp
 * \brief Array used in factorization for work space
 * \author Qiao Chen

\verbatim
Copyright (C) 2019 NumGeom Group at Stony Brook University

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
\endverbatim

 */

#ifndef _HIF_DS_SPARSEVEC_HPP
#define _HIF_DS_SPARSEVEC_HPP

#include <algorithm>
#include <numeric>
#include <vector>

#include "hif/utils/common.hpp"
#include "hif/utils/log.hpp"

namespace hif {

/// \class IndexValueArray
/// \tparam ValueType value type, e.g. \a double, \a float, etc
/// \tparam IndexType index type, e.g. \a int
/// \ingroup ds
/// \note We use \a std::vector for internal data management
///
/// This data structure stores two essential arrays: 1) a dense array of
/// application "values" and 2) a sparse array of the activated indices
/// in the value array. By dense, we mean an array that supports random
/// access; by sparse, we mean an array that only contains what we need.
/// Notice that this data structure is mainly used in sparse matrix
/// operations for efficiency, where we just need to manipulate the indices
/// without worry about touching the values. Be aware the it's may be
/// necessary that the sparse array have the same length (in terms of memory)
/// as that of the dense array due to the fact that we cannot estimate the
/// sparse size precisely.
///
/// For the efficiency purpose, we explicitly create an size counter,
/// \ref _counts, to avoid calling the \a std::vector::push_back, which is
/// quite expansive.
template <class ValueType, class IndexType>
class IndexValueArray {
 public:
  typedef ValueType                      value_type;   ///< value type
  typedef IndexType                      index_type;   ///< index type
  typedef std::vector<value_type>        array_type;   ///< value array
  typedef std::vector<index_type>        iarray_type;  ///< index array
  typedef typename array_type::size_type size_type;    ///< size
  typedef IndexValueArray                this_type;    ///< handy type wrapper

  /// \brief default constructor
  IndexValueArray() : _vals(), _inds(), _counts(0u) {}

  /// \brief constructor with dense and (optionally) sparse sizes
  /// \param[in] d_n dense size
  /// \param[in] s_n sparse size, if == 0, then use \a d_n
  explicit IndexValueArray(const size_type d_n, const size_type s_n = 0u)
      : _vals(d_n), _inds(s_n ? s_n : d_n), _counts(0u) {}

  // allow moving methods
  IndexValueArray(this_type &&) = default;
  this_type &operator=(this_type &&) = default;

  // copying methods are banned
  IndexValueArray(const this_type &) = delete;
  this_type &operator=(const this_type &) = delete;

  /// \brief resize the buffer
  /// \param[in] d_n dense size
  /// \param[in] s_n sparse size, if == 0 (default), then use \a d_n
  inline void resize(const size_type d_n, const size_type s_n = 0u) {
    _vals.resize(d_n);
    _inds.resize(s_n ? s_n : d_n);
  }

  /// \brief get the number of counts
  inline size_type size() const { return _counts; }

  /// \brief reset counter
  inline void reset_counter() { _counts = 0u; }

  /// \brief check if empty or not
  inline bool empty() const { return _counts == 0u; }

  /// \brief sort the indices
  /// \note Complexity: \f$\mathcal{O}(n\log n)\f$, \f$n\f$ is \ref _counts
  inline void sort_indices() {
    std::sort(_inds.begin(), _inds.begin() + _counts);
  }

  /// \brief sort the indices given a unary operator
  /// \tparam UnaryOp unary operator type
  template <class UnaryOp>
  inline void sort_indices(const UnaryOp &op) {
    std::sort(_inds.begin(), _inds.begin() + _counts, op);
  }

  /// \brief push back an index
  /// \param[in] i index
  inline void push_back(const size_type i) {
    hif_assert(i < _vals.size(), "%zd exceeds the value array size", i);
    hif_assert(_inds.size(), "empty array, did you call resize?");
    _inds[_counts++] = i;
  }

  /// \brief get the index
  /// \param[in] i local index in range of _counts
  inline size_type idx(const size_type i) const {
    hif_assert(i < _counts, "%zd exceeds index array bound %zd", i, _counts);
    return _inds[i];
  }

  /// \brief get the value
  /// \param[in] i local index in range of _counts
  inline value_type val(const size_type i) const { return _vals[idx(i)]; }

  /// \brief get the 1 norm of the vector
  inline value_type norm1() const {
    value_type tmp(0);
    for (size_type i(0); i < _counts; ++i) tmp += std::abs(val(i));
    return tmp;
  }

  /// \brief get the 2 norm squared of the vector
  inline value_type norm2_sq() const {
    value_type tmp(0);
    for (size_type i(0); i < _counts; ++i) tmp += val(i) * val(i);
    return tmp;
  }

  /// \brief get the 2 norm of the vector
  inline value_type norm2() const { return std::sqrt(norm2_sq()); }

  /// \brief get the maximum magnitude value, only works wit real number
  inline typename ValueTypeTrait<value_type>::value_type max_mag() const {
    using v_t = typename ValueTypeTrait<value_type>::value_type;
    v_t v(0);
    for (size_type i(0); i < _counts; ++i) v = std::max(v, std::abs(val(i)));
    return v;
  }

  /// \brief get the index corresponding maximum magnitude
  inline size_type imax_mag() const {
    if (!_counts) return std::numeric_limits<size_type>::max();
    return *std::max_element(_inds.cbegin(), _inds.cbegin() + _counts,
                             [&](const index_type i, const index_type j) {
                               return std::abs(_vals[i]) < std::abs(_vals[j]);
                             });
  }

  /// \brief get the index corresponding maximum magnitude with index bound
  template <class UnaryOp>
  inline size_type imax_mag(const UnaryOp &op) const {
    // if empty, return max value of size_t
    if (!_counts) return std::numeric_limits<size_type>::max();
    // find a starting position, where it is within the bound, i.e., typically
    // the size of leading block
    auto pos = std::find_if(_inds.cbegin(), _inds.cbegin() + _counts, op);
    // if all tail region, return max value of size_t
    if (pos == _inds.cbegin() + _counts)
      return std::numeric_limits<size_type>::max();
    size_type j(*pos);
    for (size_type i(0); i < _counts; ++i) {
      if (!op(_inds[i])) continue;
      if (std::abs(_vals[j]) < std::abs(_vals[_inds[i]])) j = _inds[i];
    }
    return j;
  }

  /// \brief find the first entry given a unary operator
  template <class UnaryOp>
  inline size_type find_if(const UnaryOp &op) const {
    if (!_counts) return std::numeric_limits<size_type>::max();
    auto pos = std::find_if(_inds.cbegin(), _inds.cbegin() + _counts, op);
    // if all tail region, return max value of size_t
    if (pos == _inds.cbegin() + _counts)
      return std::numeric_limits<size_type>::max();
    return *pos;
  }

  /// \brief inplace scaling
  /// \param[in] alpha scaling factor, i.e., v=alpha*v
  inline void scale(const value_type alpha) {
    for (size_type i(0); i < _counts; ++i) _vals[_inds[i]] *= alpha;
  }

  /// \brief operator access
  /// \param[in] i index in range of dense size, one-based aware
  inline value_type &operator[](const size_type i) {
    hif_assert(i < _vals.size(), "%zd exceeds value size bound", i);
    return _vals[i];
  }

  /// \brief operator access, constant version
  /// \param[in] i index in range of dense size, one-based aware
  inline const value_type &operator[](const size_type i) const {
    hif_assert(i < _vals.size(), "%zd exceeds value size bound", i);
    return _vals[i];
  }

  // utils
  inline array_type &       vals() { return _vals; }
  inline iarray_type &      inds() { return _inds; }
  inline const array_type & vals() const { return _vals; }
  inline const iarray_type &inds() const { return _inds; }

 protected:
  array_type  _vals;    ///< values
  iarray_type _inds;    ///< indices
  size_type   _counts;  ///< current counts
};

/// \class SparseVector
/// \tparam ValueType value type, e.g. \a double
/// \tparam IndexType index type, e.g. \a int
/// \ingroup ds
///
/// This class is mainly used in Crout update. The total memory cost is linear
/// with respect to the matrix system size and computation cost is bounded
/// by \f$\mathcal{O}(\textrm{lnnz}\log \textrm{lnnz}\f$, where
/// \f$\textrm{lnnz}\f$ is the local number of nonzeros.
template <class ValueType, class IndexType>
class SparseVector : public IndexValueArray<ValueType, IndexType> {
  typedef IndexValueArray<ValueType, IndexType> _base;  ///< base

 public:
  typedef typename _base::value_type  value_type;   ///< value
  typedef typename _base::index_type  index_type;   ///< index
  typedef typename _base::size_type   size_type;    ///< size
  typedef typename _base::array_type  array_type;   ///< value array
  typedef typename _base::iarray_type iarray_type;  ///< index array
  typedef SparseVector                this_type;    ///< handy type wrapper

 private:
  constexpr static index_type _EMPTY = static_cast<index_type>(-1);
  ///< empty flag

 public:
  /// \brief default constructor
  SparseVector() = default;

  /// \brief constructor with dense and sparse sizes
  /// \param[in] d_n dense size
  /// \param[in] s_n sparse index size, if == 0 (default), then use \a d_n
  explicit SparseVector(const size_type d_n, const size_type s_n = 0u)
      : _base(d_n, s_n),
        _dense_tags(d_n, static_cast<index_type>(-1)),
        _sparse_tags(_inds.size(), false) {}

  // ban copy methods
  SparseVector(const this_type &) = delete;
  this_type &operator=(const this_type &) = delete;

  // allow move
  SparseVector(this_type &&) = default;
  this_type &operator=(this_type &&) = default;

  /// \brief resize sparse vector
  /// \param[in] d_n dense size
  /// \param[in] s_n sparse size, if == 0 (default), using \a d_n
  /// \note Overloading the base version, i.e. \ref _base::resize
  inline void resize(const size_type d_n, const size_type s_n = 0u) {
    constexpr static auto _empty = _EMPTY;
    _base::resize(d_n, s_n);
    _dense_tags.resize(d_n);
    _sparse_tags.resize(_inds.size());
    // NOTE that we enforce to reset ALL tags to default/empty stage
    std::fill(_dense_tags.begin(), _dense_tags.end(), _empty);
    std::fill(_sparse_tags.begin(), _sparse_tags.end(), false);
  }

  /// \brief mark an index to be dropped node
  /// \param[in] i i-th entry in _inds (C-based)
  inline void mark_delete(const size_type i) {
    hif_assert(i < _counts, "%zd exceeds size bound %zd", i, _counts);
    _sparse_tags[i] = true;
  }

  /// \brief compress indices
  inline void compress_indices() {
    size_type i = 0u;
    for (auto j = i; j < _counts; ++j) {
      if (!_sparse_tags[j])
        _inds[i++] = _inds[j];
      else
        _sparse_tags[j] = false;  // NOTE that the flag reset here
    }
    _counts = i;
  }

  /// \brief push back an index at a specific Crout update step
  /// \param[in] i index
  /// \param[in] step Crout update step
  /// \return if \a true, then a new value has been pushed back
  /// \note The tag array is for unsure the union of the indices
  inline bool push_back(const index_type i, const size_type step) {
    hif_assert((size_type)i < _dense_tags.size(), "%zd exceeds the dense size",
               (size_type)i);
    if (_dense_tags[i] != static_cast<index_type>(step)) {
      _base::push_back(i);
      _dense_tags[i] = step;
      return true;  // got a new value
    }
    return false;  // not a new value
  }

  /// \brief reset current state
  inline void restore_cur_state() {
    const size_type n = _counts;
    for (size_type i(0); i < n; ++i) _dense_tags[_base::idx(i)] = _EMPTY;
  }

 protected:
  using _base::_counts;            ///< bring in base counts
  using _base::_inds;              ///< bring in base value array
  using _base::_vals;              ///< bring in base index array
  iarray_type       _dense_tags;   ///< dense tag for union
  std::vector<bool> _sparse_tags;  ///< sparse binary tag for drop
};

}  // namespace hif

#endif  // _HIF_DS_SPARSEVEC_HPP
