//@HEADER
//----------------------------------------------------------------------------
//                Copyright (C) 2019 The PSMILU AUTHORS
//----------------------------------------------------------------------------
//@HEADER

/// \file psmilu_SparseVec.hpp
/// \brief Array used in factorization for work space
/// \authors Qiao,

#ifndef _PSMILU_SPARSEVEC_HPP
#define _PSMILU_SPARSEVEC_HPP

#include <algorithm>
#include <vector>

#include "psmilu_log.hpp"
#include "psmilu_utils.hpp"

namespace psmilu {

/// \class IndexValueArray
/// \tparam ValueType value type, e.g. \a double, \a float, etc
/// \tparam IndexType index type, e.g. \a int
/// \tparam OneBased if \a true, then the array is assumed to be Fortran index
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
template <class ValueType, class IndexType, bool OneBased = false>
class IndexValueArray {
 public:
  typedef ValueType                      value_type;   ///< value type
  typedef IndexType                      index_type;   ///< index type
  typedef std::vector<value_type>        array_type;   ///< value array
  typedef std::vector<index_type>        iarray_type;  ///< index array
  typedef typename array_type::size_type size_type;    ///< size
  typedef IndexValueArray                this_type;    ///< handy type wrapper
  constexpr static bool ONE_BASED = OneBased;          ///< C index flag

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

  /// \brief push back an index
  /// \param[in] i index
  inline void push_back(const size_type i) {
    psmilu_assert((to_c_idx<size_type, OneBased>(i)) < _vals.size(),
                  "%zd exceeds the value array size", i);
    psmilu_assert(_inds.size(), "empty array, did you call resize?");
    _inds[_counts++] = i;
  }

  /// \brief get the index
  /// \param[in] i local index in range of _counts (C-based)
  inline size_type idx(const size_type i) const {
    psmilu_assert(i < _counts, "%zd exceeds index array bound %zd", i, _counts);
    return _inds[i];
  }

  /// \brief get the \b C index
  /// \param[in] i local index in range of _counts (C-based)
  inline size_type c_idx(const size_type i) const {
    return to_c_idx<size_type, OneBased>(idx(i));
  }

  /// \brief get the value
  /// \param[in] i local index in range of _counts (C-based)
  inline value_type val(const size_type i) const { return _vals[c_idx(i)]; }

  /// \brief operator access
  /// \param[in] idx idx in range of dense size, one-based aware
  inline value_type &operator[](const size_type idx) {
    psmilu_assert((to_c_idx<size_type, OneBased>(idx)) < _vals.size(),
                  "%zd exceeds value size bound", idx - OneBased);
    return _vals[to_c_idx<size_type, OneBased>(idx)];
  }

  /// \brief operator access, constant version
  /// \param[in] idx idx in range of dense size, one-based aware
  inline const value_type &operator[](const size_type idx) const {
    psmilu_assert((to_c_idx<size_type, OneBased>(idx)) < _vals.size(),
                  "%zd exceeds value size bound", idx - OneBased);
    return _vals[to_c_idx<size_type, OneBased>(idx)];
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
/// \tparam OneBased if \a false (default), then assume C index
/// \ingroup ds
///
/// This class is mainly used in Crout update. The total memory cost is linear
/// with respect to the matrix system size and computation cost is bounded
/// by \f$\mathcal{O}(\textrm{lnnz}\log \textrm{lnnz}\f$, where
/// \f$\textrm{lnnz}\f$ is the local number of nonzeros.
template <class ValueType, class IndexType, bool OneBased = false>
class SparseVector : public IndexValueArray<ValueType, IndexType, OneBased> {
  typedef IndexValueArray<ValueType, IndexType, OneBased> _base;

 public:
  typedef typename _base::value_type  value_type;   ///< value
  typedef typename _base::index_type  index_type;   ///< index
  typedef typename _base::size_type   size_type;    ///< size
  typedef typename _base::array_type  array_type;   ///< value array
  typedef typename _base::iarray_type iarray_type;  ///< index array
  typedef SparseVector                this_type;    ///< handy type wrapper

 private:
  constexpr static index_type _EMPTY = static_cast<index_type>(-1);

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
    psmilu_assert(i < _counts, "%zd exceeds size bound %zd", i, _counts);
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
    const size_type j = to_c_idx<size_type, OneBased>(i);
    psmilu_assert(j < _dense_tags.size(), "%zd exceeds the dense size", j);
    if (_dense_tags[j] != static_cast<index_type>(step)) {
      _base::push_back(i);
      _dense_tags[j] = step;
      return true;  // got a new value
    }
    return false;  // not a new value
  }

 protected:
  using _base::_counts;            ///< bring in base counts
  using _base::_inds;              ///< bring in base value array
  using _base::_vals;              ///< bring in base index array
  iarray_type       _dense_tags;   ///< dense tag for union
  std::vector<bool> _sparse_tags;  ///< sparse binary tag for drop
};

}  // namespace psmilu

#endif  // _PSMILU_SPARSEVEC_HPP
