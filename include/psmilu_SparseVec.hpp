//@HEADER
//----------------------------------------------------------------------------
//                Copyright (C) 2019 The PSMILU AUTHORS
//----------------------------------------------------------------------------
//@HEADER

/// \file psmilu_WorkArray.hpp
/// \brief Array used in factorization for work space
/// \authors Qiao,

#ifndef _PSMILU_WORKARRAY_HPP
#define _PSMILU_WORKARRAY_HPP

#include <algorithm>
#include <vector>

#include "psmilu_log.hpp"
#include "psmilu_utils.hpp"

namespace psmilu {

/// \class SparseVector
/// \tparam ValueType value type, e.g. \a double
/// \tparam IndexType index type, e.g. \a int
/// \tparam OneBased if \a true, then Fortran indexing is assumed
///
/// This class is mainly used in Crout update. The total memory cost is linear
/// with respect to the matrix system size and computation cost is bounded
/// by \f$\mathcal{O}(\textrm{lnnz}\log \textrm{lnnz}\f$, where
/// \f$\textrm{lnnz}\f$ is the local number of nonzeros.
template <class ValueType, class IndexType, bool OneBased>
class SparseVector {
 public:
  typedef ValueType                                   value_type;  ///< value
  typedef IndexType                                   index_type;  ///< index
  typedef typename std::vector<value_type>::size_type size_type;   ///< size

 private:
  constexpr static size_type  _ZERO  = static_cast<size_type>(0);
  constexpr static index_type _EMPTY = static_cast<index_type>(-1);

 public:
  /// \brief default constructor
  SparseVector() = default;

  /// \brief constructor with dense and sparse sizes
  /// \param[in] dense_n dense size
  /// \param[in] sparse_n sparse index size
  SparseVector(const size_type dense_n, const size_type sparse_n)
      : _vals(dense_n, 0),
        _dense_tags(dense_n, _EMPTY),
        _inds(sparse_n),
        _sparse_tags(sparse_n),
        _counts(0u) {}

  /// \brief resize the buffer
  inline void resize(const size_type dense_n, const size_type sparse_n) {
    _vals.resize(dense_n, 0);
    _dense_tags.resize(dense_n, _EMPTY);
    _inds.resize(sparse_n);
    _sparse_tags.resize(sparse_n);
  }

  // interfaces to the underlying arrays and getting actual size
  inline size_type                      nnz() const { return _counts; }
  inline std::vector<value_type> &      vals() { return _vals; }
  inline const std::vector<value_type> &vals() const { return _vals; }
  inline std::vector<index_type> &      inds() { return _inds; }
  inline const std::vector<index_type> &inds() const { return _inds; }

  /// \brief sort the indices
  /// \warning This function must be called after compress_indices
  inline void sort_indices() {
    std::sort(_inds.begin(), _inds.begin() + _counts);
  }

  /// \brief reset the sparse tags
  /// \note This is used in dropping
  inline void reset_sparse_tags() {
    std::fill_n(_sparse_tags.begin(), _counts, false);
  }

  /// \brief reset the size counter to zero
  inline void reset() { _counts = _ZERO; }

  /// \brief mark an index to be dropped node
  /// \param[in] i i-th entry in _inds (C-based)
  inline void mark_delete(const size_type i) {
    psmilu_assert(i < _counts, "%zd exceeds size bound %zd", i, _counts);
    _sparse_tags[i] = true;
  }

  /// \brief compress indices
  inline void compress_indices() {
    size_type i = _ZERO;
    for (auto j = i; j < _counts; ++j)
      if (_sparse_tags[j]) _inds[i++] = _inds[j];
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
    if (_dense_tags[j] != step) {
      _inds[_counts] = i;
      ++_counts;
      _dense_tags[j] = step;
      return true;
    }
    return false;
  }

 private:
  std::vector<value_type> _vals;         ///< dense value array
  std::vector<index_type> _dense_tags;   ///< dense tag for union
  std::vector<index_type> _inds;         ///< sparse indices
  std::vector<bool>       _sparse_tags;  ///< sparse binary tag for drop
  size_type               _counts;       ///< size count of sparse array
};
}  // namespace psmilu

#endif  // _PSMILU_WORKARRAY_HPP
