///////////////////////////////////////////////////////////////////////////////
//                  This file is part of the HIFIR library                   //
///////////////////////////////////////////////////////////////////////////////

/*!
 * \file hif/ds/DenseMatrix.hpp
 * \brief Dense storage for the last level in multilevel ILU
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

#ifndef _HIF_DS_DENSEMATRIX_HPP
#define _HIF_DS_DENSEMATRIX_HPP

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <type_traits>

#include "hif/ds/Array.hpp"
#include "hif/utils/common.hpp"

namespace hif {
namespace internal {

/// \class StrideIterator
/// \tparam T value type
/// \brief Row-wise iterator
/// \warning This is not efficient, just for the sake of completeness
/// \ingroup ds
template <typename T>
class StrideIterator {
 public:
  // NOTE that we cannot use random iterator
  // thus, using this iterator in some STL routines may not be efficient

  /// \name STL iterator requirements
  ///@{

  typedef T*                              pointer;            ///< pointer type
  typedef std::bidirectional_iterator_tag iterator_category;  ///< iter tag
  typedef T                               value_type;         ///< value
  typedef T&                              reference;          ///< reference
  typedef std::ptrdiff_t                  difference_type;    ///< difference

  ///@}

  /// \brief default constructor
  StrideIterator() : _ptr(nullptr), _stride(0) {}

  /// \brief constructor with pointer
  /// \param[in] ptr data pointer
  /// \param[in] stride jump size (optional)
  explicit StrideIterator(pointer ptr, const difference_type stride = 1)
      : _ptr(ptr), _stride(stride) {}

  /// \brief constructor with another iterator
  /// \tparam V another value type
  /// \param[in] other another iterator
  template <typename V>
  explicit StrideIterator(const StrideIterator<V>& other)
      : _ptr(&(*other)), _stride(other.stride()) {}

  StrideIterator(const StrideIterator&) = default;
  StrideIterator& operator=(const StrideIterator&) = default;

  /// \brief assign to another iterator
  /// \tparam V another value type
  /// \param[in] rhs right-hand side iterator
  template <typename V>
  StrideIterator& operator=(const StrideIterator<V>& rhs) {
    _ptr    = &(*rhs);
    _stride = rhs.stride();
    return *this;
  }

  // iterator interfaces
  inline pointer         operator->() const { return _ptr; }
  inline reference       operator*() const { return *_ptr; }
  inline StrideIterator& operator++() {
    _ptr += _stride;
    return *this;
  }
  inline StrideIterator& operator--() {
    _ptr -= _stride;
    return *this;
  }
  inline StrideIterator operator++(int) {
    StrideIterator tmp(*this);
    ++(*this);
    return tmp;
  }
  inline StrideIterator operator--(int) {
    StrideIterator tmp(*this);
    --(*this);
    return tmp;
  }

  // mostly for internal use
  inline difference_type stride() const { return _stride; }

  /// \brief get reference
  /// \param[in] i i-th index
  /// \note stride is applied here
  inline reference operator[](const difference_type i) const {
    return _ptr[i * _stride];
  }

  /// \brief compare equal
  /// \param[in] rhs right-hand side iterator
  inline bool operator==(const StrideIterator& rhs) const {
    return _ptr == rhs._ptr;
  }

  /// \brief compare not equal
  /// \param[in] rhs right-hand side iterator
  inline bool operator!=(const StrideIterator& rhs) const {
    return !(*this == rhs);
  }

  /// \brief compare equal with a foreign iterator
  /// \tparam V another value type
  /// \param rhs right-hand side iterator
  template <typename V>
  inline bool operator==(const StrideIterator<V>& rhs) const {
    return _ptr == &(*rhs);
  }

  /// \brief compare not equal with a foreign iterator
  /// \tparam V another value type
  /// \param rhs right-hand side iterator
  template <typename V>
  inline bool operator!=(const StrideIterator<V>& rhs) const {
    return !(*this == rhs);
  }

 private:
  pointer         _ptr;     ///< pointer
  difference_type _stride;  ///< stride size
};

}  // namespace internal

/// \class DenseMatrix
/// \brief Dense storage
/// \tparam ValueType scalar value type, e.g. \a double, \a float, etc
/// \ingroup ds
///
/// To be easily compatible with \b LAPACK, we choose to use column major
/// orientation, a.k.a. Fortran/MATLAB looping orientation.
template <class ValueType>
class DenseMatrix {
 public:
  typedef Array<ValueType>                     array_type;     ///< array
  typedef typename array_type::value_type      value_type;     ///< value
  typedef typename array_type::pointer         pointer;        ///< pointer
  typedef typename array_type::reference       reference;      ///< reference
  typedef typename array_type::size_type       size_type;      ///< size
  typedef typename array_type::const_pointer   const_pointer;  ///< constant ptr
  typedef typename array_type::const_reference const_reference;
  ///< const reference
  typedef typename array_type::iterator col_iterator;
  ///< column iterator
  typedef typename array_type::const_iterator const_col_iterator;
  ///< constant column iterator
  typedef internal::StrideIterator<value_type> row_iterator;  ///< row interator
  typedef internal::StrideIterator<const value_type> const_row_iterator;
  ///< constant row iterator
  typedef DenseMatrix this_type;  ///< handy type wrapper

  /// \brief initialize dense matrix from compressed storage
  /// \tparam Cs compressed storage type, e.g. CRS, AugCCS, etc
  /// \param[in] cs compressed storage
  /// \return a dense matrix that, mathmatically, equiv to the sparse matrix
  template <class Cs>
  inline static DenseMatrix from_sparse(const Cs& cs) {
    DenseMatrix mat;
    mat.copy_sparse(cs);
    return mat;
  }

  /// \brief default constructor
  DenseMatrix() : _nrows(0u), _ncols(0u), _data() {}

  /// \brief constructor for own data
  /// \param[in] n1 number of rows
  /// \param[in] n2 number of columns, if == 0, then a square matrix is created
  explicit DenseMatrix(const size_type n1, const size_type n2 = 0u)
      : _nrows(n1), _ncols(n2 ? n2 : n1), _data(_nrows * _ncols) {
    if (_data.status() == DATA_UNDEF) _nrows = _ncols = 0u;
  }

  /// \brief constructor for own data with init value
  /// \param[in] n1 number of rows
  /// \param[in] n2 number of columns
  /// \param[in] v init value
  DenseMatrix(const size_type n1, const size_type n2, const value_type v)
      : _nrows(n1), _ncols(n2), _data(n1 * n2, v) {
    if (_data.status() == DATA_UNDEF) _nrows = _ncols = 0u;
  }

  /// \brief copy foreign type
  /// \tparam V another value type
  /// \param[in] other another matrix
  template <class V>
  explicit DenseMatrix(const DenseMatrix<V>& other)
      : _nrows(other.nrows()), _ncols(other.ncols()), _data(other.array()) {}

  /// \brief constructor for external data
  /// \param[in] n1 number of rows
  /// \param[in] n2 number of columns
  /// \param[in] data external data
  /// \param[in] wrap if \a false (default), deepcopy is performed
  DenseMatrix(const size_type n1, const size_type n2, pointer data,
              bool wrap = false)
      : _nrows(n1), _ncols(n2), _data(n1 * n2, data, wrap) {}

  /// \brief shallow or deepcopy constructor
  /// \param[in] other another matrix
  /// \param[in] clone if \a false (default), shallow copy is used
  DenseMatrix(const this_type& other, bool clone = false)
      : _nrows(other._nrows), _ncols(other._ncols), _data(other._data, clone) {}

  // default stuffs
  this_type& operator=(const this_type&) = default;
  DenseMatrix(this_type&&)               = default;
  this_type& operator=(this_type&&) = default;

  // utilities
  inline unsigned char     status() const { return _data.status(); }
  inline size_type         size() const { return _data.size(); }
  inline size_type         nrows() const { return _nrows; }
  inline size_type         ncols() const { return _ncols; }
  inline array_type&       array() { return _data; }
  inline const array_type& array() const { return _data; }
  inline pointer           data() { return _data.data(); }
  inline const_pointer     data() const { return _data.data(); }
  inline bool              empty() const { return _data.empty(); }
  inline bool              is_squared() const { return _nrows == _ncols; }

  /// \brief resize a matrix
  /// \param[in] n1 number of rows
  /// \param[in] n2 number of columns
  /// \warning The value positions may be changed!
  inline void resize(const size_type n1, const size_type n2) {
    if (n1 == _nrows && n2 == _ncols) return;
    _nrows = n1;
    _ncols = n2;
    _data.resize(n1 * n2);
  }

  // accessing

  /// \brief 1D accessing
  /// \param[in] i i-th index in _data
  inline reference operator[](const size_type i) { return _data[i]; }

  /// \brief 1D accessing, constant version
  /// \param[in] i i-th index in _data
  inline const_reference operator[](const size_type i) const {
    return _data[i];
  }

  /// \brief 2D accessing
  /// \param[in] i i-th row
  /// \param[in] j j-th column
  /// \note Be aware \a i should go faster than \a j does
  inline reference operator()(const size_type i, const size_type j) {
    hif_assert(i < _nrows, "%zd exceeds row bound %zd", i, _nrows);
    hif_assert(j < _ncols, "%zd exceeds column bound %zd", j, _ncols);
    return _data[i + j * _nrows];
  }

  /// \brief 2D accessing, constant version
  /// \param[in] i i-th row
  /// \param[in] j j-th column
  /// \note Be aware \a i should go faster than \a j does
  inline const_reference operator()(const size_type i,
                                    const size_type j) const {
    hif_assert(i < _nrows, "%zd exceeds row bound %zd", i, _nrows);
    hif_assert(j < _ncols, "%zd exceeds column bound %zd", j, _ncols);
    return _data[i + j * _nrows];
  }

  // column-wise iterator, most efficient accessing method

  inline col_iterator col_begin(const size_type col) {
    hif_assert(col < _ncols, "%zd exceeds column bound %zd", col, _ncols);
    return _data.begin() + col * _nrows;
  }
  inline col_iterator col_end(const size_type col) {
    hif_assert(col < _ncols, "%zd exceeds column bound %zd", col, _ncols);
    return _data.begin() + (col + 1) * _nrows;
  }
  inline const_col_iterator col_begin(const size_type col) const {
    hif_assert(col < _ncols, "%zd exceeds column bound %zd", col, _ncols);
    return _data.cbegin() + col * _nrows;
  }
  inline const_col_iterator col_end(const size_type col) const {
    hif_assert(col < _ncols, "%zd exceeds column bound %zd", col, _ncols);
    return _data.cbegin() + (col + 1) * _nrows;
  }
  inline const_col_iterator col_cbegin(const size_type col) const {
    return col_begin(col);
  }
  inline const_col_iterator col_cend(const size_type col) const {
    return col_end(col);
  }

  // row-wise iterator, not efficient!

  inline row_iterator row_begin(const size_type row) {
    hif_assert(row < _nrows, "%zd exceeds row bound %zd", row, _nrows);
    return row_iterator(data() + row, _nrows);
  }
  inline row_iterator row_end(const size_type row) {
    hif_assert(row < _nrows, "%zd exceeds row bound %zd", row, _nrows);
    return row_iterator(_data.end() + row, _nrows);
  }
  inline const_row_iterator row_begin(const size_type row) const {
    hif_assert(row < _nrows, "%zd exceeds row bound %zd", row, _nrows);
    return const_row_iterator(data() + row, _nrows);
  }
  inline const_row_iterator row_end(const size_type row) const {
    hif_assert(row < _nrows, "%zd exceeds row bound %zd", row, _nrows);
    return const_row_iterator(_data.cend() + row, _nrows);
  }
  inline const_row_iterator row_cbegin(const size_type row) const {
    return row_begin(row);
  }
  inline const_row_iterator row_cend(const size_type row) const {
    return row_end(row);
  }

  /// \brief convert CRS to dense storage
  /// \tparam Cs compressed storage
  /// \param[in] crs CRS matrix
  /// \note SFINAE for CCS case
  template <class Cs>
  inline typename std::enable_if<Cs::ROW_MAJOR>::type copy_sparse(
      const Cs& crs) {
#ifdef HIF_DEBUG
    if (crs.status() == DATA_UNDEF && (crs.nrows() || crs.ncols()))
      hif_warning("input CRS ia a all-zero matrix.");
#endif
    const size_type nrows(crs.nrows());
    resize(nrows, crs.ncols());
    // first step, set all values to zero
    std::fill(_data.begin(), _data.end(), value_type());
    // not efficient for CRS to dense
    for (size_type r = 0u; r < nrows; ++r) {
      auto i_itr = crs.col_ind_cbegin(r), i_end = crs.col_ind_cend(r);
      for (auto v_itr = crs.val_cbegin(r); i_itr != i_end; ++i_itr, ++v_itr)
        (*this)(r, *i_itr) = *v_itr;
    }
  }

  /// \brief convert CCS to dense storage
  /// \tparam Cs compressed storage
  /// \param[in] ccs CCS matrix
  /// \note SFINAE for CRS case
  template <class Cs>
  inline typename std::enable_if<!Cs::ROW_MAJOR>::type copy_sparse(
      const Cs& ccs) {
#ifdef HIF_DEBUG
    if (ccs.status() == DATA_UNDEF && (ccs.nrows() || ccs.ncols()))
      hif_warning("input CCS ia a all-zero matrix.");
#endif
    const size_type ncols(ccs.ncols());
    resize(ccs.nrows(), ncols);
    // first step, set all values to zero
    std::fill(_data.begin(), _data.end(), value_type());
    // efficient for CCS to dense
    for (size_type c = 0u; c < ncols; ++c) {
      auto i_itr = ccs.row_ind_cbegin(c), i_end = ccs.row_ind_cend(c);
      const size_type ld = c * _nrows;
      for (auto v_itr = ccs.val_cbegin(c); i_itr != i_end; ++i_itr, ++v_itr)
        _data[ld + *i_itr] = *v_itr;
    }
  }

 protected:
  size_type  _nrows;  ///< number of rows
  size_type  _ncols;  ///< number of columns
  array_type _data;   ///< data attribute
};

}  // namespace hif

#endif  // _HIF_DS_DENSEMATRIX_HPP
