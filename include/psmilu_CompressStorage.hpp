//@HEADER
//----------------------------------------------------------------------------
//                Copyright (C) 2019 The PSMILU AUTHORS
//----------------------------------------------------------------------------
//@HEADER

/// \file psmilu_CompressedStorage.hpp
/// \brief Core about CRS and CCS storage formats
/// \authors Qiao,

#ifndef _PSMILU_COMPRESSEDSTORAGE_HPP
#define _PSMILU_COMPRESSEDSTORAGE_HPP

#include <algorithm>
#include <utility>

#include "psmilu_Array.hpp"
#include "psmilu_utils.hpp"

namespace psmilu {
namespace internal {

/// \class CompressedStorage
/// \brief Core of the compressed storage, including data and interfaces
/// \tparam ValueType value type, e.g. \a double, \a float, etc
/// \tparam IndexType index used, e.g. \a int, \a long, etc
/// \tparam OneBased if \a true, then Fortran based index is assumed
template <class ValueType, class IndexType, bool OneBased>
class CompressedStorage {
 public:
  typedef ValueType                            value_type;   ///< value type
  typedef value_type *                         pointer;      ///< pointer type
  typedef IndexType                            index_type;   ///< index type
  typedef index_type *                         ipointer;     ///< index pointer
  typedef CompressedStorage                    this_type;    ///< this
  typedef Array<ValueType>                     array_type;   ///< value array
  typedef Array<IndexType>                     iarray_type;  ///< index array
  typedef typename array_type::size_type       size_type;    ///< size type
  typedef typename array_type::iterator        v_iterator;   ///< value iterator
  typedef typename array_type::const_iterator  const_v_iterator;  ///< constant
  typedef typename iarray_type::iterator       i_iterator;  ///< index iterator
  typedef typename iarray_type::const_iterator const_i_iterator;  ///< constant

 private:
  constexpr static size_type _ZERO = static_cast<size_type>(0);
  constexpr static size_type _ONE  = _ZERO + 1;

 public:
  /// \brief default constructor
  CompressedStorage() = default;

  /// \brief constructor with external data, either copy (default) or wrap
  /// \param[in] n primary size
  /// \param[in] ind_start index starting positions
  /// \param[in] indices index array
  /// \param[in] vals value array
  /// \param[in] wrap if \a false (default), then do copy
  CompressedStorage(const size_type n, ipointer ind_start, ipointer indices,
                    pointer vals, bool wrap = false)
      : _ind_start(n + 1, ind_start, wrap),
        _indices(ind_start[n] - ind_start[0], indices, wrap),
        _vals(_indices.size(), vals, wrap),
        _psize(n) {}

  /// \brief constructor for own data
  /// \param[in] n primary direction size
  /// \param[in] nnz total number of nonzeros, default is 0
  /// \param[in] reserve if \a true (default), reserve space instead of resize
  /// \sa Array::resize, Array::reserve
  CompressedStorage(const size_type n, const size_type nnz = 0u,
                    bool reserve = true)
      : _ind_start(n + 1) {
    if (nnz) {
      if (reserve) {
        _indices.reserve(nnz);
        _vals.reserve(nnz);
      } else {
        _indices.resize(nnz);
        _vals.resize(nnz);
      }
    }
    std::fill(_ind_start.begin(), _ind_start.end(),
              static_cast<index_type>(OneBased));
    _psize = n;
  }

  /// \brief shallow copy constructor (default) or deep copy
  /// \param[in] other another compressed storage
  /// \param[in] clone if \a false (default), then perform shallow copy
  CompressedStorage(const this_type &other, bool clone = false)
      : _ind_start(other._ind_start, clone),
        _indices(other._indices, clone),
        _vals(other._vals, clone),
        _psize(other._psize) {}

  // default move
  CompressedStorage(this_type &&) = default;
  this_type &operator=(this_type &&) = default;
  // default shallow assignment
  this_type &operator=(const this_type &) = default;

  /// \brief check the status of the storage
  /// \return either DATA_UNDEF, DATA_WRAP, or DATA_OWN
  /// \note The status should be consistent for all three arrays.
  inline unsigned char status() const { return _indices.status(); }
  inline size_type     nnz() const {
    return status() == DATA_UNDEF ? _ZERO
                                  : _ind_start.back() - _ind_start.front();
  }

  // getting information for local value ranges

  inline v_iterator val_begin(const size_type i) {
    return _vals.begin() + to_c_idx<size_type, OneBased>(_ind_start[i]);
  }
  inline v_iterator val_end(const size_type i) {
    return val_begin(i) + _nnz(i);
  }
  inline const_v_iterator val_begin(const size_type i) const {
    return _vals.begin() + to_c_idx<size_type, OneBased>(_ind_start[i]);
  }
  inline const_v_iterator val_end(const size_type i) const {
    return val_begin(i) + _nnz(i);
  }
  inline const_v_iterator val_cbegin(const size_type i) const {
    return _vals.begin() + to_c_idx<size_type, OneBased>(_ind_start[i]);
  }
  inline const_v_iterator val_cend(const size_type i) const {
    return val_cbegin(i) + _nnz(i);
  }

  /// \brief reserve space for nnz
  /// \param[in] nnz total number of nonzeros
  /// \note This function only reserves spaces for _indices and _vals
  inline void reserve(const size_type nnz) {
    _indices.reserve(nnz);
    _vals.reserve(nnz);
  }

 protected:
  /// \brief begin assemble rows
  /// \note PSMILU only requires pushing back operations
  /// \sa _end_assemble
  inline void _begin_assemble() {
    _ind_start.resize(1);
    _ind_start.front() = static_cast<index_type>(OneBased);
    _vals.resize(0u);
    _indices.resize(0u);
  }

  /// \brief finish assembling process
  /// \sa _begin_assemble
  inline void _end_assemble() {
    const int flag = _psize + 1u == _ind_start.size()
                         ? 0
                         : _psize + 1u < _ind_start.size() ? 1 : -1;
    if (!flag) return;
    if (flag > 0) {
      psmilu_warning("pushed more entries (%zd) than requested (%zd)",
                     _ind_start.size() - 1u, _psize);
      _psize = _ind_start.size() ? _ind_start.size() - 1u : 0u;
      return;
    }
    psmilu_warning("detected empty primary entries (%zd), filling zeros",
                   _psize + 1u - _ind_start.size());
    for (auto i = _ind_start.size() - 1ul; i < _psize; ++i)
      _push_back_primary_empty(i);
  }

  /// \brief push back an empty entry
  /// \param[in] ii ii-th entry in primary direction
  inline void _push_back_primary_empty(const size_type psmilu_debug_code(ii)) {
    psmilu_assert(ii + 1u == _ind_start.size(),
                  "inconsistent pushing back at entry %zd", ii);
    _ind_start.push_back(_ind_start.back());
  }

  /// \brief push back a \b SORTED index list to _indices
  /// \tparam Iter iterator type
  /// \tparam ValueArray dense value array
  /// \param[in] ii ii-th entry in primary direction
  /// \param[in] first starting iterator
  /// \param[in] last ending iterator
  /// \param[in] v dense value array, the value can be queried from indices
  /// \warning The indices are assumed to be sorted
  template <class Iter, class ValueArray>
  inline void _push_back_primary(const size_type psmilu_debug_code(ii),
                                 Iter first, Iter last, const ValueArray &v) {
    psmilu_assert(ii + 1u == _ind_start.size(),
                  "inconsistent pushing back at entry %zd", ii);
    // first push back the list
    psmilu_assert(_indices.size() == _vals.size(), "fatal error");
    _indices.push_back(first, last);
    const auto      nnz   = std::distance(first, last);
    const size_type start = _vals.size();
    _vals.resize(start + nnz);
    psmilu_assert(_indices.size() == _vals.size(), "fatal error");
    for (auto i = _ZERO; i < nnz; ++i, ++first) {
      const auto j = to_c_idx<size_type, OneBased>(*first);
      psmilu_assert(j < v.size(), "%zd exceeds the length of v", j);
      _vals[i + start] = v[j];
    }
    // finally, push back ind start
    _ind_start.push_back(_ind_start.back() + nnz);
  }

  /// \brief check local number of nonzeros
  /// \param[in] i i-th entry in primary direction
  inline size_type _nnz(const size_type i) const {
    return _ind_start[i + 1] - _ind_start[i];
  }

  // local index start ranges

  inline i_iterator _ind_begin(const size_type i) {
    return _indices.begin() + to_c_idx<size_type, OneBased>(_ind_start[i]);
  }
  inline i_iterator _ind_end(const size_type i) {
    return ind_begin(i) + _nnz(i);
  }
  inline const_i_iterator _ind_begin(const size_type i) const {
    return _indices.begin() + to_c_idx<size_type, OneBased>(_ind_start[i]);
  }
  inline const_i_iterator _ind_end(const size_type i) const {
    return ind_begin(i) + _nnz(i);
  }
  inline const_i_iterator _ind_cbegin(const size_type i) const {
    return _indices.begin() + to_c_idx<size_type, OneBased>(_ind_start[i]);
  }
  inline const_i_iterator _ind_cend(const size_type i) const {
    return ind_cbegin(i) + _nnz(i);
  }

 protected:
  iarray_type _ind_start;  ///< index pointer array, size of n+1
  iarray_type _indices;    ///< index array, size of nnz
  array_type  _vals;       ///< numerical data array, size of nnz
  size_type   _psize;      ///< primary size
};

/// \brief convert from one type to another type, e.g. ccs->crs
/// \tparam ValueArray numerical value array type
/// \tparam IndexArray index array type
/// \tparam OneBased Fortran of C based index
/// \param[in] i_ind_start input index start array
/// \param[in] i_indices input index array
/// \param[in] i_vals input value array
/// \param[out] o_ind_start output index start array
/// \param[out] o_indices output index array
/// \param[out] o_vals output values
/// \warning o_indices should be initialized properly with value 0 or 1
/// \warning Both o_{indices,vals} should be allocated properly
/// \note Work with both Array and \a std::vector
template <class ValueArray, class IndexArray, bool OneBased>
inline void convert_storage(const IndexArray &i_ind_start,
                            const IndexArray &i_indices,
                            const ValueArray &i_vals, IndexArray &o_ind_start,
                            IndexArray &o_indices, IndexArray &o_vals) {
  typedef typename ValueArray::size_type  size_type;
  typedef typename IndexArray::value_type index_type;
  constexpr static size_type base = static_cast<size_type>(OneBased);

  // o_ind_start is allocated and uniformly assigned to be OneBased
  // both o_{indices,vals} are uninitialized arrays with size of input nnz
  psmilu_error_if(i_indices.size() != i_ind_start.back() - base,
                  "nnz %zd does not match that in start array",
                  i_indices.size());
  psmilu_error_if(i_indices.size() != i_vals.size(),
                  "nnz sizes (%zd,%zd) do not match between indices and vals",
                  i_indices.size(), i_vals.size());
  psmilu_error_if(i_indices.size() != o_indices.size(),
                  "input nnz %zd does not match of that the output (%zd)",
                  i_indices.size(), o_indices.size());
  psmilu_error_if(o_indices.size() != o_vals.size(),
                  "nnz sizes (%zd,%zd) do not match between indices and vals",
                  o_indices.size(), o_indices.size());

  const size_type o_n = o_ind_start.size() - 1u;
  const size_type i_n = i_ind_start.size() - 1u;
  const size_type nnz = i_indices.size();

  // step 1, counts nnz per secondary direction, O(nnz)
  for (const auto &p : i_indices) {
    const auto j = to_c_idx<size_type, OneBased>(p);
    psmilu_assert(j < o_n, "%zd exceeds the bound %zd", j, o_n);
    ++o_ind_start[j + 1];
  }

  // step 2, build o_ind_start, O(n)
  for (size_type i = 0u; i < o_n; ++i)
    o_ind_start[i + 1] += o_ind_start[i] - base;
  psmilu_assert(o_ind_start.back() == i_ind_start.back(), "fatal issue");

  // step 3, build output indices and values, O(nnz)
  auto i_iiter = i_indices.cbegin();  // index iterator
  auto i_viter = i_vals.cbegin();     // value iterator
  for (size_type i = 0u; i < i_n; ++i) {
    const auto lnnz = i_ind_start[i + 1] - i_ind_start[i];
    const auto lend = i_iiter + lnnz;
    for (; i_iiter != lend; ++i_iiter, ++i_viter) {
      const auto j = to_c_idx<size_type, OneBased>(*i_iiter);
      // get position
      const auto jj = to_c_idx<size_type, OneBased>(o_ind_start[j]);
      o_indices[jj] = i + base;
      o_vals[jj]    = *i_viter;
      // increment the counter
      ++o_ind_start[j];
    }
  }
  psmilu_assert(o_ind_start[o_n] == o_ind_start[o_n - 1], "fatal issue");

  // final step, revert back to previous stage for output index start, O(n)
  index_type temp(base);
  for (size_type i = 0u; i < o_n; ++i) std::swap(temp, o_ind_start[i]);

  // NOTE we can create a buffer of size n, so that the last step will become
  // optional
}

}  // namespace internal

// forward decl
template <class ValueType, class IndexType, bool OneBased = false>
class CCS;

/// \class CRS
/// \brief Compressed Row Storage (CRS) format for sparse matrices
/// \tparam ValueType numerical value type, e.g. \a double, \a float, etc
/// \tparam IndexType index type, e.g. \a int, \a long, etc
/// \tparam OneBased if \a false (default), using C-based index
template <class ValueType, class IndexType, bool OneBased = false>
class CRS : public internal::CompressedStorage<ValueType, IndexType, OneBased> {
  typedef internal::CompressedStorage<ValueType, IndexType, OneBased> _base;

 public:
  typedef ValueType                           value_type;   ///< value type
  typedef IndexType                           index_type;   ///< index type
  typedef typename _base::array_type          array_type;   ///< value array
  typedef typename _base::iarray_type         iarray_type;  ///< index array
  typedef typename _base::size_type           size_type;    ///< size type
  typedef typename _base::pointer             pointer;      ///< value pointer
  typedef typename _base::ipointer            ipointer;     ///< index pointer
  typedef CCS<ValueType, IndexType, OneBased> other_type;   ///< ccs type
  typedef typename _base::i_iterator          i_iterator;   ///< index iterator
  typedef typename _base::const_i_iterator    const_i_iterator;
  typedef typename _base::v_iterator          v_iterator;  ///< value iterator
  typedef typename _base::const_v_iterator    const_v_iterator;
  constexpr static bool ONE_BASED = OneBased;  ///< C or Fortran based

  /// \brief default constructor
  CRS() : _base(), _ncols(0u) {}

  /// \brief external data for matrix
  /// \param[in] nrows number of rows
  /// \param[in] ncols number of columns
  /// \param[in] row_start row index start array
  /// \param[in] col_ind column indices
  /// \param[in] vals value data
  /// \param[in] wrap if \a false (default), then do copy
  CRS(const size_type nrows, const size_type ncols, ipointer row_start,
      ipointer col_ind, pointer vals, bool wrap = false)
      : _base(nrows, row_start, col_ind, vals, wrap), _ncols(ncols) {}

  /// \brief external data for \b squared matrix
  /// \param[in] n number of rows and columns
  /// \param[in] row_start row index start array
  /// \param[in] col_ind column indices
  /// \param[in] vals value data
  /// \param[in] wrap if \a false (default), then do copy
  CRS(const size_type n, ipointer row_start, ipointer col_ind, pointer vals,
      bool wrap = false)
      : CRS(n, n, row_start, col_ind, vals, wrap) {}

  /// \brief own data for crs matrix
  /// \param[in] nrows number of rows
  /// \param[in] ncols number of columns
  /// \param[in] nnz total number of nonzeros, default is 0
  /// \param[in] reserve if \a true (default), then reserve space for nnz arrays
  CRS(const size_type nrows, const size_type ncols, const size_type nnz = 0u,
      bool reserve = true)
      : _base(nrows, nnz, reserve), _ncols(ncols) {}

  /// \brief shallow copy constructor (default) or make a clone
  /// \param[in] other another CRS
  /// \param[in] clone if \a false (defaut), do shallow copy
  CRS(const CRS &other, bool clone = false)
      : _base(other, clone), _ncols(other._ncols) {}

  /// \brief make a CRS based on CCS input
  /// \param[in] ccs a CCS matrix
  explicit CRS(const other_type &ccs)
      : CRS(ccs.nrows(), ccs.ncols(), ccs.nnz(), false) {
    // NOTE that the constructor will allocate memory for each of the
    // three arrays before calling the following routine
    internal::convert_storage<array_type, iarray_type, OneBased>(
        ccs.col_start(), ccs.row_ind(), ccs.vals(), row_start(), col_ind(),
        vals());
  }

  // default move
  CRS(CRS &&)  = default;
  CRS &operator=(CRS &&) = default;
  // default shallow assignment
  CRS &operator=(const CRS &) = default;

  // interface directly to internal database
  inline iarray_type &      row_start() { return _base::_ind_start; }
  inline iarray_type &      col_ind() { return _base::_indices; }
  inline array_type &       vals() { return _base::_vals; }
  inline const iarray_type &row_start() const { return _base::_ind_start; }
  inline const iarray_type &col_ind() const { return _base::_indices; }
  inline const array_type & vals() const { return _base::_vals; }

  /// \brief get number of rows
  inline size_type nrows() const { return _psize; }

  /// \brief get number of columns
  inline size_type ncols() const { return _ncols; }

  /// \brief resize rows
  /// \param[in] nrows number of rows
  inline void resize_nrows(const size_type nrows) { _psize = nrows; }

  /// \brief resize columns
  /// \param[in] ncols number of columns
  inline void resize_ncols(const size_type ncols) { _ncols = ncols; }

  /// \brief resize
  /// \param[in] nrows number of rows
  /// \param[in] ncols number of columns
  inline void resize(const size_type nrows, const size_type ncols) {
    resize_nrows(nrows);
    resize_ncols(ncols);
  }

  /// \brief get local nnz per row
  /// \param[in] i row entry (C-based index)
  /// \return number of nonzeros in row \a i
  inline size_type nnz_in_row(const size_type i) const { return _base::nnz(i); }

  // wrappers for column index local iterator ranges

  inline i_iterator col_ind_begin(const size_type i) {
    return _base::_ind_begin(i);
  }
  inline i_iterator col_ind_end(const size_type i) {
    return _base::_ind_end(i);
  }
  inline const_i_iterator col_ind_begin(const size_type i) const {
    return _base::_ind_cbegin(i);
  }
  inline const_i_iterator col_ind_end(const size_type i) const {
    return _base::_ind_cend(i);
  }
  inline const_i_iterator col_ind_cbegin(const size_type i) const {
    return _base::_ind_cbegin(i);
  }
  inline const_i_iterator col_ind_cend(const size_type i) const {
    return _base::_ind_cend(i);
  }

  /// \brief begin assemble rows
  inline void begin_assemble_rows() { _base::_begin_assemble(); }

  /// \brief finish assemble rows
  inline void end_assemble_rows() { _base::_end_assemble(); }

  /// \brief push back an empty row
  /// \param[in] row current row entry
  inline void push_back_empty_row(const size_type psmilu_debug_code(row)) {
    _base::_push_back_primary_empty(row);
  }

  /// \brief push back a new row
  /// \tparam Iter iterator type
  /// \tparam ValueArray dense value array
  /// \param[in] row current row (C-based index)
  /// \param[in] first starting iterator
  /// \param[in] last ending iterator
  /// \param[in] v dense value array, the value can be queried from indices
  /// \warning The indices are assumed to be sorted
  template <class Iter, class ValueArray>
  inline void push_back_row(const size_type psmilu_debug_code(row), Iter first,
                            Iter last, const ValueArray &v) {
    _base::_push_back_primary(row, first, last, v);
  }

 protected:
  size_type _ncols;     ///< number of columns
  using _base::_psize;  ///< number of rows (primary entries)
};

template <class ValueType, class IndexType, bool OneBased>
class CCS : public internal::CompressedStorage<ValueType, IndexType, OneBased> {
  typedef internal::CompressedStorage<ValueType, IndexType, OneBased> _base;

 public:
  typedef ValueType                           value_type;   ///< value type
  typedef IndexType                           index_type;   ///< index type
  typedef typename _base::array_type          array_type;   ///< value array
  typedef typename _base::iarray_type         iarray_type;  ///< index array
  typedef typename _base::size_type           size_type;    ///< size type
  typedef typename _base::pointer             pointer;      ///< value pointer
  typedef typename _base::ipointer            ipointer;     ///< index pointer
  typedef CRS<ValueType, IndexType, OneBased> other_type;   ///< crs type
  typedef typename _base::i_iterator          i_iterator;   ///< index iterator
  typedef typename _base::const_i_iterator    const_i_iterator;
  typedef typename _base::v_iterator          v_iterator;  ///< value iterator
  typedef typename _base::const_v_iterator    const_v_iterator;
  constexpr static bool ONE_BASED = OneBased;  ///< C or Fortran based

  /// \brief default constructor
  CCS() : _base(), _nrows(0u) {}

  /// \brief external data for matrix
  /// \param[in] nrows number of rows
  /// \param[in] ncols number of columns
  /// \param[in] col_start column index start array
  /// \param[in] row_ind row indices
  /// \param[in] vals value data
  /// \param[in] wrap if \a false (default), then do copy
  CCS(const size_type nrows, const size_type ncols, ipointer col_start,
      ipointer row_ind, pointer vals, bool wrap = false)
      : _base(ncols, col_start, row_ind, vals, wrap), _nrows(nrows) {}

  /// \brief external data for \b squared matrix
  /// \param[in] n number of rows and columns
  /// \param[in] col_start column index start array
  /// \param[in] row_ind row indices
  /// \param[in] vals value data
  /// \param[in] wrap if \a false (default), then do copy
  CCS(const size_type n, ipointer col_start, ipointer row_ind, pointer vals,
      bool wrap = false)
      : CCS(n, n, col_start, row_ind, vals, wrap) {}

  /// \brief own data for ccs matrix
  /// \param[in] nrows number of rows
  /// \param[in] ncols number of columns
  /// \param[in] nnz total number of nonzeros, default is 0
  /// \param[in] reserve if \a true (default), then reserve space for nnz arrays
  CCS(const size_type nrows, const size_type ncols, const size_type nnz = 0u,
      bool reserve = true)
      : _base(ncols, nnz, reserve), _nrows(nrows) {}

  /// \brief shallow copy constructor (default) or make a clone
  /// \param[in] other another CCS
  /// \param[in] clone if \a false (defaut), do shallow copy
  CCS(const CCS &other, bool clone = false)
      : _base(other, clone), _nrows(other._nrows) {}

  /// \brief make a CCS based on CRS input
  /// \param[in] crs a CRS matrix
  explicit CCS(const other_type &crs)
      : CCS(crs.nrows(), crs.ncols(), crs.nnz(), false) {
    internal::convert_storage<array_type, iarray_type, OneBased>(
        crs.row_start(), crs.col_ind(), crs.vals(), col_start(), row_ind(),
        vals());
  }

  // default move
  CCS(CCS &&)  = default;
  CCS &operator=(CCS &&) = default;
  // default shallow assignment
  CCS &operator=(const CCS &) = default;

  // interface directly touch to internal database
  inline iarray_type &      col_start() { return _base::_ind_start; }
  inline iarray_type &      row_ind() { return _base::_indices; }
  inline array_type &       vals() { return _base::_vals; }
  inline const iarray_type &col_start() const { return _base::_ind_start; }
  inline const iarray_type &row_ind() const { return _base::_indices; }
  inline const array_type & vals() const { return _base::_vals; }

  /// \brief get number of rows
  inline size_type nrows() const { return _nrows; }

  /// \brief get number of columns
  inline size_type ncols() const {
    return _base::status() == DATA_UNDEF ? 0u : col_start().size() - 1u;
  }

  /// \brief resize rows
  /// \param[in] nrows number of rows
  inline void resize_nrows(const size_type nrows) { _nrows = nrows; }

  /// \brief resize columns
  /// \param[in] ncols number of columns
  inline void resize_ncols(const size_type ncols) { _psize = ncols; }

  /// \brief resize
  /// \param[in] nrows number of rows
  /// \param[in] ncols number of columns
  inline void resize(const size_type nrows, const size_type ncols) {
    resize_nrows(nrows);
    resize_ncols(ncols);
  }

  /// \brief get local nnz per column
  /// \param[in] i column entry (C-based index)
  /// \return number of nonzeros in column \a i
  inline size_type nnz_in_col(const size_type i) const { return _base::nnz(i); }

  // wrappers for row index local iterator ranges

  inline i_iterator row_ind_begin(const size_type i) {
    return _base::_ind_begin(i);
  }
  inline i_iterator row_ind_end(const size_type i) {
    return _base::_ind_end(i);
  }
  inline const_i_iterator row_ind_begin(const size_type i) const {
    return _base::_ind_cbegin(i);
  }
  inline const_i_iterator row_ind_end(const size_type i) const {
    return _base::_ind_cend(i);
  }
  inline const_i_iterator row_ind_cbegin(const size_type i) const {
    return _base::_ind_cbegin(i);
  }
  inline const_i_iterator row_ind_cend(const size_type i) const {
    return _base::_ind_cend(i);
  }

  /// \brief begin assemble columns
  inline void begin_assemble_cols() { _base::_begin_assemble(); }

  /// \brief finish assemble columns
  inline void end_assemble_cols() { _base::_end_assemble(); }

  /// \brief push back an empty column
  /// \param[in] col current column entry
  inline void push_back_empty_col(const size_type psmilu_debug_code(col)) {
    _base::_push_back_primary_empty(col);
  }

  /// \brief push back a new column
  /// \tparam Iter iterator type
  /// \tparam ValueArray dense value array
  /// \param[in] col current col (C-based index)
  /// \param[in] first starting iterator
  /// \param[in] last ending iterator
  /// \param[in] v dense value array, the value can be queried from indices
  /// \warning The indices are assumed to be sorted
  template <class Iter, class ValueArray>
  inline void push_back_col(const size_type psmilu_debug_code(col), Iter first,
                            Iter last, const ValueArray &v) {
    _base::_push_back_primary(col, first, last, v);
  }

 protected:
  size_type _nrows;     ///< number of rows
  using _base::_psize;  ///< number of columns (primary entries)
};

}  // namespace psmilu

#endif  // _PSMILU_COMPRESSEDSTORAGE_HPP
