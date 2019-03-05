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
#include <type_traits>
#include <utility>

#include "psmilu_Array.hpp"
#include "psmilu_matrix_market.hpp"
#include "psmilu_native_io.hpp"
#include "psmilu_utils.hpp"

namespace psmilu {
namespace internal {

/// \class CompressedStorage
/// \brief Core of the compressed storage, including data and interfaces
/// \tparam ValueType value type, e.g. \a double, \a float, etc
/// \tparam IndexType index used, e.g. \a int, \a long, etc
/// \tparam OneBased if \a true, then Fortran based index is assumed
/// \ingroup ds
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
  CompressedStorage() { _psize = _ZERO; }

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
  explicit CompressedStorage(const size_type n, const size_type nnz = 0u,
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

  /// \brief check the validity of a matrix
  /// \tparam IsRow is row major, mainly for error message
  /// \param[in] other_size size in secondary entry, column size for CRS and
  ///                       row size for CCS
  template <bool IsRow>
  inline void check_validity(const size_type other_size) const {
    static const char *pname = IsRow ? "row" : "column";
    static const char *oname = IsRow ? "column" : "row";
    const auto         c_idx = [](const size_type i) {
      return to_c_idx<size_type, OneBased>(i);
    };

    if (status() != DATA_UNDEF) {
      psmilu_error_if(_ind_start.size() != _psize + 1,
                      "invalid %s size and %s start array length", pname,
                      pname);
      psmilu_error_if(_ind_start.front() != static_cast<index_type>(OneBased),
                      "the leading entry in %s start should be %d", pname,
                      (int)OneBased);
      if (nnz() != _indices.size() || nnz() != _vals.size())
        psmilu_error(
            "inconsistent between nnz (%s_start(end)-%s_start(first),%zd) and "
            "indices/values length %zd/%zd",
            pname, pname, nnz(), _indices.size(), _vals.size());
      // check each entry
      for (size_type i = 0u; i < _psize; ++i) {
        psmilu_error_if(!std::is_sorted(_ind_cbegin(i), _ind_cend(i)),
                        "%s %zd (C-based) is not sorted", pname, i);
        for (auto itr = _ind_cbegin(i), last = _ind_cend(i); itr != last;
             ++itr) {
          psmilu_error_if(c_idx(*itr) >= other_size,
                          "%zd (C-based) exceeds %s bound %zd in %s %zd",
                          c_idx(*itr), oname, other_size, pname, i);
        }
      }
    }
  }

 protected:
  /// \brief begin assemble rows
  /// \note PSMILU only requires pushing back operations
  /// \sa _end_assemble
  inline void _begin_assemble() {
    _ind_start.reserve(_psize + 1u);
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
    psmilu_warning("detected empty primary entries (%zd)",
                   _psize + 1u - _ind_start.size());
    _psize = _ind_start.size() - 1u;
  }

  /// \brief push back an empty entry
  /// \param[in] ii ii-th entry in primary direction
  inline void _push_back_primary_empty(const size_type ii) {
    psmilu_assert(ii + 1u == _ind_start.size(),
                  "inconsistent pushing back at entry %zd", ii);
    (void)ii;
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
  inline void _push_back_primary(const size_type ii, Iter first, Iter last,
                                 const ValueArray &v) {
    psmilu_assert(ii + 1u == _ind_start.size(),
                  "inconsistent pushing back at entry %zd", ii);
    (void)ii;
    // first push back the list
    psmilu_assert(_indices.size() == _vals.size(), "fatal error");
    _indices.push_back(first, last);
    const auto      nnz   = std::distance(first, last);
    const size_type start = _vals.size();
    psmilu_assert(nnz >= 0, "reversed iterators");
    _vals.resize(start + nnz);
    psmilu_assert(_indices.size() == _vals.size(), "fatal error");
    const size_type n = _vals.size();
    for (auto i = start; i < n; ++i, ++first) {
      const auto j = to_c_idx<size_type, OneBased>(*first);
      psmilu_assert(j < v.size(), "%zd exceeds the length of v", j);
      _vals[i] = v[j];
    }
    // finally, push back ind start
    _ind_start.push_back(_ind_start.back() + nnz);
  }

  /// \brief push back two \b SORTED index lists to _indices
  /// \tparam Iter1 iterator type 1
  /// \tparam ValueArray1 dense value array 1
  /// \tparam Iter2 iterator type 2
  /// \tparam ValueArray2 dense value array 2
  /// \param[in] ii ii-th entry in primary direction
  /// \param[in] first1 starting iterator
  /// \param[in] last1 ending iterator
  /// \param[in] v1 dense value array, the value can be queried from indices
  /// \param[in] first2 starting iterator
  /// \param[in] last2 ending iterator
  /// \param[in] v2 dense value array, the value can be queried from indices
  /// \warning The indices are assumed to be sorted
  template <class Iter1, class ValueArray1, class Iter2, class ValueArray2>
  inline void _push_back_primary(const size_type ii, Iter1 first1, Iter1 last1,
                                 const ValueArray1 &v1, Iter2 first2,
                                 Iter2 last2, const ValueArray2 &v2) {
    psmilu_assert(ii + 1u == _ind_start.size(),
                  "inconsistent pushing back at entry %zd", ii);
    (void)ii;
    // first push back the list
    psmilu_assert(_indices.size() == _vals.size(), "fatal error");
    _indices.push_back(first1, last1);
    _indices.push_back(first2, last2);
    const auto nnz1  = std::distance(first1, last1);
    const auto nnz2  = std::distance(first2, last2);
    size_type  start = _vals.size();
    psmilu_assert(nnz1 >= 0 && nnz2 >= 0, "reversed iterators");
    // first range
    _vals.resize(start + nnz1);
    const size_type n1 = _vals.size();
    for (; start < n1; ++start, ++first1) {
      const auto j = to_c_idx<size_type, OneBased>(*first1);
      psmilu_assert(j < v1.size(), "%zd exceeds the length of v1", j);
      _vals[start] = v1[j];
    }
    // second range
    _vals.resize(start + nnz2);
    const size_type n2 = _vals.size();
    for (; start < n2; ++start, ++first2) {
      const auto j = to_c_idx<size_type, OneBased>(*first2);
      psmilu_assert(j < v2.size(), "%zd exceeds the length of v2", j);
      _vals[start] = v2[j];
    }
    // finally, push back ind start
    _ind_start.push_back(_ind_start.back() + nnz1 + nnz2);
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
    return _ind_begin(i) + _nnz(i);
  }
  inline const_i_iterator _ind_begin(const size_type i) const {
    return _indices.begin() + to_c_idx<size_type, OneBased>(_ind_start[i]);
  }
  inline const_i_iterator _ind_end(const size_type i) const {
    return _ind_begin(i) + _nnz(i);
  }
  inline const_i_iterator _ind_cbegin(const size_type i) const {
    return _indices.begin() + to_c_idx<size_type, OneBased>(_ind_start[i]);
  }
  inline const_i_iterator _ind_cend(const size_type i) const {
    return _ind_cbegin(i) + _nnz(i);
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
/// \ingroup ds
template <class ValueArray, class IndexArray, bool OneBased>
inline void convert_storage(const IndexArray &i_ind_start,
                            const IndexArray &i_indices,
                            const ValueArray &i_vals, IndexArray &o_ind_start,
                            IndexArray &o_indices, ValueArray &o_vals) {
  typedef typename ValueArray::size_type  size_type;
  typedef typename IndexArray::value_type index_type;
  constexpr static size_type base = static_cast<size_type>(OneBased);

  // o_ind_start is allocated and uniformly assigned to be OneBased
  // both o_{indices,vals} are uninitialized arrays with size of input nnz
  psmilu_error_if(i_indices.size() != i_ind_start.back() - base,
                  "nnz %zd does not match that in start array %zd",
                  i_indices.size(), i_ind_start.back() - base);
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
/// \ingroup ds
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
  ///< const index iterator
  typedef typename _base::v_iterator       v_iterator;  ///< value iterator
  typedef typename _base::const_v_iterator const_v_iterator;
  ///< const value iterator
  constexpr static bool ONE_BASED = OneBased;  ///< C or Fortran based
  constexpr static bool ROW_MAJOR = true;      ///< row major

  /// \brief read a matrix market file
  /// \param[in] filename matrix file name
  /// \return A CRS matrix
  inline static CRS from_mm(const char *filename) {
    CRS       crs;
    size_type rows, cols;
    read_matrix_market<array_type, iarray_type, OneBased, true>(
        filename, crs.row_start(), crs.col_ind(), crs.vals(), rows, cols);
    crs.resize(rows, cols);
    return crs;
  }

  /// \brief read a matrix from PSMILU native binary file
  /// \param[in] filename file name
  /// \param[out] m if given, the leading block size is also returned
  /// \return A CRS matrix
  inline static CRS from_native_bin(const char *filename,
                                    size_type * m = nullptr) {
    CRS        crs;
    const auto b_size = crs.read_native_bin(filename);
    if (m) *m = b_size;
    return crs;
  }

  /// \brief read a matrix from PSMILU native ASCII file
  /// \param[in] filename file name
  /// \param[out] m if given, the leading block size is also returned
  /// \return A CRS matrix
  inline static CRS from_native_ascii(const char *filename,
                                      size_type * m = nullptr) {
    CRS        crs;
    const auto b_size = crs.read_native_ascii(filename);
    if (m) *m = b_size;
    return crs;
  }

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

  /// \brief check validity
  inline void check_validity() const {
    _base::template check_validity<true>(_ncols);
  }

  /// \brief get local nnz per row
  /// \param[in] i row entry (C-based index)
  /// \return number of nonzeros in row \a i
  inline size_type nnz_in_row(const size_type i) const {
    return _base::_nnz(i);
  }

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
  inline void push_back_empty_row(const size_type row) {
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
  inline void push_back_row(const size_type row, Iter first, Iter last,
                            const ValueArray &v) {
    _base::_push_back_primary(row, first, last, v);
  }

  /// \brief push back a new row from two lists
  /// \tparam Iter1 iterator type
  /// \tparam ValueArray1 dense value array
  /// \tparam Iter2 iterator type
  /// \tparam ValueArray2 dense value array
  /// \param[in] row current row (C-based index)
  /// \param[in] first1 starting iterator
  /// \param[in] last1 ending iterator
  /// \param[in] v1 dense value array, the value can be queried from indices
  /// \param[in] first2 starting iterator
  /// \param[in] last2 ending iterator
  /// \param[in] v2 dense value array, the value can be queried from indices
  /// \warning The indices are assumed to be sorted
  template <class Iter1, class ValueArray1, class Iter2, class ValueArray2>
  inline void push_back_row(const size_type row, Iter1 first1, Iter1 last1,
                            const ValueArray1 &v1, Iter2 first2, Iter2 last2,
                            const ValueArray2 &v2) {
    _base::_push_back_primary(row, first1, last1, v1, first2, last2, v2);
  }

  /// \brief scale by a diagonal matrix from left
  /// \tparam DiagArray diagonal array type
  /// \param[in] s diagonal matrix multiplying from left-hand side
  /// \sa scale_diag_right
  ///
  /// Mathematically, this member function is to perform
  /// \f$ \textrm{diag}(\boldsymbol{s})\boldsymbol{A}\f$; the overall complexity
  /// is in order of \f$\mathcal{O}(nnz)\f$
  template <class DiagArray>
  inline void scale_diag_left(const DiagArray &s) {
    psmilu_error_if(row_start().size() > s.size() + 1u,
                    "row sizes do not match (%zd,%zd)", row_start().size() - 1u,
                    s.size());
    const size_type n = row_start().size();
    for (size_type i = 1u; i < n; ++i) {
      const auto d = s[i - 1];
      std::for_each(_base::val_begin(i - 1), _base::val_end(i - 1),
                    [=](value_type &v) { v *= d; });
    }
  }

  /// \brief scale by a diagonal matrix from right
  /// \tparam DiagArray diagonal array type
  /// \param[in] t diagonal matrix multiplying from right-hand side
  /// \sa scale_diag_left
  ///
  /// Mathematically, this member function is to perform
  /// \f$\boldsymbol{A} \textrm{diag}(\boldsymbol{t})\f$; the overall complexity
  /// is in order of \f$\mathcal{O}(nnz)\f$
  template <class DiagArray>
  inline void scale_diag_right(const DiagArray &t) {
    psmilu_error_if(_ncols > t.size(), "column sizes do not match (%zd,%zd)",
                    _ncols, t.size());
    auto v_itr = vals().begin();
    auto i_itr = col_ind().cbegin();
    for (auto last = col_ind().cend(); i_itr != last; ++i_itr, ++v_itr)
      *v_itr *= t[to_c_idx<size_type, ONE_BASED>(*i_itr)];
  }

  /// \brief scale by two diagonal matrices from both lh and rh sides
  /// \tparam LeftDiagArray diagonal array type of left side
  /// \tparam RightDiagArray diagonal array type of right side
  /// \param[in] s diagonal matrix of left side
  /// \param[in] t diagonal matrix of right side
  ///
  /// Mathematically, this member function is to perform
  /// \f$\textrm{diag}(\boldsymbol{s}) \boldsymbol{A}
  /// \textrm{diag}(\boldsymbol{t})\f$
  template <class LeftDiagArray, class RightDiagArray>
  inline void scale_diags(const LeftDiagArray &s, const RightDiagArray &t) {
    scale_diag_left(s);
    scale_diag_right(t);
  }

  /// \brief matrix vector multiplication (low level API)
  /// \param[in] x input array pointer
  /// \param[out] y output array pointer
  /// \warning User's responsibility to maintain valid pointers
  inline void mv_nt_low(const value_type *x, value_type *y) const {
    for (size_type i = 0u; i < _psize; ++i) {
      y[i]       = 0;
      auto v_itr = _base::val_cbegin(i);
      auto i_itr = col_ind_cbegin(i);
      for (auto last = col_ind_cend(i); i_itr != last; ++i_itr, ++v_itr) {
        const size_type j = to_c_idx<size_type, ONE_BASED>(*i_itr);
        psmilu_assert(j < _ncols, "%zd exceeds column size", j);
        y[i] += *v_itr * x[j];
      }
    }
  }

  /// \brief matrix vector multiplication
  /// \tparam IArray input array type
  /// \tparam OArray output array type
  /// \param[in] x input array
  /// \param[out] y output array
  /// \note Sizes must match
  template <class IArray, class OArray>
  inline void mv_nt(const IArray &x, OArray &y) const {
    psmilu_error_if(nrows() != y.size() || ncols() != x.size(),
                    "matrix vector multiplication unmatched sizes!");
    mv_nt_low(x.data(), y.data());
  }

  /// \brief matrix transpose vector multiplication (low level API)
  /// \param[in] x input array pointer
  /// \param[out] y output array pointer
  /// \warning User's responsibility to maintain valid pointers
  inline void mv_t_low(const value_type *x, value_type *y) const {
    if (!_psize) return;
    std::fill_n(y, ncols(), 0);
    for (size_type i = 0u; i < _psize; ++i) {
      const auto temp  = x[i];
      auto       v_itr = _base::val_cbegin(i);
      auto       i_itr = col_ind_cbegin(i);
      for (auto last = col_ind_cend(i); i_itr != last; ++i_itr, ++v_itr) {
        const size_type j = to_c_idx<size_type, ONE_BASED>(*i_itr);
        psmilu_assert(j < _ncols, "%zd exceeds column size", j);
        y[j] += *v_itr * temp;
      }
    }
  }

  /// \brief matrix transpose vector multiplication
  /// \tparam IArray input array type
  /// \tparam OArray output array type
  /// \param[in] x input array
  /// \param[out] y output array
  /// \note Sizes must match
  /// \note Compute \f$y=\boldsymbol{A}^Tx\f$
  template <class IArray, class OArray>
  inline void mv_t(const IArray &x, OArray &y) const {
    psmilu_error_if(nrows() != x.size() || ncols() != y.size(),
                    "T(matrix) vector multiplication unmatched sizes!");
    mv_t_low(x.data(), y.data());
  }

  /// \brief matrix vector multiplication
  /// \tparam IArray input array type
  /// \tparam OArray output array type
  /// \param[in] x input array
  /// \param[out] y output array
  /// \param[in] tran if \a false (default), perform normal matrix vector
  /// \sa mv_nt, mv_t
  template <class IArray, class OArray>
  inline void mv(const IArray &x, OArray &y, bool tran = false) const {
    !tran ? mv_nt(x, y) : mv_t(x, y);
  }

  /// \brief read a native binary file
  /// \param[in] fname file name
  /// \return leading symmetric block size
  inline size_type read_native_bin(const char *fname) {
    return psmilu::read_native_bin(fname, *this);
  }

  /// \brief write a native binary file
  /// \param[in] fname file name
  /// \param[in] m leading block size
  inline void write_native_bin(const char *fname, const size_type m = 0) const {
    psmilu::write_native_bin(fname, *this, m);
  }

  /// \brief write a native ASCII file
  /// \param[in] fname file name
  /// \param[in] m leading block size
  inline void write_native_ascii(const char *    fname,
                                 const size_type m = 0) const {
    psmilu::write_native_ascii<true, ONE_BASED>(fname, row_start(), _ncols,
                                                col_ind(), vals(), m);
  }

  /// \brief read data from an ASCII file
  /// \param[in] fname file name
  /// \return the leading symmetric block size
  inline size_type read_native_ascii(const char *fname) {
    bool        is_row, is_c;
    char        dtype;
    size_type   row, col, Nnz, m;
    iarray_type i_start, is;
    array_type  vs;
    std::tie(is_row, is_c, dtype, row, col, Nnz, m) =
        psmilu::read_native_ascii(fname, i_start, is, vs);
    const int shift = ONE_BASED - !is_c;
    if (shift) {
      for (auto &v : i_start) v += shift;
      for (auto &v : is) v += shift;
    }
    if (is_row) {
      row_start() = std::move(i_start);
      col_ind()   = std::move(is);
      vals()      = std::move(vs);
      _ncols      = col;
      _psize      = row;
    } else
      *this =
          CRS(other_type(row, col, i_start.data(), is.data(), vs.data(), true));

    return m;
  }

 protected:
  size_type _ncols;     ///< number of columns
  using _base::_psize;  ///< number of rows (primary entries)
};

/// \brief wrap user data
/// \tparam OneBased if \a true, then assume Fortran index system
/// \tparam ValueType value type, e.g. \a double
/// \tparam IndexType index type, e.g. \a int
/// \param[in] nrows number of rows
/// \param[in] ncols number of columns
/// \param[in] row_start data for row pointer, size of \a nrows + 1
/// \param[in] col_ind column indices
/// \param[in] vals numerical data
/// \param[in] check if \a true (default), then perform validation checking
/// \param[in] help_sort help sort unsorted rows (if any), require \a check=1
/// \return a \ref CRS matrix wrapped around user data.
/// \warning It's the user's responsibility to maintain the external data
/// \ingroup ds
template <bool OneBased, class ValueType, class IndexType>
inline CRS<ValueType, IndexType, OneBased> wrap_crs(
    const typename CRS<ValueType, IndexType, OneBased>::size_type nrows,
    const typename CRS<ValueType, IndexType, OneBased>::size_type ncols,
    IndexType *row_start, IndexType *col_ind, ValueType *vals,
    bool check = true, bool help_sort = false) {
  using return_type = CRS<ValueType, IndexType, OneBased>;
  using size_type   = typename CRS<ValueType, IndexType, OneBased>::size_type;
  constexpr static bool WRAP = true;
  static_assert(std::is_integral<IndexType>::value, "must be integer");

  // run time

  return_type mat(nrows, ncols, const_cast<IndexType *>(row_start),
                  const_cast<IndexType *>(col_ind),
                  const_cast<ValueType *>(vals), WRAP);

  if (check) {
    if (row_start[0] != (IndexType)OneBased)
      psmilu_error("first entry of row_start does not agree with OneBased.");
    Array<ValueType> buf;
    for (size_type i = 0u; i < nrows; ++i) {
      if (!mat.nnz_in_row(i)) {
        psmilu_warning("row %zd is empty!", i);
        continue;
      }
      auto last = mat.col_ind_cend(i), first = mat.col_ind_cbegin(i);
      auto itr = std::is_sorted_until(
          first, last,
          [](const IndexType i, const IndexType j) { return i <= j; });
      if (itr != last) {
        if (!help_sort)
          psmilu_error(
              "%zd row is not sorted, the checking failed at entry %td, run "
              "with help_sort=true",
              i, itr - first);
        else {
          buf.resize(mat.ncols() + OneBased);
          if (buf.status() == DATA_UNDEF)
            psmilu_error("memory allocation failed!");
          // load values, note that we can just load [itr,last), but we want to
          // loop through all indices to ensure they are bounded
          auto v_itr = mat.val_cbegin(i);
          for (itr = first; itr != last; ++itr, ++v_itr) {
            psmilu_error_if(
                *itr < OneBased || (size_type)*itr >= mat.ncols() + OneBased,
                "%zd exceeds column size %zd", size_type(*itr), mat.ncols());
            buf[*itr] = *v_itr;
          }
          std::sort(mat.col_ind_begin(i), mat.col_ind_end(i));
          auto vv_itr = mat.val_begin(i);
          last        = mat.col_ind_cend(i);
          first       = mat.col_ind_cbegin(i);
          for (itr = first; itr != last; ++itr, ++vv_itr) *vv_itr = buf[*itr];
        }
      } else
        psmilu_error_if(size_type(*(last - 1)) >= mat.ncols() + OneBased,
                        "%zd exceeds column size %zd", size_type(*(last - 1)),
                        mat.ncols());
    }
  }
  return mat;
}

/// \brief wrap user data in C index
/// \tparam ValueType value type, e.g. \a double
/// \tparam IndexType index type, e.g. \a int
/// \param[in] nrows number of rows
/// \param[in] ncols number of columns
/// \param[in] row_start data for row pointer, size of \a nrows + 1
/// \param[in] col_ind column indices
/// \param[in] vals numerical data
/// \param[in] check if \a true (default), then perform validation checking
/// \param[in] help_sort help sort unsorted rows (if any), require \a check=1
/// \return a \ref CRS matrix wrapped around user data.
/// \warning It's the user's responsibility to maintain the external data
/// \ingroup ds
template <class ValueType, class IndexType>
inline CRS<ValueType, IndexType> wrap_crs_0(
    const typename CRS<ValueType, IndexType>::size_type nrows,
    const typename CRS<ValueType, IndexType>::size_type ncols,
    const IndexType *row_start, const IndexType *col_ind, const ValueType *vals,
    bool check = true, bool help_sort = false) {
  return wrap_crs<false>(nrows, ncols, row_start, col_ind, vals, check,
                         help_sort);
}

/// \brief wrap user data in Fortran index
/// \tparam ValueType value type, e.g. \a double
/// \tparam IndexType index type, e.g. \a int
/// \param[in] nrows number of rows
/// \param[in] ncols number of columns
/// \param[in] row_start data for row pointer, size of \a nrows + 1
/// \param[in] col_ind column indices
/// \param[in] vals numerical data
/// \param[in] check if \a true (default), then perform validation checking
/// \param[in] help_sort help sort unsorted rows (if any), require \a check=1
/// \return a \ref CRS matrix wrapped around user data.
/// \warning It's the user's responsibility to maintain the external data
/// \ingroup ds
template <class ValueType, class IndexType>
inline CRS<ValueType, IndexType, true> wrap_crs_1(
    const typename CRS<ValueType, IndexType, true>::size_type nrows,
    const typename CRS<ValueType, IndexType, true>::size_type ncols,
    const IndexType *row_start, const IndexType *col_ind, const ValueType *vals,
    bool check = true, bool help_sort = false) {
  return wrap_crs<true>(nrows, ncols, row_start, col_ind, vals, check,
                        help_sort);
}

/// \class CCS
/// \brief Compressed Column Storage (CCS) format for sparse matrices
/// \tparam ValueType numerical value type, e.g. \a double, \a float, etc
/// \tparam IndexType index type, e.g. \a int, \a long, etc
/// \tparam OneBased if \a false (default), using C-based index
/// \ingroup ds
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
  ///< const index iterator
  typedef typename _base::v_iterator       v_iterator;  ///< value iterator
  typedef typename _base::const_v_iterator const_v_iterator;
  ///< const value iterator
  constexpr static bool ONE_BASED = OneBased;  ///< C or Fortran based
  constexpr static bool ROW_MAJOR = false;     ///< column major

  /// \brief read a matrix market file
  /// \param[in] filename matrix file name
  /// \return A CCS matrix
  inline static CCS from_mm(const char *filename) {
    CCS       ccs;
    size_type rows, cols;
    read_matrix_market<array_type, iarray_type, OneBased, false>(
        filename, ccs.col_start(), ccs.row_ind(), ccs.vals(), rows, cols);
    ccs.resize(rows, cols);
    return ccs;
  }

  /// \brief read from a native PSMILU binary file
  /// \param[in] filename file name
  /// \param[out] m if given, then it will store the leading symmetric size
  /// \return A CCS matrix
  inline static CCS from_native_bin(const char *filename,
                                    size_type * m = nullptr) {
    CCS        ccs;
    const auto b_size = ccs.read_native_bin(filename);
    if (m) *m = b_size;
    return ccs;
  }

  /// \brief read from a native PSMILU ASCII file
  /// \param[in] filename file name
  /// \param[out] m if given, then it will store the leading symmetric size
  /// \return A CCS matrix
  inline static CCS from_native_ascii(const char *filename,
                                      size_type * m = nullptr) {
    CCS        ccs;
    const auto b_size = ccs.read_native_ascii(filename);
    if (m) *m = b_size;
    return ccs;
  }

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
  inline size_type ncols() const { return _psize; }

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

  /// \brief check validity
  inline void check_validity() const {
    _base::template check_validity<false>(_nrows);
  }

  /// \brief get local nnz per column
  /// \param[in] i column entry (C-based index)
  /// \return number of nonzeros in column \a i
  inline size_type nnz_in_col(const size_type i) const {
    return _base::_nnz(i);
  }

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
  inline void push_back_empty_col(const size_type col) {
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
  inline void push_back_col(const size_type col, Iter first, Iter last,
                            const ValueArray &v) {
    _base::_push_back_primary(col, first, last, v);
  }

  /// \brief push back a new column from two lists
  /// \tparam Iter1 iterator type
  /// \tparam ValueArray1 dense value array
  /// \tparam Iter2 iterator type
  /// \tparam ValueArray2 dense value array
  /// \param[in] col current col (C-based index)
  /// \param[in] first1 starting iterator
  /// \param[in] last1 ending iterator
  /// \param[in] v1 dense value array, the value can be queried from indices
  /// \param[in] first2 starting iterator
  /// \param[in] last2 ending iterator
  /// \param[in] v2 dense value array, the value can be queried from indices
  /// \warning The indices are assumed to be sorted
  template <class Iter1, class ValueArray1, class Iter2, class ValueArray2>
  inline void push_back_col(const size_type col, Iter1 first1, Iter1 last1,
                            const ValueArray1 &v1, Iter2 first2, Iter2 last2,
                            const ValueArray2 &v2) {
    _base::_push_back_primary(col, first1, last1, v1, first2, last2, v2);
  }

  /// \brief scale by a diagonal matrix from left
  /// \tparam DiagArray diagonal array type
  /// \param[in] s diagonal matrix multiplying from left-hand side
  /// \sa scale_diag_right
  ///
  /// Mathematically, this member function is to perform
  /// \f$ \textrm{diag}(\boldsymbol{s})\boldsymbol{A}\f$; the overall complexity
  /// is in order of \f$\mathcal{O}(nnz)\f$
  template <class DiagArray>
  inline void scale_diag_left(const DiagArray &s) {
    psmilu_error_if(_nrows > s.size(), "row sizes do not match (%zd,%zd)",
                    _nrows, s.size());
    auto v_itr = vals().begin();
    auto i_itr = row_ind().cbegin();
    for (auto last = row_ind().cend(); i_itr != last; ++i_itr, ++v_itr)
      *v_itr *= s[to_c_idx<size_type, ONE_BASED>(*i_itr)];
  }

  /// \brief scale by a diagonal matrix from right
  /// \tparam DiagArray diagonal array type
  /// \param[in] t diagonal matrix multiplying from left-hand side
  /// \sa scale_diag_left
  ///
  /// Mathematically, this member function is to perform
  /// \f$\boldsymbol{A} \textrm{diag}(\boldsymbol{s})\f$; the overall complexity
  /// is in order of \f$\mathcal{O}(nnz)\f$
  template <class DiagArray>
  inline void scale_diag_right(const DiagArray &t) {
    psmilu_error_if(col_start().size() > t.size() + 1u,
                    "column sizes do not match (%zd,%zd)",
                    col_start().size() - 1u, t.size());
    const size_type n = col_start().size();
    for (size_type i = 1u; i < n; ++i) {
      const auto d = t[i - 1];
      std::for_each(_base::val_begin(i - 1), _base::val_end(i - 1),
                    [=](value_type &v) { v *= d; });
    }
  }

  /// \brief scale by two diagonal matrices from both lh and rh sides
  /// \tparam LeftDiagArray diagonal array type of left side
  /// \tparam RightDiagArray diagonal array type of right side
  /// \param[in] s diagonal matrix of left side
  /// \param[in] t diagonal matrix of right side
  ///
  /// Mathematically, this member function is to perform
  /// \f$\textrm{diag}(\boldsymbol{s}) \boldsymbol{A}
  /// \textrm{diag}(\boldsymbol{t})\f$
  template <class LeftDiagArray, class RightDiagArray>
  inline void scale_diags(const LeftDiagArray &s, const RightDiagArray &t) {
    scale_diag_left(s);
    scale_diag_right(t);
  }

  /// \brief matrix vector multiplication (low level API)
  /// \param[in] x input array pointer
  /// \param[out] y output array pointer
  /// \warning User's responsibilty to ensure valid pointers
  inline void mv_nt_low(const value_type *x, value_type *y) const {
    if (!_psize) return;
    std::fill_n(y, nrows(), 0);
    for (size_type i = 0u; i < _psize; ++i) {
      const auto temp  = x[i];
      auto       v_itr = _base::val_cbegin(i);
      auto       i_itr = row_ind_cbegin(i);
      for (auto last = row_ind_cend(i); i_itr != last; ++i_itr, ++v_itr) {
        const size_type j = to_c_idx<size_type, ONE_BASED>(*i_itr);
        psmilu_assert(j < _nrows, "%zd exceeds the size bound", j);
        y[j] += temp * *v_itr;
      }
    }
  }

  /// \brief matrix vector multiplication
  /// \tparam IArray input array type
  /// \tparam OArray output array type
  /// \param[in] x input array
  /// \param[out] y output array
  /// \note Sizes must match
  template <class IArray, class OArray>
  inline void mv_nt(const IArray &x, OArray &y) const {
    psmilu_error_if(nrows() != y.size() || ncols() != x.size(),
                    "matrix vector unmatched sizes!");
    mv_nt_low(x.data(), y.data());
  }

  /// \brief matrix transpose vector multiplication (low level API)
  /// \param[in] x input array pointer
  /// \param[out] y output array pointer
  /// \warning User's responsibilty to ensure valid pointers
  inline void mv_t_low(const value_type *x, value_type *y) const {
    for (size_type i = 0u; i < _psize; ++i) {
      y[i]       = 0;
      auto v_itr = _base::val_cbegin(i);
      auto i_itr = row_ind_cbegin(i);
      for (auto last = row_ind_cend(i); i_itr != last; ++i_itr, ++v_itr) {
        const size_type j = to_c_idx<size_type, ONE_BASED>(*i_itr);
        psmilu_assert(j < _nrows, "%zd exceeds the size bound", j);
        y[i] += x[j] * *v_itr;
      }
    }
  }

  /// \brief matrix transpose vector multiplication
  /// \tparam IArray input array type
  /// \tparam OArray output array type
  /// \param[in] x input array
  /// \param[out] y output array
  /// \note Sizes must match
  template <class IArray, class OArray>
  inline void mv_t(const IArray &x, OArray &y) const {
    psmilu_error_if(nrows() != x.size() || ncols() != y.size(),
                    "T(matrix) vector unmatched sizes!");
    mv_t_low(x.data(), y.data());
  }

  /// \brief matrix vector multiplication
  /// \tparam IArray input array type
  /// \tparam OArray output array type
  /// \param[in] x input array
  /// \param[out] y output array
  /// \param[in] tran if \a false (default), then perform normal matrix vector
  /// \sa mv_t, mv_nt
  template <class IArray, class OArray>
  inline void mv(const IArray &x, OArray &y, bool tran = false) const {
    !tran ? mv_nt(x, y) : mv_t(x, y);
  }

  /// \brief read a native PSMILU binary file
  /// \param[in] fname file name
  /// \return leading symmetric block size
  inline size_type read_native_bin(const char *fname) {
    return psmilu::read_native_bin(fname, *this);
  }

  /// \brief write to a native PSMILU binary file
  /// \param[in] fname file name
  /// \param[in] m leading block size
  inline void write_native_bin(const char *fname, const size_type m = 0) const {
    psmilu::write_native_bin(fname, *this, m);
  }

  /// \brief write a native ASCII file
  /// \param[in] fname file name
  /// \param[in] m leading block size
  inline void write_native_ascii(const char *    fname,
                                 const size_type m = 0) const {
    psmilu::write_native_ascii<false, ONE_BASED>(fname, col_start(), _nrows,
                                                 row_ind(), vals(), m);
  }

  /// \brief read data from an ASCII file
  /// \param[in] fname file name
  /// \return the leading symmetric block size
  inline size_type read_native_ascii(const char *fname) {
    bool        is_row, is_c;
    char        dtype;
    size_type   row, col, Nnz, m;
    iarray_type i_start, is;
    array_type  vs;
    std::tie(is_row, is_c, dtype, row, col, Nnz, m) =
        psmilu::read_native_ascii(fname, i_start, is, vs);
    const int shift = ONE_BASED - !is_c;
    if (shift) {
      for (auto &v : i_start) v += shift;
      for (auto &v : is) v += shift;
    }
    if (!is_row) {
      col_start() = std::move(i_start);
      row_ind()   = std::move(is);
      vals()      = std::move(vs);
      _nrows      = row;
      _psize      = col;
    } else
      *this =
          CCS(other_type(row, col, i_start.data(), is.data(), vs.data(), true));

    return m;
  }

 protected:
  size_type _nrows;     ///< number of rows
  using _base::_psize;  ///< number of columns (primary entries)
};

}  // namespace psmilu

#endif  // _PSMILU_COMPRESSEDSTORAGE_HPP
