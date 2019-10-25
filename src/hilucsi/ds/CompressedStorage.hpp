///////////////////////////////////////////////////////////////////////////////
//                This file is part of HILUCSI project                       //
///////////////////////////////////////////////////////////////////////////////

/*!
 * \file hilucsi/ds/CompressedStorage.hpp
 * \brief Core about CRS and CCS storage formats
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

#ifndef _HILUCSI_DS_COMPRESSEDSTORAGE_HPP
#define _HILUCSI_DS_COMPRESSEDSTORAGE_HPP

#include <algorithm>
#include <iterator>
#include <type_traits>
#include <utility>

#include "hilucsi/ds/Array.hpp"
#include "hilucsi/utils/common.hpp"
#include "hilucsi/utils/io.hpp"

namespace hilucsi {
namespace internal {

/// \class CompressedStorage
/// \brief Core of the compressed storage, including data and interfaces
/// \tparam ValueType value type, e.g. \a double, \a float, etc
/// \tparam IndexType index used, e.g. \a int, \a long, etc
/// \ingroup ds
template <class ValueType, class IndexType>
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

 public:
  /// \brief default constructor
  CompressedStorage() { _psize = 0; }

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
    std::fill(_ind_start.begin(), _ind_start.end(), index_type(0));
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
    return status() == DATA_UNDEF ? 0 : _ind_start.back() - _ind_start.front();
  }

  // getting information for local value ranges

  inline v_iterator val_begin(const size_type i) {
    return _vals.begin() + _ind_start[i];
  }
  inline v_iterator val_end(const size_type i) {
    return val_begin(i) + _nnz(i);
  }
  inline const_v_iterator val_begin(const size_type i) const {
    return _vals.begin() + _ind_start[i];
  }
  inline const_v_iterator val_end(const size_type i) const {
    return val_begin(i) + _nnz(i);
  }
  inline const_v_iterator val_cbegin(const size_type i) const {
    return _vals.begin() + _ind_start[i];
  }
  inline const_v_iterator val_cend(const size_type i) const {
    return val_cbegin(i) + _nnz(i);
  }
  inline size_type  nnz_in_primary(const size_type i) const { return _nnz(i); }
  inline i_iterator ind_begin(const size_type i) { return _ind_begin(i); }
  inline i_iterator ind_end(const size_type i) { return _ind_end(i); }
  inline const_i_iterator ind_begin(const size_type i) const {
    return _ind_cbegin(i);
  }
  inline const_i_iterator ind_end(const size_type i) const {
    return _ind_cend(i);
  }
  inline const_i_iterator ind_cbegin(const size_type i) const {
    return _ind_cbegin(i);
  }
  inline const_i_iterator ind_cend(const size_type i) const {
    return _ind_cend(i);
  }
  inline iarray_type &      inds() { return _indices; }
  inline const iarray_type &inds() const { return _indices; }
  inline iarray_type &      ind_start() { return _ind_start; }
  inline const iarray_type &ind_start() const { return _ind_start; }

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

    if (status() != DATA_UNDEF) {
      hilucsi_error_if(_ind_start.size() < _psize + 1,
                       "invalid %s size and %s start array length", pname,
                       pname);
      hilucsi_error_if(_ind_start.front() != 0,
                       "the leading entry in %s start should be zero", pname);
      if (nnz() > _indices.size() || nnz() > _vals.size())
        hilucsi_error(
            "inconsistent between nnz (%s_start(end)-%s_start(first),%zd) and "
            "indices/values length %zd/%zd",
            pname, pname, nnz(), _indices.size(), _vals.size());
      // check each entry
      for (size_type i = 0u; i < _psize; ++i) {
        hilucsi_error_if(
            !std::is_sorted(
                _ind_cbegin(i), _ind_cend(i),
                [](const index_type l, const index_type r) { return l <= r; }),
            "%s %zd (C-based) is not sorted", pname, i);
        for (auto itr = _ind_cbegin(i), last = _ind_cend(i); itr != last;
             ++itr) {
          hilucsi_error_if(size_type(*itr) >= other_size,
                           "%zd (C-based) exceeds %s bound %zd in %s %zd",
                           size_type(*itr), oname, other_size, pname, i);
        }
      }
    }
  }

  /// \brief filter out small values
  /// \param[in] eps eliminating threshold, default is 0
  /// \param[in] shrink if \a false (default), then do not shrink memory
  /// \return total elminated entries
  ///
  /// All \a mag(values) that are <= eps * max(abs(local value)) will be
  /// eliminated, and the resulting matrix will be compressed.
  inline size_type eliminate(value_type eps = 0, const bool shrink = false) {
    if (_psize) {
      const size_type psize = _psize;
      eps                   = eps < 0.0 ? 0.0 : eps;
      const size_type nz    = nnz();
      size_type       j(0);
      auto            v_itr_first = _vals.cbegin();
      auto            i_itr_first = _indices.cbegin();
      size_type       pos(0);
      for (size_type i(0); i < psize; ++i) {
        const size_type j_bak(j);
        auto            v_itr = v_itr_first + pos;
        auto            i_itr = i_itr_first + pos;
        const auto      lnnz  = _ind_start[i + 1] - pos;
        if (lnnz) {
          auto             last = v_itr + lnnz;
          const value_type thres =
              eps *
              std::abs(*std::max_element(
                  v_itr, last, [](const value_type l, const value_type r) {
                    return std::abs(l) < std::abs(r);
                  }));
          auto i_itr2 = i_itr;
          for (auto itr = v_itr; itr != last; ++itr, ++i_itr2)
            if (std::abs(*itr) > thres) {
              _indices[j] = *i_itr2;
              _vals[j++]  = *itr;
            }
        }
        // update position
        pos               = _ind_start[i + 1];
        _ind_start[i + 1] = _ind_start[i] + (j - j_bak);
      }
      const size_type elms = nz - j;
      if (elms && shrink) {
        do {
          array_type vals2(nnz());
          hilucsi_error_if(nnz() && vals2.status() == DATA_UNDEF,
                           "memory allocation failed");
          std::copy_n(v_itr_first, nnz(), vals2.begin());
          _vals.swap(vals2);
        } while (false);
        do {
          iarray_type indices2(nnz());
          hilucsi_error_if(nnz() && indices2.status() == DATA_UNDEF,
                           "memory allocation failed");
          std::copy_n(i_itr_first, nnz(), indices2.begin());
          _indices.swap(indices2);
        } while (false);
      } else if (elms) {
        _vals.resize(nnz());
        _indices.resize(nnz());
      }
      return elms;
    } else
      return 0;
  }

 protected:
  /// \brief begin assemble rows
  /// \note HILUCSI only requires pushing back operations
  /// \sa _end_assemble
  inline void _begin_assemble() {
    _ind_start.reserve(_psize + 1u);
    _ind_start.resize(1);
    _ind_start.front() = 0;
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
      // hilucsi_warning("pushed more entries (%zd) than requested (%zd)",
      //                _ind_start.size() - 1u, _psize);
      _psize = _ind_start.size() ? _ind_start.size() - 1u : 0u;
      return;
    }
    // hilucsi_warning("detected empty primary entries (%zd)",
    //                _psize + 1u - _ind_start.size());
    _psize = _ind_start.size() - 1u;
  }

  /// \brief push back an empty entry
  /// \param[in] ii ii-th entry in primary direction
  inline void _push_back_primary_empty(const size_type ii) {
    hilucsi_assert(ii + 1u == _ind_start.size(),
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
    hilucsi_assert(ii + 1u == _ind_start.size(),
                   "inconsistent pushing back at entry %zd", ii);
    (void)ii;
    // first push back the list
    hilucsi_assert(_indices.size() == _vals.size(), "fatal error");
    _indices.push_back(first, last);
    const auto      nnz   = std::distance(first, last);
    const size_type start = _vals.size();
    hilucsi_assert(nnz >= 0, "reversed iterators");
    _vals.resize(start + nnz);
    hilucsi_assert(_indices.size() == _vals.size(), "fatal error");
    const size_type n = _vals.size();
    for (auto i = start; i < n; ++i, ++first) {
      hilucsi_assert(size_type(*first) < v.size(),
                     "%zd exceeds the length of v", size_type(*first));
      _vals[i] = v[*first];
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
    hilucsi_assert(ii + 1u == _ind_start.size(),
                   "inconsistent pushing back at entry %zd", ii);
    (void)ii;
    // first push back the list
    hilucsi_assert(_indices.size() == _vals.size(), "fatal error");
    _indices.push_back(first1, last1);
    _indices.push_back(first2, last2);
    const auto nnz1  = std::distance(first1, last1);
    const auto nnz2  = std::distance(first2, last2);
    size_type  start = _vals.size();
    hilucsi_assert(nnz1 >= 0 && nnz2 >= 0, "reversed iterators");
    // first range
    _vals.resize(start + nnz1);
    const size_type n1 = _vals.size();
    for (; start < n1; ++start, ++first1) {
      hilucsi_assert(size_type(*first1) < v1.size(),
                     "%zd exceeds the length of v1", size_type(*first1));
      _vals[start] = v1[*first1];
    }
    // second range
    _vals.resize(start + nnz2);
    const size_type n2 = _vals.size();
    for (; start < n2; ++start, ++first2) {
      hilucsi_assert(size_type(*first2) < v2.size(),
                     "%zd exceeds the length of v2", size_type(*first2));
      _vals[start] = v2[*first2];
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
    return _indices.begin() + _ind_start[i];
  }
  inline i_iterator _ind_end(const size_type i) {
    return _ind_begin(i) + _nnz(i);
  }
  inline const_i_iterator _ind_begin(const size_type i) const {
    return _indices.begin() + _ind_start[i];
  }
  inline const_i_iterator _ind_end(const size_type i) const {
    return _ind_begin(i) + _nnz(i);
  }
  inline const_i_iterator _ind_cbegin(const size_type i) const {
    return _indices.begin() + _ind_start[i];
  }
  inline const_i_iterator _ind_cend(const size_type i) const {
    return _ind_cbegin(i) + _nnz(i);
  }

  /// \brief split against secondary size
  /// \tparam IsSecond if \a true, then extract the second part
  /// \param[in] m size split against to
  /// \param[in] start position start array
  /// \param[out] indptr index start array
  /// \param[out] indices index value array
  /// \param[out] vals values
  template <bool IsSecond>
  inline void _split(const size_type m, const index_type *start,
                     iarray_type &indptr, iarray_type &indices,
                     array_type &vals) const {
    if (!_ind_start.size()) return;
    indptr.resize(_ind_start.size());
    hilucsi_error_if(indptr.status() == DATA_UNDEF, "memory allocation failed");
    auto      ind_first = _indices.cbegin();
    size_type nnz(0);
    for (size_type i(0); i < _psize; ++i) {
      auto itr = ind_first + start[i];
      nnz += IsSecond ? _ind_cend(i) - itr : itr - _ind_cbegin(i);
    }
    // if we have an empty matrix
    if (!nnz) {
      std::fill(indptr.begin(), indptr.end(), index_type(0));
      return;
    }
    indices.resize(nnz);
    hilucsi_error_if(indices.status() == DATA_UNDEF,
                     "memory allocation failed");
    vals.resize(nnz);
    hilucsi_error_if(vals.status() == DATA_UNDEF, "memory allocation failed");

    auto i_itr = indices.begin();
    auto v_itr = vals.begin();

    auto val_first = _vals.cbegin();

    // now set the indptr front
    indptr.front() = 0;

    for (size_type i = 0; i < _psize; ++i) {
      auto split_ind_itr = start[i] + ind_first;
      auto split_val_itr = start[i] + val_first;
      auto bak_itr       = i_itr;
      if (IsSecond) {
        i_itr = std::copy(split_ind_itr, _ind_cend(i), i_itr);
        for (auto itr = bak_itr; itr != i_itr; ++itr) *itr -= m;
        v_itr = std::copy(split_val_itr, val_cend(i), v_itr);
      } else {
        i_itr = std::copy(_ind_cbegin(i), split_ind_itr, i_itr);
        v_itr = std::copy(val_cbegin(i), split_val_itr, v_itr);
      }
      indptr[i + 1] = indptr[i] + (i_itr - bak_itr);
    }
  }

  /// \brief split against secondary size
  /// \tparam IsSecond if \a true, then extract the second part
  /// \param[in] m size split against to
  /// \param[out] indptr index start array
  /// \param[out] indices index value array
  /// \param[out] vals values
  template <bool IsSecond>
  inline void _split(const size_type m, iarray_type &indptr,
                     iarray_type &indices, array_type &vals) const {
    if (!_ind_start.size()) return;
    indptr.resize(_ind_start.size());
    hilucsi_error_if(indptr.status() == DATA_UNDEF, "memory allocation failed");
    // const size_type n         = _ind_start.size() - 1;
    auto ind_first = _indices.cbegin();
    // size_type nnz(0);
    for (size_type i(0); i < _psize; ++i) {
      // store the position first
      auto find_itr = find_sorted(_ind_cbegin(i), _ind_cend(i), m).second;
      // nnz += IsSecond ? _ind_cend(i) - find_itr : find_itr - _ind_cbegin(i);
      indptr[i + 1] = find_itr - ind_first;
    }
    _split<IsSecond>(m, indptr.data() + 1, indptr, indices, vals);
  }

  /// \brief split and store to the counterpart type
  /// \tparam IsSecond if \a true, then extract the second part
  /// \param[in] m size split against to
  /// \param[in] n overall size of the secondary direction
  /// \param[in] start position start array
  /// \param[out] indptr index start array
  /// \param[out] indices index value array
  /// \param[out] vals values
  template <bool IsSecond>
  inline void _split_dual(const size_type m, const size_type n,
                          const index_type *start, iarray_type &indptr,
                          iarray_type &indices, array_type &vals) const {
    if (!_ind_start.size()) return;
    indptr.resize((IsSecond ? n - m : m) + 1);
    hilucsi_error_if(indptr.status() == DATA_UNDEF, "memory allocation failed");
    // initialize as zeros
    std::fill(indptr.begin(), indptr.end(), index_type(0));
    auto ind_first = _indices.cbegin();
    for (size_type i(0); i < _psize; ++i) {
      // store the position first
      auto find_itr = ind_first + start[i];
      if (IsSecond) {
        auto Itr = _ind_cend(i);
        for (auto itr = find_itr; itr != Itr; ++itr) ++indptr[*itr - m + 1];
      } else
        for (auto itr = _ind_cbegin(i); itr != find_itr; ++itr)
          ++indptr[*itr + 1];
    }
    const size_type N2 = indptr.size() - 1;
    for (size_type i(0); i < N2; ++i) indptr[i + 1] += indptr[i];

    if (!indptr[N2]) {
      std::fill(indptr.begin(), indptr.end(), index_type(0));
      return;
    }

    indices.resize(indptr[N2]);
    hilucsi_error_if(indices.status() == DATA_UNDEF,
                     "memory allocation failed");
    vals.resize(indptr[N2]);
    hilucsi_error_if(vals.status() == DATA_UNDEF, "memory allocation failed");

    auto val_first = _vals.cbegin();

    for (size_type i = 0; i < _psize; ++i) {
      auto split_ind_itr = start[i] + ind_first;
      if (IsSecond) {
        auto v_itr = start[i] + val_first;
        auto Itr   = _ind_cend(i);
        for (auto itr = split_ind_itr; itr != Itr; ++itr, ++v_itr) {
          const auto idx       = *itr - m;
          indices[indptr[idx]] = i;
          vals[indptr[idx]]    = *v_itr;
          ++indptr[idx];
        }
      } else {
        auto v_itr = val_cbegin(i);
        for (auto itr = _ind_cbegin(i); itr != split_ind_itr; ++itr, ++v_itr) {
          const auto idx       = *itr;
          indices[indptr[idx]] = i;
          vals[indptr[idx]]    = *v_itr;
          ++indptr[idx];
        }
      }
    }

    hilucsi_assert(indptr[N2] == indptr[N2 - 1], "fatal");

    index_type tmp(0);
    for (size_type i(0); i < N2; ++i) std::swap(indptr[i], tmp);
  }

  /// \brief split and store to the counterpart type
  /// \tparam IsSecond if \a true, then extract the second part
  /// \param[in] m size split against to
  /// \param[in] n overall size of the secondary direction
  /// \param[out] indptr index start array
  /// \param[out] indices index value array
  /// \param[out] vals values
  template <bool IsSecond>
  inline void _split_dual(const size_type m, const size_type n,
                          iarray_type &indptr, iarray_type &indices,
                          array_type &vals) const {
    if (!_ind_start.size()) return;
    iarray_type buf(_psize);
    auto        ind_first = _indices.cbegin();
    hilucsi_error_if(buf.status() == DATA_UNDEF, "memory allocation failed");
    for (size_type i = 0; i < _psize; ++i)
      buf[i] = find_sorted(_ind_cbegin(i), _ind_cend(i), m).second - ind_first;
    _split_dual<IsSecond>(m, n, buf.data(), indptr, indices, vals);
  }

  /// \brief construct permutated matrix
  /// \tparam DoValue if \a true, then also compute the values
  /// \tparam OnlyLeading if \a true, then will do a submatrix of leading block
  /// \param[in] p perm to origin mapping
  /// \param[in] qt origin to perm mapping
  /// \param[in] leading leading block size
  /// \param[out] indptr index pointer array
  /// \param[out] indices index array
  /// \param[out] vals numerical value array
  template <bool DoValue, bool OnlyLeading>
  inline void _compute_perm(const iarray_type &p, const iarray_type &qt,
                            const size_type leading, iarray_type &indptr,
                            iarray_type &indices, array_type &vals) const {
    const size_type m(p.size()), n(qt.size());
    hilucsi_error_if(m != _psize, "permutation out of bound");
    array_type      buf;
    const size_type psize = OnlyLeading ? leading : m;
    if (DoValue) {
      buf.resize(OnlyLeading ? leading : n);
      hilucsi_error_if(buf.status() == DATA_UNDEF, "memory allocation failed");
    }
    indptr.resize(psize + 1);
    hilucsi_error_if(indptr.status() == DATA_UNDEF, "memory allocation failed");
    // compute an upper bound of nnz, efficiently
    size_type nnz(0);
    for (size_type i(0); i < psize; ++i) nnz += _nnz(p[i]);
    if (!nnz) {
      std::fill(indptr.begin(), indptr.end(), index_type(0));
      return;
    }
    indptr.front() = 0;
    indices.resize(nnz);
    hilucsi_error_if(indices.status() == DATA_UNDEF,
                     "memory allocation failed");
    if (DoValue) {
      vals.resize(nnz);
      hilucsi_error_if(vals.status() == DATA_UNDEF, "memory allocation failed");
    }
    auto i_itr = indices.begin();
    auto v_itr = vals.begin();
    for (size_type i(0); i < psize; ++i) {
      auto last    = _ind_cend(p[i]);
      auto V_itr   = val_cbegin(p[i]);
      auto itr_bak = i_itr;
      for (auto itr = _ind_cbegin(p[i]); itr != last; ++itr, V_itr += DoValue) {
        if (!OnlyLeading) {
          *i_itr++ = qt[*itr];
          if (DoValue) buf[qt[*itr]] = *V_itr;
        } else {
          const size_type inv_idx = qt[*itr];
          if (inv_idx < psize) {
            *i_itr++ = inv_idx;
            if (DoValue) buf[inv_idx] = *V_itr;
          }
        }
      }
      // sort
      indptr[i + 1] = indptr[i] + (i_itr - itr_bak);
      std::sort(itr_bak, i_itr);
      if (DoValue)
        for (auto itr = itr_bak; itr != i_itr; ++itr, ++v_itr)
          *v_itr = buf[*itr];
    }
    indices.resize(i_itr - indices.begin());
    if (DoValue) vals.resize(v_itr - vals.begin());
  }

  /// \brief extract leading block structure
  /// \param[in] m leading block size
  /// \param[out] indptr index pointer array
  /// \param[out] indices index array
  inline void _extract_leading_struct(const size_type m, iarray_type &indptr,
                                      iarray_type &indices) const {
    indptr.resize(m + 1);
    hilucsi_error_if(indptr.status() == DATA_UNDEF, "memory allocation failed");
    indptr.front() = 0;
    size_type nz(0);
    for (size_type i(0); i < m; ++i) {
      auto last     = find_sorted(ind_cbegin(i), ind_cend(i), m).second;
      indptr[i + 1] = last - _indices.cbegin();
      nz += (last - ind_cbegin(i));
    }
    indices.resize(nz);
    if (nz) {
      hilucsi_error_if(indices.status() == DATA_UNDEF,
                       "memory allocation failed");
      indptr.front() = 0;
      auto itr       = indices.begin();
      for (size_type i(0); i < m; ++i) {
        auto last     = _indices.cbegin() + indptr[i + 1];
        itr           = std::copy(ind_cbegin(i), last, itr);
        indptr[i + 1] = indptr[i] + (last - ind_cbegin(i));
      }
    }
  }

  /// \brief extract leading block
  /// \param[in] m leading block size
  /// \param[out] indptr index pointer array
  /// \param[out] indices index array
  /// \param[out] vals numerical value array
  inline void _extract_leading(const size_type m, iarray_type &indptr,
                               iarray_type &indices, array_type &vals) const {
    _extract_leading_struct(m, indptr, indices);
    vals.resize(indices.size());
    hilucsi_error_if(vals.status() == DATA_UNDEF, "memory allocation failed");
    auto v_itr = vals.begin();
    for (size_type i(0); i < m; ++i)
      v_itr = std::copy_n(val_cbegin(i), indptr[i + 1] - indptr[i], v_itr);
  }

  /// \brief destroy the memory explicitly
  inline void _destroy() {
    iarray_type().swap(_ind_start);
    iarray_type().swap(_indices);
    array_type().swap(_vals);
    _psize = 0;
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
template <class ValueArray, class IndexArray>
inline void convert_storage(const IndexArray &i_ind_start,
                            const IndexArray &i_indices,
                            const ValueArray &i_vals, IndexArray &o_ind_start,
                            IndexArray &o_indices, ValueArray &o_vals) {
  typedef typename ValueArray::size_type  size_type;
  typedef typename IndexArray::value_type index_type;

  // both o_{indices,vals} are uninitialized arrays with size of input nnz
  hilucsi_error_if(i_indices.size() != (size_type)i_ind_start.back(),
                   "nnz %zd does not match that in start array %zd",
                   i_indices.size(), i_ind_start.back());
  hilucsi_error_if(i_indices.size() != i_vals.size(),
                   "nnz sizes (%zd,%zd) do not match between indices and vals",
                   i_indices.size(), i_vals.size());
  hilucsi_error_if(i_indices.size() != o_indices.size(),
                   "input nnz %zd does not match of that the output (%zd)",
                   i_indices.size(), o_indices.size());
  hilucsi_error_if(o_indices.size() != o_vals.size(),
                   "nnz sizes (%zd,%zd) do not match between indices and vals",
                   o_indices.size(), o_indices.size());

  const size_type o_n = o_ind_start.size() - 1u;
  const size_type i_n = i_ind_start.size() - 1u;

  // step 1, counts nnz per secondary direction, O(nnz)
  for (const auto p : i_indices) {
    hilucsi_assert((size_type)p < o_n, "%zd exceeds the bound %zd",
                   (size_type)p, o_n);
    ++o_ind_start[p + 1];
  }

  // step 2, build o_ind_start, O(n)
  for (size_type i = 0u; i < o_n; ++i) o_ind_start[i + 1] += o_ind_start[i];
  hilucsi_assert(o_ind_start.back() == i_ind_start.back(), "fatal issue");

  // step 3, build output indices and values, O(nnz)
  auto i_iiter = i_indices.cbegin();  // index iterator
  auto i_viter = i_vals.cbegin();     // value iterator
  for (size_type i = 0u; i < i_n; ++i) {
    const auto lnnz = i_ind_start[i + 1] - i_ind_start[i];
    const auto lend = i_iiter + lnnz;
    for (; i_iiter != lend; ++i_iiter, ++i_viter) {
      const auto j = *i_iiter;
      // get position
      const auto jj = o_ind_start[j];
      o_indices[jj] = i;
      o_vals[jj]    = *i_viter;
      // increment the counter
      ++o_ind_start[j];
    }
  }
  hilucsi_assert(o_ind_start[o_n] == o_ind_start[o_n - 1], "fatal issue");

  // final step, revert back to previous stage for output index start, O(n)
  index_type temp(0);
  for (size_type i = 0u; i < o_n; ++i) std::swap(temp, o_ind_start[i]);

  // NOTE we can create a buffer of size n, so that the last step will become
  // optional
}

}  // namespace internal

// forward decl
template <class ValueType, class IndexType>
class CCS;

/// \class CRS
/// \brief Compressed Row Storage (CRS) format for sparse matrices
/// \tparam ValueType numerical value type, e.g. \a double, \a float, etc
/// \tparam IndexType index type, e.g. \a int, \a long, etc
/// \ingroup ds
template <class ValueType, class IndexType>
class CRS : public internal::CompressedStorage<ValueType, IndexType> {
  typedef internal::CompressedStorage<ValueType, IndexType> _base;

 public:
  typedef ValueType                        value_type;   ///< value type
  typedef IndexType                        index_type;   ///< index type
  typedef typename _base::array_type       array_type;   ///< value array
  typedef typename _base::iarray_type      iarray_type;  ///< index array
  typedef typename _base::size_type        size_type;    ///< size type
  typedef typename _base::pointer          pointer;      ///< value pointer
  typedef typename _base::ipointer         ipointer;     ///< index pointer
  typedef CCS<ValueType, IndexType>        other_type;   ///< ccs type
  typedef typename _base::i_iterator       i_iterator;   ///< index iterator
  typedef typename _base::const_i_iterator const_i_iterator;
  ///< const index iterator
  typedef typename _base::v_iterator       v_iterator;  ///< value iterator
  typedef typename _base::const_v_iterator const_v_iterator;
  constexpr static bool                    ROW_MAJOR = true;  ///< row major

  /// \brief read a matrix from HILUCSI native binary file
  /// \param[in] filename file name
  /// \param[out] m if given, the leading block size is also returned
  /// \return A CRS matrix
  inline static CRS from_bin(const char *filename, size_type *m = nullptr) {
    CRS        crs;
    const auto b_size = crs.read_bin(filename);
    if (m) *m = b_size;
    return crs;
  }

  /// \brief read a matrix from HILUCSI native ASCII file
  /// \param[in] filename file name
  /// \param[out] m if given, the leading block size is also returned
  /// \return A CRS matrix
  inline static CRS from_ascii(const char *filename, size_type *m = nullptr) {
    CRS        crs;
    const auto b_size = crs.read_ascii(filename);
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
    if (ccs.nnz())
      internal::convert_storage<array_type, iarray_type>(
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
  /// \f$ \textrm{diag}(\mathbf{s})\mathbf{A}\f$; the overall complexity
  /// is in order of \f$\mathcal{O}(nnz)\f$
  template <class DiagArray>
  inline void scale_diag_left(const DiagArray &s) {
    hilucsi_error_if(row_start().size() > s.size() + 1u,
                     "row sizes do not match (%zd,%zd)",
                     row_start().size() - 1u, s.size());
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
  /// \f$\mathbf{A} \textrm{diag}(\mathbf{t})\f$; the overall complexity
  /// is in order of \f$\mathcal{O}(nnz)\f$
  template <class DiagArray>
  inline void scale_diag_right(const DiagArray &t) {
    hilucsi_error_if(_ncols > t.size(), "column sizes do not match (%zd,%zd)",
                     _ncols, t.size());
    auto v_itr = vals().begin();
    auto i_itr = col_ind().cbegin();
    for (auto last = col_ind().cend(); i_itr != last; ++i_itr, ++v_itr)
      *v_itr *= t[*i_itr];
  }

  /// \brief scale by two diagonal matrices from both lh and rh sides
  /// \tparam LeftDiagArray diagonal array type of left side
  /// \tparam RightDiagArray diagonal array type of right side
  /// \param[in] s diagonal matrix of left side
  /// \param[in] t diagonal matrix of right side
  ///
  /// Mathematically, this member function is to perform
  /// \f$\textrm{diag}(\mathbf{s}) \mathbf{A}
  /// \textrm{diag}(\mathbf{t})\f$
  template <class LeftDiagArray, class RightDiagArray>
  inline void scale_diags(const LeftDiagArray &s, const RightDiagArray &t) {
    scale_diag_left(s);
    scale_diag_right(t);
  }

  /// \brief matrix vector multiplication with different value type
  /// \tparam Vx other value type for \a x
  /// \tparam Vy other value type for \a y
  /// \param[in] x input array pointer
  /// \param[out] y output array pointer
  /// \warning User's responsibility to maintain valid pointers
  template <class Vx, class Vy>
  inline void mv_nt_low(const Vx *x, Vy *y) const {
    mv_nt_low(x, size_type(0), _psize, y);
    for (size_type i = 0u; i < _psize; ++i) {
      y[i]       = 0;
      auto v_itr = _base::val_cbegin(i);
      auto i_itr = col_ind_cbegin(i);
      for (auto last = col_ind_cend(i); i_itr != last; ++i_itr, ++v_itr) {
        hilucsi_assert(size_type(*i_itr) < _ncols, "%zd exceeds column size",
                       size_type(*i_itr));
        y[i] += *v_itr * x[*i_itr];
      }
    }
  }

  /// \brief matrix vector for kernel MT compatibility with different type
  /// \tparam Vx other value type for \a x
  /// \tparam Vy other value type for \a y
  /// \param[in] x input array pointer
  /// \param[in] istart index start
  /// \param[in] len local length
  /// \param[out] y output array pointer
  template <class Vx, class Vy>
  inline void mv_nt_low(const Vx *x, const size_type istart,
                        const size_type len, Vy *y) const {
    for (size_type i = istart, n = istart + len; i < n; ++i) {
      y[i]       = 0;
      auto v_itr = _base::val_cbegin(i);
      auto i_itr = col_ind_cbegin(i);
      for (auto last = col_ind_cend(i); i_itr != last; ++i_itr, ++v_itr) {
        hilucsi_assert(size_type(*i_itr) < _ncols, "%zd exceeds column size",
                       size_type(*i_itr));
        y[i] += *v_itr * x[*i_itr];
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
    hilucsi_error_if(nrows() != y.size() || ncols() != x.size(),
                     "matrix vector multiplication unmatched sizes!");
    mv_nt_low(x.data(), y.data());
  }

  /// \brief matrix vector multiplication for MT (CRS only)
  /// \tparam IArray input array type
  /// \tparam OArray output array type
  /// \param[in] x input array
  /// \param[in] istart index start
  /// \param[in] len local length
  /// \param[out] y output array
  /// \note Sizes must match
  template <class IArray, class OArray>
  inline void mv_nt(const IArray &x, const size_type istart,
                    const size_type len, OArray &y) const {
    hilucsi_error_if(nrows() != y.size() || ncols() != x.size(),
                     "matrix vector multiplication unmatched sizes!");
    hilucsi_error_if(istart >= nrows(), "%zd exceeds the row size %zd", istart,
                     nrows());
    hilucsi_error_if(istart + len > nrows(),
                     "out-of-bound pass-of-end range detected");
    mv_nt_low(x.data(), istart, len, y.data());
  }

  /// \brief matrix transpose vector multiplication with different type
  /// \tparam Vx other value type for \a x
  /// \tparam Vy other value type for \a y
  /// \param[in] x input array pointer
  /// \param[out] y output array pointer
  /// \warning User's responsibility to maintain valid pointers
  template <class Vx, class Vy>
  inline void mv_t_low(const Vx *x, Vy *y) const {
    if (!_psize) return;
    std::fill_n(y, ncols(), Vy(0));
    for (size_type i = 0u; i < _psize; ++i) {
      const auto temp  = x[i];
      auto       v_itr = _base::val_cbegin(i);
      auto       i_itr = col_ind_cbegin(i);
      for (auto last = col_ind_cend(i); i_itr != last; ++i_itr, ++v_itr) {
        const size_type j = *i_itr;
        hilucsi_assert(j < _ncols, "%zd exceeds column size", j);
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
  /// \note Compute \f$y=\mathbf{A}^Tx\f$
  template <class IArray, class OArray>
  inline void mv_t(const IArray &x, OArray &y) const {
    hilucsi_error_if(nrows() != x.size() || ncols() != y.size(),
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

  /// \brief Assume as a strict lower matrix and solve with forward sub
  /// \tparam RhsType Rhs type
  /// \param[in,out] y Input rhs, output solution
  /// \note Advanced use for preconditioner solve.
  /// \sa solve_as_strict_upper
  template <class RhsType>
  inline void solve_as_strict_lower(RhsType &y) const {
    // retrieve value type from rhs
    using value_type_ = typename std::remove_reference<decltype(y[0])>::type;
    for (size_type j(1); j < _psize; ++j) {
      auto itr   = col_ind_cbegin(j);
      auto v_itr = _base::val_cbegin(j);
      typename std::conditional<(sizeof(value_type_) < sizeof(value_type)),
                                value_type, value_type_>::type tmp(0);
      for (auto last = col_ind_cend(j); itr != last; ++itr, ++v_itr)
        tmp += *v_itr * y[*itr];
      y[j] -= tmp;
    }
  }

  /// \brief Assume as a strict upper matrix and solve with backward sub
  /// \tparam RhsType Rhs type
  /// \param[in,out] y Input rhs, output solution
  /// \note Advanced use for preconditioner solve.
  /// \sa solve_as_strict_lower
  template <class RhsType>
  inline void solve_as_strict_upper(RhsType &y) const {
    // retrieve value type from rhs
    using value_type_ = typename std::remove_reference<decltype(y[0])>::type;
    hilucsi_assert(_psize, "cannot be empty");
    for (size_type j = _psize - 1; j != 0u; --j) {
      const size_type j1    = j - 1;
      auto            itr   = col_ind_cbegin(j1);
      auto            v_itr = _base::val_cbegin(j1);
      typename std::conditional<(sizeof(value_type_) < sizeof(value_type)),
                                value_type, value_type_>::type tmp(0);
      for (auto last = col_ind_cend(j1); itr != last; ++itr, ++v_itr)
        tmp += *v_itr * y[*itr];
      y[j1] -= tmp;
    }
  }

  /// \brief read a native binary file
  /// \param[in] fname file name
  /// \return leading symmetric block size
  inline size_type read_bin(const char *fname) {
    return hilucsi::read_bin(fname, *this);
  }

  /// \brief write a native binary file
  /// \param[in] fname file name
  /// \param[in] m leading block size
  inline void write_bin(const char *fname, const size_type m = 0) const {
    hilucsi::write_bin(fname, *this, m);
  }

  /// \brief write a native ASCII file
  /// \param[in] fname file name
  /// \param[in] m leading block size
  inline void write_ascii(const char *fname, const size_type m = 0) const {
    hilucsi::write_ascii<true>(fname, row_start(), _ncols, col_ind(), vals(),
                               m);
  }

  /// \brief read data from an ASCII file
  /// \param[in] fname file name
  /// \return the leading symmetric block size
  inline size_type read_ascii(const char *fname) {
    bool        is_row, is_c;
    char        dtype;
    size_type   row, col, Nnz, m;
    iarray_type i_start, is;
    array_type  vs;
    std::tie(is_row, is_c, dtype, row, col, Nnz, m) =
        hilucsi::read_ascii(fname, i_start, is, vs);
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

  /// \brief split the matrix against column
  /// \tparam IsSecond if \a true, then the second half is returned
  /// \param[in] m column size
  ///
  /// If \a IsSecond is \a true, then the result matrix is \a A[:,m:]; otherwise
  /// the result matrix is \a A[:,0:m]
  template <bool IsSecond>
  inline CRS split(const size_type m) const {
    hilucsi_error_if(m > _ncols, "invalid split threshold");
    CRS B;
    B.resize(_psize, IsSecond ? _ncols - m : m);
    _base::template _split<IsSecond>(m, B.row_start(), B.col_ind(), B.vals());
    return B;
  }

  /// \brief split the matrix against column
  /// \tparam IsSecond if \a true, then the second half is returned
  /// \param[in] m column size
  /// \param[in] start starting position array, at least size of nrows()
  ///
  /// If \a IsSecond is \a true, then the result matrix is \a A[:,m:]; otherwise
  /// the result matrix is \a A[:,0:m]
  template <bool IsSecond>
  inline CRS split(const size_type m, const iarray_type &start) const {
    hilucsi_error_if(m > _ncols, "invalid split threshold");
    hilucsi_error_if(start.size() < _psize, "invalid starting position size");
    CRS B;
    B.resize(_psize, IsSecond ? _ncols - m : m);
    _base::template _split<IsSecond>(m, start.data(), B.row_start(),
                                     B.col_ind(), B.vals());
    return B;
  }

  /// \brief split the matrix against column
  /// \param[in] m column size
  inline std::pair<CRS, CRS> split(const size_type m) const {
    return std::make_pair(split<false>(m), split<true>(m));
  }

  /// \brief split the matrix against column
  /// \param[in] m column size
  /// \param[in] start starting position array, at least size of \a m
  inline std::pair<CRS, CRS> split(const size_type    m,
                                   const iarray_type &start) const {
    return std::make_pair(split<false>(m, start), split<true>(m, start));
  }

  /// \brief split and store to \ref CCS
  /// \tparam IsSecond if \a true, then the second half is returned
  /// \param[in] m column size
  template <bool IsSecond>
  inline other_type split_ccs(const size_type m) const {
    hilucsi_error_if(m > _ncols, "invalid split threshold");
    other_type B;
    B.resize(_psize, IsSecond ? _ncols - m : m);
    _base::template _split_dual<IsSecond>(m, _ncols, B.col_start(), B.row_ind(),
                                          B.vals());
    return B;
  }

  /// \brief split and store to \ref CCS
  /// \tparam IsSecond if \a true, then the second half is returned
  /// \param[in] m column size
  /// \param[in] start starting position array, at least size of nrows()
  template <bool IsSecond>
  inline other_type split_ccs(const size_type    m,
                              const iarray_type &start) const {
    hilucsi_error_if(m > _ncols, "invalid split threshold");
    hilucsi_error_if(start.size() < _psize, "invalid starting position size");
    other_type B;
    B.resize(_psize, IsSecond ? _ncols - m : m);
    _base::template _split_dual<IsSecond>(m, _ncols, start.data(),
                                          B.col_start(), B.row_ind(), B.vals());
    return B;
  }

  /// \brief split the matrix against column and store to \ref CCS
  /// \param[in] m column size
  inline std::pair<other_type, other_type> split_ccs(const size_type m) const {
    return std::make_pair(split_ccs<false>(m), split_ccs<true>(m));
  }

  /// \brief split the matrix against column and store to \ref CCS
  /// \param[in] m column size
  /// \param[in] start starting position array, at least size of nrows()
  inline std::pair<other_type, other_type> split_ccs(
      const size_type m, const iarray_type &start) const {
    return std::make_pair(split_ccs<false>(m, start),
                          split_ccs<true>(m, start));
  }

  /// \brief compute permutation CRS
  /// \param[in] p row permutation, from perm to origin
  /// \param[in] qt column permutation, from origin to perm
  /// \param[in] leading if > 0, then extract leading block
  /// \param[in] struct_only if \a false (default), then also compute values
  inline CRS compute_perm(const iarray_type &p, const iarray_type &qt,
                          const size_type leading     = 0,
                          const bool      struct_only = false) const {
    hilucsi_error_if(qt.size() != ncols(),
                     "invalid column permutation vector length");
    const bool do_all =
        leading == 0u || (leading == _psize && leading == _ncols);
    CRS A;
    if (do_all) {
      A.resize(_psize, _ncols);
      if (!struct_only)
        _base::template _compute_perm<true, false>(
            p, qt, leading, A.row_start(), A.col_ind(), A.vals());
      else
        _base::template _compute_perm<false, false>(
            p, qt, leading, A.row_start(), A.col_ind(), A.vals());
    } else {
      hilucsi_error_if(leading > std::min(_psize, _ncols),
                       "invalid leading block size");
      A.resize(leading, leading);
      if (!struct_only)
        _base::template _compute_perm<true, true>(p, qt, leading, A.row_start(),
                                                  A.col_ind(), A.vals());
      else
        _base::template _compute_perm<false, true>(
            p, qt, leading, A.row_start(), A.col_ind(), A.vals());
    }
    return A;
  }

  /// \brief extract leading block
  /// \param[in] m leading size
  inline CRS extract_leading(const size_type m) const {
    CRS B;
    B.resize(m, m);
    _base::_extract_leading(m, B.row_start(), B.col_ind(), B.vals());
    return B;
  }

  /// \brief explicit destroy the memory
  inline void destroy() {
    _base::_destroy();
    _ncols = 0;
  }

 protected:
  size_type _ncols;     ///< number of columns
  using _base::_psize;  ///< number of rows (primary entries)
};

/// \brief wrap user data
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
inline CRS<ValueType, IndexType> wrap_crs(
    const typename CRS<ValueType, IndexType>::size_type nrows,
    const typename CRS<ValueType, IndexType>::size_type ncols,
    IndexType *row_start, IndexType *col_ind, ValueType *vals,
    bool check = true, bool help_sort = false) {
  using return_type          = CRS<ValueType, IndexType>;
  using size_type            = typename CRS<ValueType, IndexType>::size_type;
  constexpr static bool WRAP = true;
  static_assert(std::is_integral<IndexType>::value, "must be integer");

  // run time

  return_type mat(nrows, ncols, const_cast<IndexType *>(row_start),
                  const_cast<IndexType *>(col_ind),
                  const_cast<ValueType *>(vals), WRAP);

  if (check) {
    if (row_start[0] != 0)
      hilucsi_error("first entry of row_start does not agree with 0.");
    Array<ValueType> buf;
    for (size_type i = 0u; i < nrows; ++i) {
      if (!mat.nnz_in_row(i)) {
        hilucsi_warning("row %zd is empty!", i);
        continue;
      }
      auto last = mat.col_ind_cend(i), first = mat.col_ind_cbegin(i);
      auto itr = std::is_sorted_until(
          first, last,
          [](const IndexType i, const IndexType j) { return i <= j; });
      if (itr != last) {
        if (!help_sort)
          hilucsi_error(
              "%zd row is not sorted, the checking failed at entry %td, run "
              "with help_sort=true",
              i, itr - first);
        else {
          buf.resize(mat.ncols());
          if (buf.status() == DATA_UNDEF)
            hilucsi_error("memory allocation failed!");
          // load values, note that we can just load [itr,last), but we want to
          // loop through all indices to ensure they are bounded
          auto v_itr = mat.val_cbegin(i);
          for (itr = first; itr != last; ++itr, ++v_itr) {
            hilucsi_error_if(*itr < 0 || size_type(*itr) >= mat.ncols(),
                             "%zd exceeds column size %zd", size_type(*itr),
                             mat.ncols());
            buf[*itr] = *v_itr;
          }
          std::sort(mat.col_ind_begin(i), mat.col_ind_end(i));
          auto vv_itr = mat.val_begin(i);
          last        = mat.col_ind_cend(i);
          first       = mat.col_ind_cbegin(i);
          for (itr = first; itr != last; ++itr, ++vv_itr) *vv_itr = buf[*itr];
        }
      } else
        hilucsi_error_if(size_type(*(last - 1)) >= mat.ncols(),
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
inline CRS<ValueType, IndexType> wrap_crs(
    const typename CRS<ValueType, IndexType>::size_type nrows,
    const typename CRS<ValueType, IndexType>::size_type ncols,
    const IndexType *row_start, const IndexType *col_ind, const ValueType *vals,
    bool check = true, bool help_sort = false) {
  return wrap_crs<false>(nrows, ncols, row_start, col_ind, vals, check,
                         help_sort);
}

/// \class CCS
/// \brief Compressed Column Storage (CCS) format for sparse matrices
/// \tparam ValueType numerical value type, e.g. \a double, \a float, etc
/// \tparam IndexType index type, e.g. \a int, \a long, etc
/// \ingroup ds
template <class ValueType, class IndexType>
class CCS : public internal::CompressedStorage<ValueType, IndexType> {
  typedef internal::CompressedStorage<ValueType, IndexType> _base;

 public:
  typedef ValueType                        value_type;   ///< value type
  typedef IndexType                        index_type;   ///< index type
  typedef typename _base::array_type       array_type;   ///< value array
  typedef typename _base::iarray_type      iarray_type;  ///< index array
  typedef typename _base::size_type        size_type;    ///< size type
  typedef typename _base::pointer          pointer;      ///< value pointer
  typedef typename _base::ipointer         ipointer;     ///< index pointer
  typedef CRS<ValueType, IndexType>        other_type;   ///< crs type
  typedef typename _base::i_iterator       i_iterator;   ///< index iterator
  typedef typename _base::const_i_iterator const_i_iterator;
  ///< const index iterator
  typedef typename _base::v_iterator       v_iterator;  ///< value iterator
  typedef typename _base::const_v_iterator const_v_iterator;
  ///< const value iterator
  constexpr static bool ROW_MAJOR = false;  ///< column major

  /// \brief read from a native HILUCSI binary file
  /// \param[in] filename file name
  /// \param[out] m if given, then it will store the leading symmetric size
  /// \return A CCS matrix
  inline static CCS from_bin(const char *filename, size_type *m = nullptr) {
    CCS        ccs;
    const auto b_size = ccs.read_bin(filename);
    if (m) *m = b_size;
    return ccs;
  }

  /// \brief read from a native HILUCSI ASCII file
  /// \param[in] filename file name
  /// \param[out] m if given, then it will store the leading symmetric size
  /// \return A CCS matrix
  inline static CCS from_ascii(const char *filename, size_type *m = nullptr) {
    CCS        ccs;
    const auto b_size = ccs.read_ascii(filename);
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
    if (crs.nnz())
      internal::convert_storage<array_type, iarray_type>(
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
  /// \f$ \textrm{diag}(\mathbf{s})\mathbf{A}\f$; the overall complexity
  /// is in order of \f$\mathcal{O}(nnz)\f$
  template <class DiagArray>
  inline void scale_diag_left(const DiagArray &s) {
    hilucsi_error_if(_nrows > s.size(), "row sizes do not match (%zd,%zd)",
                     _nrows, s.size());
    auto v_itr = vals().begin();
    auto i_itr = row_ind().cbegin();
    for (auto last = row_ind().cend(); i_itr != last; ++i_itr, ++v_itr)
      *v_itr *= s[*i_itr];
  }

  /// \brief scale by a diagonal matrix from right
  /// \tparam DiagArray diagonal array type
  /// \param[in] t diagonal matrix multiplying from left-hand side
  /// \sa scale_diag_left
  ///
  /// Mathematically, this member function is to perform
  /// \f$\mathbf{A} \textrm{diag}(\mathbf{s})\f$; the overall complexity
  /// is in order of \f$\mathcal{O}(nnz)\f$
  template <class DiagArray>
  inline void scale_diag_right(const DiagArray &t) {
    hilucsi_error_if(col_start().size() > t.size() + 1u,
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
  /// \f$\textrm{diag}(\mathbf{s}) \mathbf{A}
  /// \textrm{diag}(\mathbf{t})\f$
  template <class LeftDiagArray, class RightDiagArray>
  inline void scale_diags(const LeftDiagArray &s, const RightDiagArray &t) {
    scale_diag_left(s);
    scale_diag_right(t);
  }

  /// \brief matrix vector multiplication with different value type
  /// \tparam Vx other value type for \a x
  /// \tparam Vy other value type for \a y
  /// \param[in] x input array pointer
  /// \param[out] y output array pointer
  /// \warning User's responsibilty to ensure valid pointers
  template <class Vx, class Vy>
  inline void mv_nt_low(const Vx *x, Vy *y) const {
    if (!_psize) return;
    std::fill_n(y, nrows(), Vy(0));
    for (size_type i = 0u; i < _psize; ++i) {
      const auto temp  = x[i];
      auto       v_itr = _base::val_cbegin(i);
      auto       i_itr = row_ind_cbegin(i);
      for (auto last = row_ind_cend(i); i_itr != last; ++i_itr, ++v_itr) {
        hilucsi_assert(size_type(*i_itr) < _nrows, "%zd exceeds the size bound",
                       size_type(*i_itr));
        y[*i_itr] += temp * *v_itr;
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
    hilucsi_error_if(nrows() != y.size() || ncols() != x.size(),
                     "matrix vector unmatched sizes!");
    mv_nt_low(x.data(), y.data());
  }

  /// \brief matrix transpose vector multiplication with different type
  /// \tparam Vx other value type for \a x
  /// \tparam Vy other value type for \a y
  /// \param[in] x input array pointer
  /// \param[out] y output array pointer
  /// \warning User's responsibilty to ensure valid pointers
  template <class Vx, class Vy>
  inline void mv_t_low(const Vx *x, Vy *y) const {
    for (size_type i = 0u; i < _psize; ++i) {
      y[i]       = 0;
      auto v_itr = _base::val_cbegin(i);
      auto i_itr = row_ind_cbegin(i);
      for (auto last = row_ind_cend(i); i_itr != last; ++i_itr, ++v_itr) {
        hilucsi_assert(size_type(*i_itr) < _nrows, "%zd exceeds the size bound",
                       size_type(*i_itr));
        y[i] += x[*i_itr] * *v_itr;
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
    hilucsi_error_if(nrows() != x.size() || ncols() != y.size(),
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

  /// \brief Assume as a strict lower matrix and solve with forward sub
  /// \tparam RhsType Rhs type
  /// \param[in,out] y Input rhs, output solution
  /// \note Advanced use for preconditioner solve.
  /// \sa solve_as_strict_upper
  template <class RhsType>
  inline void solve_as_strict_lower(RhsType &y) const {
    for (size_type j(0); j < _psize; ++j) {
      const auto y_j   = y[j];
      auto       itr   = row_ind_cbegin(j);
      auto       v_itr = _base::val_cbegin(j);
      for (auto last = row_ind_cend(j); itr != last; ++itr, ++v_itr) {
        hilucsi_assert(size_type(*itr) < _psize, "%zd exceeds system size %zd",
                       size_type(*itr), _psize);
        y[*itr] -= *v_itr * y_j;
      }
    }
  }

  /// \brief Assume as a strict upper matrix and solve with backward sub
  /// \tparam RhsType Rhs type
  /// \param[in,out] y Input rhs, output solution
  /// \note Advanced use for preconditioner solve.
  /// \sa solve_as_strict_lower
  template <class RhsType>
  inline void solve_as_strict_upper(RhsType &y) const {
    using rev_iterator   = std::reverse_iterator<const_i_iterator>;
    using rev_v_iterator = std::reverse_iterator<const_v_iterator>;

    for (size_type j(_psize - 1); j != 0u; --j) {
      const auto y_j   = y[j];
      auto       itr   = rev_iterator(row_ind_cend(j));
      auto       v_itr = rev_v_iterator(_base::val_cend(j));
      for (auto last = rev_iterator(row_ind_cbegin(j)); itr != last;
           ++itr, ++v_itr)
        y[*itr] -= *v_itr * y_j;
    }
  }

  /// \brief read a native HILUCSI binary file
  /// \param[in] fname file name
  /// \return leading symmetric block size
  inline size_type read_bin(const char *fname) {
    return hilucsi::read_bin(fname, *this);
  }

  /// \brief write to a native HILUCSI binary file
  /// \param[in] fname file name
  /// \param[in] m leading block size
  inline void write_bin(const char *fname, const size_type m = 0) const {
    hilucsi::write_bin(fname, *this, m);
  }

  /// \brief write a native ASCII file
  /// \param[in] fname file name
  /// \param[in] m leading block size
  inline void write_ascii(const char *fname, const size_type m = 0) const {
    hilucsi::write_ascii<false>(fname, col_start(), _nrows, row_ind(), vals(),
                                m);
  }

  /// \brief read data from an ASCII file
  /// \param[in] fname file name
  /// \return the leading symmetric block size
  inline size_type read_ascii(const char *fname) {
    bool        is_row, is_c;
    char        dtype;
    size_type   row, col, Nnz, m;
    iarray_type i_start, is;
    array_type  vs;
    std::tie(is_row, is_c, dtype, row, col, Nnz, m) =
        hilucsi::read_ascii(fname, i_start, is, vs);
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

  /// \brief split against row
  /// \tparam IsSecond if \a true, then the offset part is returned
  /// \param[in] m splitted row size
  template <bool IsSecond>
  inline CCS split(const size_type m) const {
    hilucsi_error_if(m > _nrows, "invalid row size");
    CCS B;
    B.resize(IsSecond ? _nrows - m : m, _psize);
    _base::template _split<IsSecond>(m, B.col_start(), B.row_ind(), B.vals());
    return B;
  }

  /// \brief split against row
  /// \tparam IsSecond if \a true, then the offset part is returned
  /// \param[in] m splitted row size
  /// \param[in] start starting position array
  template <bool IsSecond>
  inline CCS split(const size_type m, const iarray_type &start) const {
    hilucsi_error_if(m > _nrows, "invalid row size");
    hilucsi_error_if(start.size() < _psize, "invalid starting position array");
    CCS B;
    B.resize(IsSecond ? _nrows - m : m, _psize);
    _base::template _split<IsSecond>(m, start.data(), B.col_start(),
                                     B.row_ind(), B.vals());
    return B;
  }

  /// \brief split against row
  /// \param[in] m splitted row size
  inline std::pair<CCS, CCS> split(const size_type m) const {
    return std::make_pair(split<false>(m), split<true>(m));
  }

  /// \brief split against row
  /// \param[in] m splitted row size
  /// \param[in] start starting position array
  inline std::pair<CCS, CCS> split(const size_type    m,
                                   const iarray_type &start) const {
    return std::make_pair(split<false>(m, start), split<true>(m, start));
  }

  /// \brief split against row and store in \ref CRS
  /// \tparam IsSecond if \a true, then the offset part is returned
  /// \param[in] m splitted row size
  template <bool IsSecond>
  inline other_type split_crs(const size_type m) const {
    hilucsi_error_if(m > _nrows, "invalid row size");
    other_type B;
    B.resize(IsSecond ? _nrows - m : m, _psize);
    _base::template _split_dual<IsSecond>(m, _nrows, B.row_start(), B.col_ind(),
                                          B.vals());
    return B;
  }

  /// \brief split against row and store in \ref CRS
  /// \tparam IsSecond if \a true, then the offset part is returned
  /// \param[in] m splitted row size
  /// \param[in] start starting position array
  template <bool IsSecond>
  inline other_type split_crs(const size_type    m,
                              const iarray_type &start) const {
    hilucsi_error_if(m > _nrows, "invalid row size");
    hilucsi_error_if(start.size() < _psize, "invalid starting position array");
    other_type B;
    B.resize(IsSecond ? _nrows - m : m, _psize);
    _base::template _split_dual<IsSecond>(m, _nrows, start.data(),
                                          B.row_start(), B.col_ind(), B.vals());
    return B;
  }

  /// \brief split against row and store in \ref CRS
  /// \param[in] m splitted row size
  inline std::pair<other_type, other_type> split_crs(const size_type m) const {
    return std::make_pair(split_crs<false>(m), split_crs<true>(m));
  }

  /// \brief split against row and store in \ref CRS
  /// \param[in] m splitted row size
  /// \param[in] start starting position array
  inline std::pair<other_type, other_type> split_crs(
      const size_type m, const iarray_type &start) const {
    return std::make_pair(split_crs<false>(m, start),
                          split_crs<true>(m, start));
  }

  /// \brief compute permutation CCS
  /// \param[in] pt row permutation, from origin to perm
  /// \param[in] q column permutation, from perm to origin
  /// \param[in] leading if > 0, then extract leading block
  /// \param[in] struct_only if \a false (default), then also compute values
  inline CCS compute_perm(const iarray_type &pt, const iarray_type &q,
                          const size_type leading     = 0,
                          const bool      struct_only = false) const {
    hilucsi_error_if(pt.size() != nrows(),
                     "invalid row permutation vector length");
    const bool do_all =
        leading == 0u || (leading == _psize && leading == _nrows);
    CCS A;
    if (do_all) {
      A.resize(_nrows, _psize);
      if (!struct_only)
        _base::template _compute_perm<true, false>(
            q, pt, leading, A.col_start(), A.row_ind(), A.vals());
      else
        _base::template _compute_perm<false, false>(
            q, pt, leading, A.col_start(), A.row_ind(), A.vals());
    } else {
      hilucsi_error_if(leading > std::min(_psize, _nrows),
                       "invalid leading block size");
      A.resize(leading, leading);
      if (!struct_only)
        _base::template _compute_perm<true, true>(q, pt, leading, A.col_start(),
                                                  A.row_ind(), A.vals());
      else
        _base::template _compute_perm<false, true>(
            q, pt, leading, A.col_start(), A.row_ind(), A.vals());
    }
    return A;
  }

  /// \brief extract leading block
  /// \param[in] m leading block size
  inline CCS extract_leading(const size_type m) const {
    CCS B;
    B.resize(m, m);
    _base::_extract_leading(m, B.col_start(), B.row_ind(), B.vals());
    return B;
  }

  /// \brief explicit destroy the data
  inline void destroy() {
    _base::_destroy();
    _nrows = 0;
  }

 protected:
  size_type _nrows;     ///< number of rows
  using _base::_psize;  ///< number of columns (primary entries)
};

}  // namespace hilucsi

#endif  // _HILUCSI_DS_COMPRESSEDSTORAGE_HPP
