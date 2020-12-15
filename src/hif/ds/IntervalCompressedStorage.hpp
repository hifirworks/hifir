///////////////////////////////////////////////////////////////////////////////
//                  This file is part of HIF project                         //
///////////////////////////////////////////////////////////////////////////////

/*!
 * \file hif/ds/IntervalCompressedStorage.hpp
 * \brief Interval-based compressed storages
 * \note This data structure is designed to be used only with assembled static
 *       matrices.
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

#ifndef _HIF_DS_INTERVALCOMPRESSEDSTORAGE_HPP
#define _HIF_DS_INTERVALCOMPRESSEDSTORAGE_HPP

#include <cstdint>
#include <memory>

#include "hif/ds/CompressedStorage.hpp"

namespace hif {

namespace internal {

/*!
 * \addtogroup ds
 * @{
 */

/// \brief Determine total number of intervals given a compressed storage
/// \tparam MaxInterval Maximum interval upper bound
/// \tparam CsType Compressed storage, i.e., \ref CRS or \ref CCS
/// \tparam IndexArray Index array type, i.e., \ref Array
/// \param[in] A Input compressed matrix
/// \param[out] itrv_start output interval starting position array
template <std::size_t MaxInterval, class CsType, class IndexArray>
inline void determine_nitrvs(const CsType &A, IndexArray &itrv_start) {
  using size_type = typename CsType::size_type;
  // resize
  itrv_start.resize(A.ind_start().size());
  if (!itrv_start.size()) return;

  // the first element should be 1
  itrv_start.front() = 0;

  const auto n = itrv_start.size() - 1;

  for (size_type i(0); i < n; ++i) {
    if (!A.nnz_in_primary(i)) {
      // empty row/column
      itrv_start[i + 1] = itrv_start[i];
      continue;
    }
    auto      iter = A.ind_cbegin(i) + 1, iter_last = A.ind_cend(i);
    size_type nitrvs(1);  // NOTE: always start with 1
    size_type len(1);
    while (iter != iter_last) {
      if (*(iter - 1) + 1 != *iter || len > MaxInterval) {
        ++nitrvs;  // increment number of intervals
        len = 0;   // reset length
      }
      ++iter;
      ++len;
    }
    itrv_start[i + 1] = itrv_start[i] + nitrvs;
  }
}

/// \brief Build the interval arrays
/// \tparam CsType Compressed storage, i.e., \ref CRS or \ref CCS
/// \tparam IndexArray Index array type, i.e., \ref Array
/// \tparam IntervalArray Interval array type, i.e., \ref Array
/// \param[in] A Input compressed matrix
/// \param[in] itrv_start Input interval startingposition array
/// \param[out] ind_start Output starting index of each interval
/// \param[out] lens Output lengths of each interval, as to \a ind_start
template <class CsType, class IndexArray, class IntervalArray>
inline void build_intervals(const CsType &A, const IndexArray &itrv_start,
                            IndexArray &ind_start, IntervalArray &lens) {
  using size_type = typename CsType::size_type;

  const size_type total_nitrvs = itrv_start.back();
  ind_start.resize(total_nitrvs);
  lens.resize(total_nitrvs);

  const auto n = itrv_start.size() - 1;

  auto ind_iter = ind_start.begin();
  auto len_iter = lens.begin();

  for (size_type i(0); i < n; ++i) {
    if (!A.nnz_in_primary(i)) continue;
    auto iter = A.ind_cbegin(i), iter_last = A.ind_cend(i);
    *ind_iter = *iter++;  // increment iter here
    *len_iter = 1;
    while (iter != iter_last) {
      // XXX: !len:-> overflow back to zero
      if (*(iter - 1) + 1 != *iter || !*len_iter) {
        // get a breaking point
        *++ind_iter = *iter;
        if (!*len_iter) --(*len_iter);
        *++len_iter = 0;
      }
      ++iter;
      ++(*len_iter);
    }
    ++ind_iter;
    ++len_iter;
    hif_assert(ind_iter == ind_start.begin() + itrv_start[i + 1], "fatal, %zd",
               size_type((ind_start.begin() + itrv_start[i + 1]) - ind_iter));
    hif_assert(len_iter == lens.begin() + itrv_start[i + 1], "fatal, %zd",
               size_type((lens.begin() + itrv_start[i + 1]) - len_iter));
  }
}

/// \brief Analyzing the cost ratio of interval to classical storages
/// \tparam INDEX_SIZE Index size
/// \tparam VALUE_SIZE Value type size
/// \tparam INTERVAL_SIZE Interval type size
/// \param[in] n Number of rows/columns
/// \param[in] nnz Total number of nonzeros
/// \param[in] nitrvs Total number of itervals
/// \return double
/// \note If the result is greater than 1, it means we are wasting memory by
///       using interval storage
template <std::size_t INDEX_SIZE, std::size_t VALUE_SIZE,
          std::size_t INTERVAL_SIZE>
inline double analyze_storage_cost_ratio(const std::size_t n,
                                         const std::size_t nnz,
                                         const std::size_t nitrvs) {
  // cost of classical
  const double classical_size =
      (double)nnz * (INDEX_SIZE + VALUE_SIZE) + (double)(n + 1) * INDEX_SIZE;
  // cost of interval
  const double interval_size = (double)nnz * VALUE_SIZE +
                               (double)nitrvs * (INDEX_SIZE + INTERVAL_SIZE) +
                               2.0 * (n + 1) * INDEX_SIZE;
  return classical_size != 0.0 ? interval_size / classical_size : 0.0;
}

/*!
 * @}
 */

}  // namespace internal

/*!
 * \addtogroup ds
 * @{
 */

/// \class IntervalCRS
/// \brief Interval-based CRS representation
/// \tparam ValueType numerical value type, e.g. \a double, \a float, etc
/// \tparam IndexType index type, e.g. \a int, \a long, etc
/// \tparam IntervalType Interval type, default is \a std::uint8_t
template <class ValueType, class IndexType, class IntervalType = std::uint8_t>
class IntervalCRS {
  // NOTE we use CRS as trait to derive consistent public types
  using _base_type = CRS<ValueType, IndexType>;  ///< base type
 public:
  using value_type       = typename _base_type::value_type;   ///< value type
  using index_type       = typename _base_type::index_type;   ///< index type
  using array_type       = typename _base_type::array_type;   ///< value array
  using iarray_type      = typename _base_type::iarray_type;  ///< index array
  using size_type        = typename _base_type::size_type;    ///< size
  using pointer          = typename _base_type::pointer;      ///< pointer type
  using ipointer         = typename _base_type::ipointer;     ///< index pointer
  using i_iterator       = typename _base_type::i_iterator;  ///< index iterator
  using const_i_iterator = typename _base_type::const_i_iterator;
  ///< const index iterator
  using v_iterator       = typename _base_type::v_iterator;  ///< value iterator
  using const_v_iterator = typename _base_type::const_v_iterator;
  ///< const value iterator
  using crs_type      = _base_type;                       ///< CRS type
  using ccs_type      = typename _base_type::other_type;  ///< CCS type
  using interval_type = IntervalType;                     ///< interval type
  constexpr static bool          ROW_MAJOR    = true;     ///< row major flag
  constexpr static interval_type MAX_INTERVAL = static_cast<interval_type>(-1);
  ///< max interval length, cast from -1
  static_assert(std::is_unsigned<interval_type>::value,
                "interval type must be unsigned");

  /// \brief default constructor
  IntervalCRS() = default;

  /// \brief allow implicit convert from a rvalue reference of CRS
  /// \param[in,out] A CRS of rvalue reference
  /// \param[in] smart_convert (optional) Perform smart converting
  IntervalCRS(crs_type &&A, const bool smart_convert = true)
      : _nrows(A.nrows()), _ncols(A.ncols()) {
    internal::determine_nitrvs<MAX_INTERVAL>(A, _itrv_start);
    bool convert_to_interval = true;
    if (smart_convert && _itrv_start.size())
      convert_to_interval = size_type(2) * _itrv_start.back() <= A.nnz();
    // NOTE, we can check number of intervals to do smart_convert
    if (convert_to_interval) {
      internal::build_intervals(A, _itrv_start, _col_ind_start, _col_len);
      // destroy A
      _row_start = std::move(A.row_start());
      _vals      = std::move(A.vals());
      A.destroy();
      _ref.reset();
    } else {
      _ref = std::make_shared<crs_type>(std::move(A));
      iarray_type().swap(_itrv_start);  // destroy allocated space
    }
  }

  /// \brief convert from a CCS matrix
  /// \param[in] A CCS input
  /// \param[in] smart_convert (optional) Perform smart converting
  IntervalCRS(const ccs_type &A, const bool smart_convert = true)
      : IntervalCRS(crs_type(A), smart_convert) {}

  /// \brief default copy constructor
  IntervalCRS(const IntervalCRS &) = default;

  /// \brief default move constructor
  IntervalCRS(IntervalCRS &&) = default;

  /// \brief default assignment
  IntervalCRS &operator=(const IntervalCRS &) = default;

  /// \brief default move assignment
  IntervalCRS &operator=(IntervalCRS &&) = default;

  /// \brief get number of rows
  inline size_type nrows() const { return _nrows; }

  /// \brief get number of columns
  inline size_type ncols() const { return _ncols; }

  /// \brief get the number of nnz
  inline size_type nnz() const { return _ref ? _ref->nnz() : _vals.size(); }

  /// \brief get the number of intervals
  inline size_type nitrvs() const {
    return _ref ? size_type(0) : _col_ind_start.size();
  }

  /// \brief analyze storage ratio from interval to classical
  /// \note Results greater than 1 means we are wasting storage by using
  ///       interval based storage
  inline double storage_cost_ratio() const {
    if (_ref) return 1.0;
    return internal::analyze_storage_cost_ratio<
        sizeof(index_type), sizeof(value_type), sizeof(interval_type)>(
        nrows(), nnz(), nitrvs());
  }

  /// \brief Check if converted to interval-based structure
  inline bool converted() const { return !_ref; }

  /// \brief matrix vector multiplication (low-level API)
  /// \tparam Vx other value type for \a x
  /// \tparam Vy other value type for \a y
  /// \param[in] x input array pointer
  /// \param[in] istart index start
  /// \param[in] len local length
  /// \param[out] y output array pointer
  /// \warning User's responsibility to maintain valid pointers
  template <class Vx, class Vy>
  inline void mv_nt_low(const Vx *x, const size_type istart,
                        const size_type len, Vy *y) const {
    if (_ref)
      _ref->mv_nt_low(x, istart, len, y);
    else {
      const auto n = istart + len;
      for (size_type i(istart); i < n; ++i) {
        auto val_i    = _vals.cbegin() + _row_start[i];
        auto len_iter = _col_len.cbegin() + _itrv_start[i];
        auto last     = _col_ind_start.cbegin() + _itrv_start[i + 1];
        typename std::conditional<(sizeof(Vy) > sizeof(value_type)), Vy,
                                  value_type>::type tmp(0);
        // loop thru all intervals
        for (auto iter = _col_ind_start.cbegin() + _itrv_start[i]; iter != last;
             ++iter, ++len_iter) {
          const auto          idx = *iter;
          const interval_type m   = *len_iter;
          // for each interval, loop thru the consecutive range
          for (interval_type j(0); j < m; ++j, ++val_i)
            tmp += *val_i * x[idx + j];
        }
        y[i] = tmp;
      }
    }
  }

  /// \brief matrix vector multiplication with different value type
  /// \tparam Vx other value type for \a x
  /// \tparam Vy other value type for \a y
  /// \param[in] x input array pointer
  /// \param[out] y output array pointer
  /// \warning User's responsibility to maintain valid pointers
  template <class Vx, class Vy>
  inline void mv_nt_low(const Vx *x, Vy *y) const {
    mv_nt_low(x, 0, _nrows, y);
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
    hif_error_if(nrows() != y.size() || ncols() != x.size(),
                 "matrix vector multiplication unmatched sizes!");
    hif_error_if(istart >= nrows(), "%zd exceeds the row size %zd", istart,
                 nrows());
    hif_error_if(istart + len > nrows(),
                 "out-of-bound pass-of-end range detected");
    mv_nt_low(x.data(), istart, len, y.data());
  }

  /// \brief matrix vector multiplication
  /// \tparam IArray input array type
  /// \tparam OArray output array type
  /// \param[in] x input array
  /// \param[out] y output array
  /// \note Sizes must match
  template <class IArray, class OArray>
  inline void mv_nt(const IArray &x, OArray &y) const {
    hif_error_if(nrows() != y.size() || ncols() != x.size(),
                 "matrix vector multiplication unmatched sizes!");
    mv_nt_low(x.data(), y.data());
  }

  /// \brief matrix transpose vector multiplication with different type
  /// \tparam Vx other value type for \a x
  /// \tparam Vy other value type for \a y
  /// \param[in] x input array pointer
  /// \param[out] y output array pointer
  /// \warning User's responsibility to maintain valid pointers
  template <class Vx, class Vy>
  inline void mv_t_low(const Vx *x, Vy *y) const {
    if (!_nrows) return;
    if (_ref)
      _ref->mv_t_low(x, y);
    else {
      std::fill_n(y, _ncols, Vy(0));
      for (size_type i(0); i < _nrows; ++i) {
        const auto temp     = x[i];
        auto       val_i    = _vals.cbegin() + _row_start[i];
        auto       len_iter = _col_len.cbegin() + _itrv_start[i];
        auto       last     = _col_ind_start.cbegin() + _itrv_start[i + 1];
        // loop thru all intervals
        for (auto iter = _col_ind_start.cbegin() + _itrv_start[i]; iter != last;
             ++iter, ++len_iter) {
          const auto          idx = *iter;
          const interval_type m   = *len_iter;
          // for each interval, loop thru the consecutive range
          for (interval_type j(0); j < m; ++j, ++val_i)
            y[idx + j] += *val_i * temp;
        }
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
    hif_error_if(nrows() != x.size() || ncols() != y.size(),
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
    if (_ref) {
      _ref->solve_as_strict_lower(y);
      return;
    }
    // retrieve value type from rhs
    using value_type_ = typename std::remove_reference<decltype(y[0])>::type;
    for (size_type j(1); j < _nrows; ++j) {
      auto v_itr = _vals.cbegin() + _row_start[j];
      typename std::conditional<(sizeof(value_type_) < sizeof(value_type)),
                                value_type, value_type_>::type tmp(0);
      auto len_iter = _col_len.cbegin() + _itrv_start[j];
      auto last     = _col_ind_start.cbegin() + _itrv_start[j + 1];
      // loop thru all intervals
      for (auto iter = _col_ind_start.cbegin() + _itrv_start[j]; iter != last;
           ++iter, ++len_iter) {
        const auto          idx = *iter;
        const interval_type m   = *len_iter;
        // for each interval, loop thru the consecutive range
        for (interval_type i(0); i < m; ++i, ++v_itr)
          tmp += *v_itr * y[idx + i];
      }
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
    if (_ref) {
      _ref->solve_as_strict_upper(y);
      return;
    }
    // retrieve value type from rhs
    using value_type_ = typename std::remove_reference<decltype(y[0])>::type;
    hif_assert(_nrows, "cannot be empty");
    for (size_type j = _nrows - 1; j != 0u; --j) {
      const size_type j1    = j - 1;
      auto            v_itr = _vals.cbegin() + _row_start[j1];
      typename std::conditional<(sizeof(value_type_) < sizeof(value_type)),
                                value_type, value_type_>::type tmp(0);
      auto len_iter = _col_len.cbegin() + _itrv_start[j1];
      auto last     = _col_ind_start.cbegin() + _itrv_start[j];
      // loop thru all intervals
      for (auto iter = _col_ind_start.cbegin() + _itrv_start[j1]; iter != last;
           ++iter, ++len_iter) {
        const auto          idx = *iter;
        const interval_type m   = *len_iter;
        // for each interval, loop thru the consecutive range
        for (interval_type i(0); i < m; ++i, ++v_itr)
          tmp += *v_itr * y[idx + i];
      }
      y[j1] -= tmp;
    }
  }

 protected:
  size_type            _nrows, _ncols;  ///< number of rows/columns
  iarray_type          _row_start;      ///< row pointer for values
  iarray_type          _itrv_start;     ///< row pointer for column indices
  array_type           _vals;           ///< numerical values
  iarray_type          _col_ind_start;  ///< column index start of each interval
  Array<interval_type> _col_len;        ///< interval length
  std::shared_ptr<crs_type> _ref;       ///< reference to A for smart convert
};

/// \class IntervalCCS
/// \brief Interval-based CCS representation
/// \tparam ValueType numerical value type, e.g. \a double, \a float, etc
/// \tparam IndexType index type, e.g. \a int, \a long, etc
/// \tparam IntervalType Interval type, default is \a std::uint8_t
template <class ValueType, class IndexType, class IntervalType = std::uint8_t>
class IntervalCCS {
  // NOTE we use CCS as trait to derive consistent public types
  using _base_type = CCS<ValueType, IndexType>;  ///< base type
 public:
  using value_type       = typename _base_type::value_type;   ///< value type
  using index_type       = typename _base_type::index_type;   ///< index type
  using array_type       = typename _base_type::array_type;   ///< value array
  using iarray_type      = typename _base_type::iarray_type;  ///< index array
  using size_type        = typename _base_type::size_type;    ///< size
  using pointer          = typename _base_type::pointer;      ///< pointer type
  using ipointer         = typename _base_type::ipointer;     ///< index pointer
  using i_iterator       = typename _base_type::i_iterator;  ///< index iterator
  using const_i_iterator = typename _base_type::const_i_iterator;
  ///< const index iterator
  using v_iterator       = typename _base_type::v_iterator;  ///< value iterator
  using const_v_iterator = typename _base_type::const_v_iterator;
  ///< const value iterator
  using ccs_type      = _base_type;                       ///< CCS type
  using crs_type      = typename _base_type::other_type;  ///< CRS type
  using interval_type = IntervalType;                     ///< interval type
  constexpr static bool          ROW_MAJOR    = false;    ///< row major flag
  constexpr static interval_type MAX_INTERVAL = static_cast<interval_type>(-1);
  ///< max interval length, cast from -1
  static_assert(std::is_unsigned<interval_type>::value,
                "interval type must be unsigned");

  /// \brief default constructor
  IntervalCCS() = default;

  /// \brief allow implicit convert from a rvalue reference of CCS
  /// \param[in,out] A CCS of rvalue reference
  /// \param[in] smart_convert (optional) Perform smart converting
  IntervalCCS(ccs_type &&A, const bool smart_convert = true)
      : _nrows(A.nrows()), _ncols(A.ncols()) {
    internal::determine_nitrvs<MAX_INTERVAL>(A, _itrv_start);
    bool convert_to_interval = true;
    if (smart_convert && _itrv_start.size())
      convert_to_interval = size_type(2) * _itrv_start.back() <= A.nnz();
    if (convert_to_interval) {
      internal::build_intervals(A, _itrv_start, _row_ind_start, _row_len);
      // destroy A
      _col_start = std::move(A.col_start());
      _vals      = std::move(A.vals());
      A.destroy();
      _ref.reset();
    } else {
      _ref = std::make_shared<ccs_type>(std::move(A));
      iarray_type().swap(_itrv_start);  // destroy allocated space
    }
  }

  /// \brief convert from a CRS matrix
  /// \param[in] A CRS input
  /// \param[in] smart_convert (optional) Perform smart converting
  IntervalCCS(const crs_type &A, const bool smart_convert = true)
      : IntervalCCS(ccs_type(A), smart_convert) {}

  /// \brief default copy constructor
  IntervalCCS(const IntervalCCS &) = default;

  /// \brief default move constructor
  IntervalCCS(IntervalCCS &&) = default;

  /// \brief default assignment
  IntervalCCS &operator=(const IntervalCCS &) = default;

  /// \brief default move assignment
  IntervalCCS &operator=(IntervalCCS &&) = default;

  /// \brief get number of rows
  inline size_type nrows() const { return _nrows; }

  /// \brief get number of columns
  inline size_type ncols() const { return _ncols; }

  /// \brief get the number of nnz
  inline size_type nnz() const { return _ref ? _ref->nnz() : _vals.size(); }

  /// \brief get the number of intervals
  inline size_type nitrvs() const {
    return _ref ? size_type(0) : _row_ind_start.size();
  }

  /// \brief analyze storage ratio from interval to classical
  /// \note Results greater than 1 means we are wasting storage by using
  ///       interval based storage
  inline double storage_cost_ratio() const {
    if (_ref) return 1.0;
    return internal::analyze_storage_cost_ratio<
        sizeof(index_type), sizeof(value_type), sizeof(interval_type)>(
        ncols(), nnz(), nitrvs());
  }

  /// \brief Check if converted to interval-based structure
  inline bool converted() const { return !_ref; }

  /// \brief matrix vector multiplication with different value type
  /// \tparam Vx other value type for \a x
  /// \tparam Vy other value type for \a y
  /// \param[in] x input array pointer
  /// \param[out] y output array pointer
  /// \warning User's responsibilty to ensure valid pointers
  template <class Vx, class Vy>
  inline void mv_nt_low(const Vx *x, Vy *y) const {
    if (!_ncols) return;
    if (_ref)
      _ref->mv_nt_low(x, y);
    else {
      std::fill_n(y, _nrows, Vy(0));
      for (size_type i(0); i < _ncols; ++i) {
        const auto temp     = x[i];
        auto       val_i    = _vals.cbegin() + _col_start[i];
        auto       len_iter = _row_len.cbegin() + _itrv_start[i];
        auto       last     = _row_ind_start.cbegin() + _itrv_start[i + 1];
        // loop thru all intervals
        for (auto iter = _row_ind_start.cbegin() + _itrv_start[i]; iter != last;
             ++iter, ++len_iter) {
          const auto          idx = *iter;
          const interval_type m   = *len_iter;
          // for each interval, loop thru the consecutive range
          for (interval_type j(0); j < m; ++j, ++val_i)
            y[idx + j] += *val_i * temp;
        }
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
    hif_error_if(nrows() != y.size() || ncols() != x.size(),
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
    if (_ref)
      _ref->mv_t_low(x, y);
    else {
      for (size_type i(0); i < _ncols; ++i) {
        typename std::conditional<(sizeof(Vy) > sizeof(value_type)), Vy,
                                  value_type>::type tmp(0);
        auto val_i    = _vals.cbegin() + _col_start[i];
        auto len_iter = _row_len.cbegin() + _itrv_start[i];
        auto last     = _row_ind_start.cbegin() + _itrv_start[i + 1];
        // loop thru all intervals
        for (auto iter = _row_ind_start.cbegin() + _itrv_start[i]; iter != last;
             ++iter, ++len_iter) {
          const auto          idx = *iter;
          const interval_type m   = *len_iter;
          // for each interval, loop thru the consecutive range
          for (interval_type j(0); j < m; ++j, ++val_i)
            tmp += *val_i * x[idx + j];
        }
        y[i] = tmp;
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
    hif_error_if(nrows() != x.size() || ncols() != y.size(),
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
    if (_ref) {
      _ref->solve_as_strict_lower(y);
      return;
    }
    for (size_type j(0); j < _ncols; ++j) {
      const auto y_j      = y[j];
      auto       v_itr    = _vals.cbegin() + _col_start[j];
      auto       len_iter = _row_len.cbegin() + _itrv_start[j];
      auto       last     = _row_ind_start.cbegin() + _itrv_start[j + 1];
      // loop thru all intervals
      for (auto iter = _row_ind_start.cbegin() + _itrv_start[j]; iter != last;
           ++iter, ++len_iter) {
        const auto          idx = *iter;
        const interval_type m   = *len_iter;
        // for each interval, loop thru the consecutive range
        for (interval_type i(0); i < m; ++i, ++v_itr)
          y[idx + i] -= *v_itr * y_j;
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
    if (_ref) {
      _ref->solve_as_strict_upper(y);
      return;
    }
    using rev_iterator   = std::reverse_iterator<const_i_iterator>;
    using rev_v_iterator = std::reverse_iterator<const_v_iterator>;
    using rev_itrv_iterator =
        std::reverse_iterator<typename Array<interval_type>::const_iterator>;

    for (size_type j(_ncols - 1); j != 0u; --j) {
      const auto y_j   = y[j];
      auto       v_itr = rev_v_iterator(_vals.cbegin() + _col_start[j + 1]);
      auto len_iter = rev_itrv_iterator(_row_len.cbegin() + _itrv_start[j + 1]);
      auto last     = rev_iterator(_row_ind_start.cbegin() + _itrv_start[j]);
      // loop through all intervals in reversed order
      for (auto iter =
               rev_iterator(_row_ind_start.cbegin() + _itrv_start[j + 1]);
           iter != last; ++iter, ++len_iter) {
        const auto          idx = *iter - 1;
        const interval_type m   = *len_iter;
        for (interval_type i(m); i != 0u; --i, ++v_itr)
          y[idx + i] -= *v_itr * y_j;
      }
    }
  }

 protected:
  size_type            _nrows, _ncols;  ///< number of rows/columns
  iarray_type          _col_start;      ///< column pointer for values
  iarray_type          _itrv_start;     ///< column pointer for intervals
  array_type           _vals;           ///< numerical values
  iarray_type          _row_ind_start;  ///< row index start of each interval
  Array<interval_type> _row_len;        ///< interval length
  std::shared_ptr<ccs_type> _ref;       ///< reference to A for smart convert
};

/// \brief Trait for determining interval compressed storage
/// \note Use SFINAE techinique
template <class Cs, typename = void>
struct is_interval_cs : std::false_type {};

/// \brief Trait for determining interval compressed storage
/// \note Use SFINAE techinique
template <class Cs>
struct is_interval_cs<Cs, decltype(std::declval<Cs>().nitrvs(), void())>
    : std::true_type {};

/// \brief helper trait to define interval type from classical
/// \warning \a Cs must be classical type, e.g., CRS or CCS
template <class Cs, class IntervalType = std::uint8_t>
struct using_interval_from_classical {
  static_assert(!is_interval_cs<Cs>::value, "cannot be interval type");
  using type = typename std::conditional<
      Cs::ROW_MAJOR,
      IntervalCRS<typename Cs::value_type, typename Cs::index_type,
                  IntervalType>,
      IntervalCCS<typename Cs::value_type, typename Cs::index_type,
                  IntervalType>>::type;
  ///< interval-based type derived from classical version
};

/*!
 * @}
 */ // end group ds

}  // namespace hif

#endif  // _HIF_DS_INTERVALCOMPRESSEDSTORAGE_HPP