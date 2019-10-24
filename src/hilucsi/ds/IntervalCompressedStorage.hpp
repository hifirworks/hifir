///////////////////////////////////////////////////////////////////////////////
//                This file is part of HILUCSI project                       //
///////////////////////////////////////////////////////////////////////////////

/*!
 * \file hilucsi/ds/IntervalCompressedStorage.hpp
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

#ifndef _HILUCSI_DS_INTERVALCOMPRESSEDSTORAGE_HPP
#define _HILUCSI_DS_INTERVALCOMPRESSEDSTORAGE_HPP

#include <cstdint>
#include <memory>

#include "hilucsi/ds/CompressedStorage.hpp"

namespace hilucsi {

namespace internal {

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
        len = 1;   // reset length
      }
      ++iter;
      ++len;
    }
    itrv_start[i + 1] = itrv_start[i] + nitrvs;
  }
}

template <class CsType, class IndexArray, class IntervalArray>
inline void build_intervals(const CsType &A, const IndexArray &itrv_start,
                            IndexArray &ind_start, IntervalArray &lens) {
  using size_type     = typename CsType::size_type;
  using interval_type = typename IntervalArray::value_type;

  const size_type total_nitrvs = itrv_start.back();
  ind_start.resize(total_nitrvs);
  lens.resize(total_nitrvs);

  const auto n = itrv_start.size() - 1;

  auto ind_iter = ind_start.begin();
  auto len_iter = lens.begin();

  for (size_type i(0); i < n; ++i) {
    if (!A.nnz_in_primary(i)) continue;
    auto iter = A.ind_cbegin(i), iter_last = A.ind_cend(i);
    *ind_iter++ = *iter++;  // increment iter here
    *len_iter++ = 1;
    while (iter != iter_last) {
      // XXX: !len:-> overflow back to zero
      if (*(iter - 1) + 1 != *iter || !*len_iter) {
        // get a breaking point
        *ind_iter++ = *iter;
        --(*len_iter);
        *len_iter++ = 1;
      }
      ++iter;
      ++(*len_iter);
    }
    hilucsi_assert(ind_iter == ind_start.begin() + itrv_start[i + 1], "fatal");
    hilucsi_assert(len_iter == lens.begin() + itrv_start[i + 1], "fatal");
  }
}
}  // namespace internal

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
  IntervalCRS(crs_type &&A, const bool /* smart_convert */ = true)
      : _nrows(A.nrows()), _ncols(A.ncols()) {
    internal::determine_nitrvs<MAX_INTERVAL>(A, _itrv_start);
    // NOTE, we can check number of intervals to do smart_convert
    if (true) {
      internal::build_intervals(A, _itrv_start, _col_ind_start, _col_len);
      // destroy A
      _row_start = std::move(A.row_start());
      _vals      = std::move(A.vals());
      _ref.reset();
    } else
      _ref = std::make_shared<crs_type>(std::move(A));
  }

  /// \brief convert from a CCS matrix
  /// \param[in] A CCS input
  IntervalCRS(const ccs_type &A) : IntervalCRS(crs_type(A)) {}

  /// \brief get number of rows
  inline size_type nrows() const { return _nrows; }

  /// \brief get number of columns
  inline size_type ncols() const { return _ncols; }

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
        const auto val_i    = _vals.cbegin() + _row_start[i];
        auto       len_iter = _col_len.cbegin() + _itrv_start[i];
        auto       last     = _col_ind_start.cbegin() + _itrv_start[i + 1];
        Vy         tmp(0);
        for (auto iter = _col_ind_start.cbegin() + _itrv_start[i]; iter != last;
             ++iter, ++len_iter) {
          const auto      idx = *iter;
          const size_type m   = *len_iter;
          for (size_type j(0); j < m; ++j, ++val_i) tmp += *val_i * x[idx + j];
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
    hilucsi_error_if(nrows() != y.size() || ncols() != x.size(),
                     "matrix vector multiplication unmatched sizes!");
    hilucsi_error_if(istart >= nrows(), "%zd exceeds the row size %zd", istart,
                     nrows());
    hilucsi_error_if(istart + len > nrows(),
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
    hilucsi_error_if(nrows() != y.size() || ncols() != x.size(),
                     "matrix vector multiplication unmatched sizes!");
    mv_nt_low(x.data(), y.data());
  }

 protected:
  size_type            _nrows, _ncols;  ///< number of rows/columns
  array_type           _row_start;      ///< row pointer for values
  array_type           _itrv_start;     ///< row pointer for column indices
  array_type           _vals;           ///< numerical values
  array_type           _col_ind_start;  ///< column index start of each interval
  Array<interval_type> _col_len;        ///< interval length
  std::shared_ptr<crs_type> _ref;       ///< reference to A for smart convert
};

}  // namespace hilucsi

#endif  // _HILUCSI_DS_INTERVALCOMPRESSEDSTORAGE_HPP
