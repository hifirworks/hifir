///////////////////////////////////////////////////////////////////////////////
//                This file is part of HILUCSI project                       //
///////////////////////////////////////////////////////////////////////////////

/*!
 * \file hilucsi/utils/common.hpp
 * \brief Utility/helper routines
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

#ifndef _HILUCSI_UTILS_UTILS_HPP
#define _HILUCSI_UTILS_UTILS_HPP

#include <algorithm>
#include <complex>
#include <iterator>
#include <limits>
#include <new>
#include <type_traits>
#include <vector>

#include "hilucsi/utils/log.hpp"

namespace hilucsi {

/*!
 * \addtogroup util
 * @{
 */

/// \brief apply binary search on a sorted array
/// \tparam Iter iterator type
/// \tparam ValueType value type that can be casted to the derefence of Iter
/// \param[in] first starting iterator
/// \param[in] last ending iterator
/// \param[in] v target value
/// \return pair<bool, Iter> that the \a first indicates if or not we find
/// v, while the \a second is the location
/// \todo replace the implementation with one-sided binary search
template <class Iter, class ValueType>
inline std::pair<bool, Iter> find_sorted(Iter first, Iter last,
                                         const ValueType &v) {
  // NOTE, we use the c++ std binary search for now
  // WARNING! When replacing the impl, make sure the behavior aligns with
  // standard lower_bound!
  auto lower = std::lower_bound(first, last, v);
  return std::make_pair(lower != last && (ValueType)*lower == v, lower);
}

/// \brief rotate a subset of vector, s.t. src appears to the left most pos
/// \tparam ArrayType input and output array type
/// \param[in] n **local** size of how many items will be shifted
/// \param[in] src index in **global** range
/// \param[in,out] v input and output array
/// \note Complexity: \f$\mathcal{O}(n)\f$
/// \sa rotate_right
template <class ArrayType>
inline void rotate_left(const typename ArrayType::size_type n,
                        const typename ArrayType::size_type src, ArrayType &v) {
  auto itr_first = v.begin() + src, itr_last = itr_first + n;
  std::rotate(itr_first, itr_first + 1, itr_last);
}

/// \brief rotate a subset of vector, s.t. src appears to the right most pos
/// \tparam ArrayType input and output array type
/// \param[in] n **local** size of how many items will be shifted
/// \param[in] src index in **global** range
/// \param[in,out] v input and output array
/// \note Complexity: \f$\mathcal{O}(n)\f$
/// \sa rotate_left
///
/// For right rotation, we need to use \a reverse_iterator
template <class ArrayType>
inline void rotate_right(const typename ArrayType::size_type n,
                         const typename ArrayType::size_type src,
                         ArrayType &                         v) {
  typedef std::reverse_iterator<typename ArrayType::iterator> iterator;
  // NOTE requiring explicit construction
  iterator itr_first(v.begin() + src + 1), itr_last(itr_first + n);
  std::rotate(itr_first, itr_first + 1, itr_last);
}

// /// \brief convert a given index to c-based index
// /// \tparam IndexType integer type
// /// \tparam OneBased if \a true not the index is Fortran based
// /// \param[in] i input index
// /// \return C-based index
// template <class IndexType, bool OneBased>
// inline constexpr IndexType to_c_idx(const IndexType i) {
//   return i - static_cast<IndexType>(OneBased);
// }

// /// \brief convert a C-based index to original input index
// /// \tparam IndexType integer type
// /// \tparam OneBased if \a true not the index is Fortran based
// /// \param[in] i input index
// /// \return Original index
// template <class IndexType, bool OneBased>
// inline constexpr IndexType to_ori_idx(const IndexType i) {
//   return i + static_cast<IndexType>(OneBased);
// }

/// \brief trait extract value type
/// \tparam T value type
/// \note For user-defined types, instance this trait
///
/// By default, the value type is \a void for compilation error handling
template <class T>
struct ValueTypeTrait {
  typedef void value_type;  ///< value type
};

/// \class Const
/// \brief constant values
/// \tparam T value type
template <class T>
class Const {
 public:
  typedef typename ValueTypeTrait<T>::value_type value_type;  ///< value type
  typedef std::numeric_limits<value_type>        std_trait;   ///< std trait

  constexpr static value_type MIN  = std_trait::min();  ///< safe machine min
  constexpr static value_type MAX  = std_trait::max();  ///< safe machine max
  constexpr static value_type EPS  = std_trait::epsilon();  ///< machine prec
  constexpr static value_type ZERO = value_type();          ///< zero
  constexpr static value_type ONE  = value_type(1);         ///< one

  static_assert(!std::is_same<value_type, void>::value,
                "not a support value type, instance ValueTypeTrait first");
};

/// \brief ensure consistent between data types of \a T and input array
/// \tparam T desired data type
/// \tparam ArrayType array container type, see \ref Array
/// \param[in] v input array
/// \param[in] copy_if_needed if \a true (default), then copy the values
///
/// This helper function simply checks if \a sizeof(T) equals to that of
/// \a ArrayType::value_type, if not, then allocate new array with the same
/// size of \a v. Notice that the above condition should be used to deallocate
/// the memory.
template <class T, class ArrayType>
inline T *ensure_type_consistency(const ArrayType &v,
                                  const bool       copy_if_needed = true) {
  constexpr static bool consist =
      sizeof(T) == sizeof(typename ArrayType::value_type);
  if (consist) return (T *)v.data();
  T *ptr = new (std::nothrow) T[v.size()];
  hilucsi_error_if(!ptr, "memory allocation failed");
  if (copy_if_needed) std::copy(v.cbegin(), v.cend(), ptr);
  return ptr;
}

/// \brief trait to determine mixed precision relationships
/// \tparam T value type
///
/// We instantiate for \a double, \a float, and \a half (if included)
template <class T>
struct ValueTypeMixedTrait {
  typedef T boost_type;   ///< high precision
  typedef T reduce_type;  ///< reduced precision
};

/*!
 * @}
 */ // group util

#ifndef DOXYGEN_SHOULD_SKIP_THIS

// long double
template <>
struct ValueTypeTrait<long double> {
  using value_type = long double;
};

// double
template <>
struct ValueTypeTrait<double> {
  using value_type = double;
};

// float
template <>
struct ValueTypeTrait<float> {
  using value_type = float;
};

// for standard complex numbers
template <class T>
struct ValueTypeTrait<std::complex<T>> {
  typedef typename ValueTypeTrait<T>::value_type value_type;
};

// mixed traits for long double
template <>
struct ValueTypeMixedTrait<long double> {
  using boost_type  = long double;
  using reduce_type = double;
};

// mixed traits double
template <>
struct ValueTypeMixedTrait<double> {
  using boost_type  = long double;
  using reduce_type = float;
};

// mixed traits for float
template <>
struct ValueTypeMixedTrait<float> {
  using boost_type = double;
  using reduce_type =
#  ifdef _HILUCSI_UTILS_HALF_HPP
      half;
#  else
      float;
#  endif
};

#  ifdef _HILUCSI_UTILS_HALF_HPP
template <>
struct ValueTypeMixedTrait<half> {
  using boost_type  = float;
  using reduce_type = half;
};
#  endif

// mixed traits for complex
template <class T>
struct ValueTypeMixedTrait<std::complex<T>> {
  using boost_type = std::complex<typename ValueTypeMixedTrait<T>::boost_type>;
  using reduce_type =
      std::complex<typename ValueTypeMixedTrait<T>::reduce_type>;
};

#endif  // DOXYGEN_SHOULD_SKIP_THIS

namespace internal {

/*!
 * \addtogroup util
 * @{
 */

/// \class SpVInternalExtractor
/// \brief advanced helper class for extract internal data attributes from
///        \ref SparseVector
/// \tparam SpVecType sparse vector type, see \ref SparseVector
/// \ingroup util
template <class SpVecType>
class SpVInternalExtractor : public SpVecType {
 public:
  typedef SpVecType                  base;         ///< base type
  typedef typename base::size_type   size_type;    ///< size_type
  typedef typename base::iarray_type iarray_type;  ///< index array type

  inline size_type &        counts() { return base::_counts; }
  inline iarray_type &      dense_tags() { return base::_dense_tags; }
  inline std::vector<bool> &sparse_tags() { return base::_sparse_tags; }

  inline const size_type &  counts() const { return base::_counts; }
  inline const iarray_type &dense_tags() const { return base::_dense_tags; }
  inline const std::vector<bool> &sparse_tags() const {
    return base::_sparse_tags;
  }
};

/// \struct StdoutStruct
/// \brief struct wrapped around \a stdout
struct StdoutStruct {
  template <class... Args>
  inline void operator()(const char *f, Args... args) const {
    hilucsi_info(f, args...);
  }
};

/// \struct StderrStruct
/// \brief struct wrapped around \a stderr
struct StderrStruct {
  template <class... Args>
  inline void operator()(const char *file, const char *func,
                         const unsigned line, const char *f,
                         Args... args) const {
    if (warn_flag()) warning(nullptr, file, func, line, f, args...);
  }
};

/// \struct DummyStreamer
/// \brief dummy streamer (empty functor)
struct DummyStreamer {
  template <class... Args>
  inline void operator()(const char *, Args...) const {}
};

/// \struct DummyErrorStreamer
/// \brief streamer with error/warning information for dummy usage
struct DummyErrorStreamer {
  template <class... Args>
  inline void operator()(const char *, const char *, const unsigned,
                         const char *, Args...) const {}
};

/*!
 * @}
 */

}  // namespace internal
}  // namespace hilucsi

#endif
