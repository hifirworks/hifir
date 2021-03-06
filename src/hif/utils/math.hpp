///////////////////////////////////////////////////////////////////////////////
//                  This file is part of the HIFIR library                   //
///////////////////////////////////////////////////////////////////////////////

/*!
 * \file hif/utils/math.hpp
 * \brief Some handy BLAS 1 vector routines
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

#ifndef _HIF_UTILS_MATH_HPP
#define _HIF_UTILS_MATH_HPP

#include <algorithm>
#include <cmath>
#include <complex>
#include <type_traits>

#include "hif/utils/common.hpp"
#include "hif/utils/log.hpp"

namespace hif {

/*!
 * \addtogroup util
 * @{
 */

/// \brief real helper function
using std::real;

/// \brief imaginary helper function
using std::imag;

/// \brief get the abs value
using std::abs;

/// \brief conjugate helper function
/// \tparam T abstract value type, real type
/// \param[in] v value input
/// \return conceputally, the return is the "conjugate" of \a v
template <class T>
inline typename std::enable_if<std::is_floating_point<T>::value, T>::type
conjugate(const T &v) {
  return v;
}

/// \brief conjugate helper function
/// \tparam T abstract value type, complex type
/// \param[in] v value input
/// \return conceputally, the return is the "conjugate" of \a v
template <class T>
inline typename std::enable_if<!std::is_floating_point<T>::value, T>::type
conjugate(const T &v) {
  return std::conj(v);
}

/// \brief compute the dot product
/// \tparam ArrayType array type, see, for instance, \ref Array
/// \tparam Iteratable iteratable type for second array
/// \param[in] v1 first array input
/// \param[in] v2 second array input
/// \return dot product of \a v1 and \a v2
template <class ArrayType, class Iteratable>
inline typename ArrayType::value_type inner(const ArrayType & v1,
                                            const Iteratable &v2) {
  using value_type = typename ArrayType::value_type;
  value_type tmp(0);
  const auto n = v1.size();
  for (auto i = 0ul; i < n; ++i) tmp += conjugate(v1[i]) * v2[i];
  return tmp;
}

/// \brief compute the norm 2 square
/// \tparam ArrayType array type, see, for instance, \ref Array
/// \param[in] v array input
template <class ArrayType>
inline typename ValueTypeTrait<typename ArrayType::value_type>::value_type
norm2_sq(const ArrayType &v) {
  // get the scalar type
  using scalar_type =
      typename ValueTypeTrait<typename ArrayType::value_type>::value_type;
  scalar_type tmp(0);
  const auto  n = v.size();
  for (auto i = 0ul; i < n; ++i) tmp += real(conjugate(v[i]) * v[i]);
  return tmp;
}

/// \brief compute the Euclidean norm of a given vector
/// \tparam ArrayType array type, see, for instance, \ref Array
/// \param[in] v array input
template <class ArrayType>
inline typename ValueTypeTrait<typename ArrayType::value_type>::value_type
norm2(const ArrayType &v) {
  // get the scalar type
  using value_type  = typename ArrayType::value_type;
  using scalar_type = typename ValueTypeTrait<value_type>::value_type;

  scalar_type tmp(0);
  const auto  n = v.size();
  if (!n) return scalar_type(0);
  // get the max mag
  const scalar_type max_mag = abs(*std::max_element(
      v.cbegin(), v.cend(), [](const value_type &l, const value_type &r) {
        return abs(l) < abs(r);
      }));

  if (max_mag == Const<scalar_type>::ZERO)
    for (auto i = 0ul; i < n; ++i) tmp += abs(v[i]);
  else {
    const auto alpha = scalar_type(1) / max_mag;
    value_type a;
    for (auto i = 0ul; i < n; ++i) {
      a = v[i] * alpha;
      tmp += real(conjugate(a) * a);
    }
    tmp = max_mag * std::sqrt(tmp);
  }
  return tmp;
}

/// \brief normalize an array
/// \tparam ArrayType array type, see, for instance, \ref Array
/// \param[in,out] v array input, upon output, it will be normalized
template <class ArrayType>
inline void normalize(ArrayType &v) {
  // get the scalar type
  using scalar_type =
      typename ValueTypeTrait<typename ArrayType::value_type>::value_type;
  const auto inv_nrm = scalar_type(1) / norm2(v);
  for (auto iter = v.begin(); iter != v.end(); ++iter) *iter *= inv_nrm;
}

/// \brief normalize an array and save the result to another array
/// \tparam ArrayType array type, see, for instance, \ref Array
/// \tparam Iter Iterator type
/// \param[in] v array input
/// \param[out] w iterator output
template <class ArrayType, class Iter>
inline void normalize2(const ArrayType &v, Iter &w) {
  // get the scalar type
  using scalar_type =
      typename ValueTypeTrait<typename ArrayType::value_type>::value_type;
  const auto inv_nrm = scalar_type(1) / norm2(v);
  const auto n       = v.size();
  for (auto i(0ul); i < n; ++i) w[i] = v[i] * inv_nrm;
}

/*!
 * @}
 */

#ifndef DOXYGEN_SHOULD_SKIP_THIS

// // long double
// template <>
// inline long double conjugate(const long double &v) {
//   return v;
// }

// // double
// template <>
// inline double conjugate(const double &v) {
//   return v;
// }

// // float
// template <>
// inline float conjugate(const float &v) {
//   return v;
// }

// template <>
// inline std::complex<long double> conjugate(const std::complex<long double>
// &v) {
//   return std::conj(v);
// }

// template <>
// inline std::complex<double> conjugate(const std::complex<double> &v) {
//   return std::conj(v);
// }

// template <>
// inline std::complex<float> conjugate(const std::complex<float> &v) {
//   return std::conj(v);
// }

// The standard does not support binary complex arithmetic operations with
// lhs and rhs that have different types. We provide a series of overloaded
// binary operators based on
// https://en.cppreference.com/w/cpp/numeric/complex/operator_arith3
// NOTE: The return type is always the LHS

template <class T, class V>
inline std::complex<T> operator+(const std::complex<T> &lhs,
                                 const std::complex<V> &rhs) {
  return lhs + std::complex<T>(rhs);
}

template <class T, class V>
inline std::complex<T> operator+(const std::complex<T> &lhs, const V rhs) {
  return lhs + static_cast<T>(rhs);
}

template <class T, class V>
inline std::complex<T> operator+(const T lhs, const std::complex<V> &rhs) {
  return std::complex<T>(lhs) + std::complex<T>(rhs);
}

template <class T, class V>
inline std::complex<T> operator-(const std::complex<T> &lhs,
                                 const std::complex<V> &rhs) {
  return lhs - std::complex<T>(rhs);
}

template <class T, class V>
inline std::complex<T> operator-(const std::complex<T> &lhs, const V rhs) {
  return lhs - static_cast<T>(rhs);
}

template <class T, class V>
inline std::complex<T> operator-(const T lhs, const std::complex<V> &rhs) {
  return std::complex<T>(lhs) - std::complex<T>(rhs);
}

template <class T, class V>
inline std::complex<T> operator*(const std::complex<T> &lhs,
                                 const std::complex<V> &rhs) {
  return lhs * std::complex<T>(rhs);
}

template <class T, class V>
inline std::complex<T> operator*(const std::complex<T> &lhs, const V rhs) {
  return lhs * static_cast<T>(rhs);
}

template <class T, class V>
inline std::complex<T> operator*(const T lhs, const std::complex<V> &rhs) {
  return std::complex<T>(lhs) * std::complex<T>(rhs);
}

template <class T, class V>
inline std::complex<T> operator/(const std::complex<T> &lhs,
                                 const std::complex<V> &rhs) {
  return lhs / std::complex<T>(rhs);
}

template <class T, class V>
inline std::complex<T> operator/(const std::complex<T> &lhs, const V rhs) {
  return lhs / static_cast<T>(rhs);
}

template <class T, class V>
inline std::complex<T> operator/(const T lhs, const std::complex<V> &rhs) {
  return std::complex<T>(lhs) / std::complex<T>(rhs);
}

#endif  // DOXYGEN_SHOULD_SKIP_THIS

}  // namespace hif

#endif  // _HIF_UTILS_MATH_HPP
