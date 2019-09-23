///////////////////////////////////////////////////////////////////////////////
//                This file is part of HILUCSI project                       //
///////////////////////////////////////////////////////////////////////////////

/*!
 * \file hilucsi/utils/math.hpp
 * \brief Some handy BLAS 1 vector routines
 * \authors Qiao,

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

#ifndef _HILUCSI_UTILS_MATH_HPP
#define _HILUCSI_UTILS_MATH_HPP

#include <algorithm>
#include <complex>

#include "hilucsi/utils/common.hpp"
#include "hilucsi/utils/log.hpp"

namespace hilucsi {

/*!
 * \addtogroup util
 * @{
 */

/// \brief conjugate helper function
/// \tparam T abstract value type
/// \param[in] v value input
/// \return conceputally, the return is the "conjugate" of \a v
///
/// We instantiate this function for common data types, in general, the user
/// should supply his/her own rule for how to conjugate a value type
template <class T>
inline T conj(const T &v);

/// \brief get the abs value
/// \tparam T abstract value type
/// \param[in] v value input
/// \return conceputally, the return is the "conjugate" of \a v
template <class T>
inline typename ValueTypeTrait<T>::value_type abs(const T &v);

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
  for (auto i = 0ul; i < n; ++i) tmp += conj(v1[i]) * v2[i];
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
  for (auto i = 0ul; i < n; ++i) tmp += conj(v[i]) * v[i];
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
      tmp += conj(a) * a;
    }
    tmp = max_mag * std::sqrt(tmp);
  }
  return tmp;
}

/*!
 * @}
 */

#ifndef DOXYGEN_SHOULD_SKIP_THIS

// double
template <>
inline double conj(const double &v) {
  return v;
}

// float
template <>
inline float conj(const float &v) {
  return v;
}

template <class T>
inline std::complex<T> conj(const std::complex<T> &v) {
  return std::conj(v);
}

// double
template <>
inline double abs(const double &v) {
  return std::abs(v);
}

// float
template <>
inline float abs(const float &v) {
  return std::abs(v);
}

// complex
template <class T>
inline T abs(const std::complex<T> &v) {
  return std::abs(v);
}

#endif  // DOXYGEN_SHOULD_SKIP_THIS

}  // namespace hilucsi

#endif  // _HILUCSI_UTILS_MATH_HPP
