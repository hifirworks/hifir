///////////////////////////////////////////////////////////////////////////////
//                  This file is part of HIF project                         //
///////////////////////////////////////////////////////////////////////////////

/*!
 * \file hif/utils/mv_helper.hpp
 * \brief Matrix-vector product helper for sparse matrices and \a std::function
 * \author Qiao Chen

\verbatim
Copyright (C) 2021 NumGeom Group at Stony Brook University

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

#ifndef _HIF_UTILS_MVHELPER_HPP
#define _HIF_UTILS_MVHELPER_HPP

#include <functional>

#include "hif/utils/common.hpp"

namespace hif {

/*!
 * \addtogroup alg
 * @{
 */

/// \brief matrix-vector product
/// \tparam CsType compressed storage type, e.g., \a CRS
/// \tparam IArray input array type
/// \tparam OArray output array type
/// \param[in] A input matrix
/// \param[in] x input array
/// \param[out] y output array
/// \param[in] tran (optional) transpose/Hermitian flag, default is false
template <class CsType, class IArray, class OArray, typename T = void>
inline void mv(const CsType &A, const IArray &x, OArray &y,
               const bool tran = false) {
  A.mv(x, y, tran);
}

/// \brief matrix-vector product with \a std::function
/// \tparam IArray input array type
/// \tparam OArray output array type
/// \param[in] Afunc input matrix-like operator
/// \param[in] x input array
/// \param[out] y output array
/// \param[in] tran (optional) transpose/Hermitian flag, default is false
template <class IArray, class OArray, typename T = void>
inline void
mv(const std::function<void(const void *, const typename IArray::size_type,
                            const char, void *, const char, const bool)> &Afunc,
   const IArray &x, OArray &y, const bool tran = false) {
  using i_value_type = typename IArray::value_type;
  using o_value_type = typename OArray::value_type;
  Afunc((const void *)x.data(), x.size(),
        ValueTypeTrait<i_value_type>::signature, (void *)y.data(),
        ValueTypeTrait<o_value_type>::signature, tran);
}

/*!
 * @}
 */

}  // namespace hif

#endif  // _HIF_UTILS_MVHELPER_HPP