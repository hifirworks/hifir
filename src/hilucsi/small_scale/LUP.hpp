///////////////////////////////////////////////////////////////////////////////
//                This file is part of HILUCSI project                       //
///////////////////////////////////////////////////////////////////////////////

/*!
 * \file hilucsi/small_scale/LUP.hpp
 * \brief Small scale solver with dense LU with partial pivoting
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

#ifndef _HILUCSI_SMALLSCALE_LUP_HPP
#define _HILUCSI_SMALLSCALE_LUP_HPP

#include <algorithm>

#include "hilucsi/Options.h"
#include "hilucsi/ds/Array.hpp"
#include "hilucsi/small_scale/SmallScaleBase.hpp"
#include "hilucsi/small_scale/lapack.hpp"

namespace hilucsi {

/// \class LUP
/// \brief LU with partial pivoting for small scale solver
/// \tparam ValueType value type, e.g. \a double
/// \ingroup sss
template <class ValueType>
class LUP : public internal::SmallScaleBase<ValueType> {
  using _base = internal::SmallScaleBase<ValueType>;

 public:
  typedef ValueType                 value_type;     ///< value type
  typedef Lapack<value_type>        lapack_kernel;  ///< lapack backend
  typedef typename _base::size_type size_type;      ///< size type

  /// \brief get the solver type
  inline static const char *method() { return "LUP"; }

  /// \brief default constructor
  LUP() = default;

  /// \brief perform LU with partial pivoting
  /// \param[in] opts control parameters, see \ref Options
  inline void factorize(const Options &opts) {
    hilucsi_error_if(_mat.empty(), "matrix is still empty!");
    hilucsi_error_if(!_base::is_squared(), "the matrix must be squared!");
    if (hilucsi_verbose(INFO, opts))
      hilucsi_info("factorizing dense level by LU...");

    _ipiv.resize(_mat.nrows());
    const auto info = lapack_kernel::getrf(_mat, _ipiv);
    if (info < 0)
      hilucsi_error("GETRF returned negative info!");
    else if (info > 0) {
      hilucsi_warning(
          "GETRF returned positive info, U(%zd,%zd) is exactly zero! "
          "Consider using QRCP for small scale solver!",
          (size_type)info, (size_type)info);
      _rank = info - 1;
    } else
      _rank = _mat.ncols();
  }

  /// \brief solve \f$\mathbf{LUx}=\mathbf{Px}\f$
  /// \param[in,out] x input rhs, output solution
  /// \param[in] tran (optional) tranpose flag
  inline void solve(Array<value_type> &x, const size_type /* rank */ = 0,
                    const bool         tran = false) const {
    hilucsi_error_if(
        _mat.empty() || _ipiv.empty(),
        "either the matrix is not set or the factorization has not yet done!");
    hilucsi_error_if(x.size() != _mat.nrows(),
                     "unmatched sizes between system and rhs");
    if (lapack_kernel::getrs(_mat, _ipiv, x, tran ? 'T' : 'N') < 0)
      hilucsi_error("GETRS returned negative info!");
  }

  /// \brief wrapper if \a value_type is different from input's
  template <class ArrayType>
  inline void solve(ArrayType &x, const bool tran = false) const {
    _x.resize(x.size());
    std::copy(x.cbegin(), x.cend(), _x.begin());
    solve(_x, tran);
    std::copy(_x.cbegin(), _x.cend(), x.begin());
  }

 protected:
  using _base::_mat;                ///< matrix
  using _base::_rank;               ///< rank
  Array<hilucsi_lapack_int> _ipiv;  ///< row pivoting array
  mutable Array<value_type> _x;     ///< buffer if type is different
};

}  // namespace hilucsi

#endif  // _HILUCSI_SMALLSCALE_LUP_HPP
