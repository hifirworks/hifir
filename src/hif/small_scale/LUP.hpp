///////////////////////////////////////////////////////////////////////////////
//                  This file is part of HIF project                         //
///////////////////////////////////////////////////////////////////////////////

/*!
 * \file hif/small_scale/LUP.hpp
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

#ifndef _HIF_SMALLSCALE_LUP_HPP
#define _HIF_SMALLSCALE_LUP_HPP

#include <algorithm>
#include <array>

#include "hif/Options.h"
#include "hif/ds/Array.hpp"
#include "hif/small_scale/SmallScaleBase.hpp"
#include "hif/small_scale/lapack.hpp"

namespace hif {

/// \class LUP
/// \brief LU with partial pivoting for small scale solver
/// \tparam ValueType value type, e.g. \a double
/// \ingroup sss
template <class ValueType>
class LUP : public SmallScaleBase<ValueType> {
  using _base = SmallScaleBase<ValueType>;

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
    hif_error_if(_mat.empty(), "matrix is still empty!");
    hif_error_if(!_base::is_squared(), "the matrix must be squared!");
    if (hif_verbose(INFO, opts)) hif_info("factorizing dense level by LU...");

    _ipiv.resize(_mat.nrows());
    const auto info = lapack_kernel::getrf(_mat, _ipiv);
    if (info < 0)
      hif_error("GETRF returned negative info!");
    else if (info > 0) {
      hif_warning(
          "GETRF returned positive info, U(%zd,%zd) is exactly zero! "
          "Consider using QRCP for small scale solver!",
          (size_type)info, (size_type)info);
      _rank = info - 1;
    } else
      _rank = _mat.ncols();
  }

  /// \brief refactorize the dense block
  /// \param[in] opts control parameters, see \ref Options
  inline void refactorize(const Options &opts) {
    // copy backup
    _mat.resize(_mat_backup.nrows(), _mat_backup.ncols());
    std::copy(_mat_backup.array().cbegin(), _mat_backup.array().cend(),
              _mat.array().begin());
    factorize(opts);
  }

  /// \brief solve \f$\mathbf{LUx}=\mathbf{Px}\f$
  /// \param[in,out] x input rhs, output solution
  /// \param[in] tran (optional) tranpose flag
  inline void solve(Array<value_type> &x, const size_type /* rank */ = 0,
                    const bool         tran = false) const {
    hif_error_if(
        _mat.empty() || _ipiv.empty(),
        "either the matrix is not set or the factorization has not yet done!");
    hif_error_if(x.size() != _mat.nrows(),
                 "unmatched sizes between system and rhs");
    if (lapack_kernel::getrs(_mat, _ipiv, x, tran ? 'T' : 'N') < 0)
      hif_error("GETRS returned negative info!");
  }

  /// \brief wrapper if \a value_type is different from input's
  template <class ArrayType>
  inline void solve(ArrayType &x, const bool tran = false) const {
    _base::_x.resize(x.size());
    std::copy(x.cbegin(), x.cend(), _base::_x.begin());
    solve(_base::_x, tran);
    std::copy(_base::_x.cbegin(), _base::_x.cend(), x.begin());
  }

  /// \brief solve with multiple RHS
  template <class T, size_type Nrhs>
  inline void solve_mrhs(Array<std::array<T, Nrhs>> &x, const size_type = 0,
                         const bool                  tran = false) const {
    hif_error_if(
        _mat.empty() || _ipiv.empty(),
        "either the matrix is not set or the factorization has not yet done!");
    hif_error_if(x.size() != _mat.nrows(),
                 "unmatched sizes between system and rhs");
    _base::_mrhs.resize(x.size(), Nrhs);
    for (size_type j = 0; j < Nrhs; ++j)
      for (size_type i(0); i < x.size(); ++i) _mrhs(i, j) = x[i][j];
    if (lapack_kernel::getrs(_mat, _ipiv, _base::_mrhs, tran ? 'T' : 'N') < 0)
      hif_error("GETRS returned negative info!");
    for (size_type j = 0; j < Nrhs; ++j)
      for (size_type i(0); i < x.size(); ++i) x[i][j] = _mrhs(i, j);
  }

 protected:
  using _base::_mat;            ///< matrix
  using _base::_mat_backup;     ///< matrix backup
  using _base::_rank;           ///< rank
  Array<hif_lapack_int> _ipiv;  ///< row pivoting array
};

}  // namespace hif

#endif  // _HIF_SMALLSCALE_LUP_HPP
