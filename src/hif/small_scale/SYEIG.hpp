///////////////////////////////////////////////////////////////////////////////
//                  This file is part of HIF project                         //
///////////////////////////////////////////////////////////////////////////////

/*!
 * \file hif/small_scale/SYEIG.hpp
 * \brief Small scale solver with symmetric eigendecomposition
 * \author Qiao Chen

\verbatim
Copyright (C) 2020 NumGeom Group at Stony Brook University

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

#ifndef _HIF_SMALLSCALE_SYEIG_HPP
#define _HIF_SMALLSCALE_SYEIG_HPP

#include <algorithm>

#include "hif/Options.h"
#include "hif/ds/Array.hpp"
#include "hif/small_scale/SmallScaleBase.hpp"
#include "hif/small_scale/lapack.hpp"
#include "hif/utils/common.hpp"
#include "hif/utils/math.hpp"

namespace hif {

/// \class SYEIG
/// \brief Symmetric eigendecompsition for small scale solver
/// \tparam ValueType value type, e.g. \a double
/// \ingroup sss
template <class ValueType>
class SYEIG : public SmallScaleBase<ValueType> {
  using _base = SmallScaleBase<ValueType>;

 public:
  typedef ValueType                 value_type;     ///< value type
  typedef Lapack<value_type>        lapack_kernel;  ///< lapack backend
  typedef typename _base::size_type size_type;      ///< size type
  typedef typename ValueTypeTrait<value_type>::value_type scalar_type;
  ///< scalar type

  /// \brief get the solver type
  inline static const char *method() { return "SYEIG"; }

  /// \brief default constructor
  SYEIG() = default;

  /// \brief compute decomposition and determine the rank
  /// \param[in] opts control parameters, see \ref Options
  inline void factorize(const Options &opts) {
    hif_error_if(_mat.empty(), "matrix is still empty!");
    hif_error_if(!_base::is_squared(), "the matrix must be squared!");
    if (hif_verbose(INFO, opts))
      hif_info("factorizing dense level by symmetric eigendecomp...");

    // esimate the workspace
    value_type sz;
    lapack_kernel::syev('L', _mat, _w, &sz, -1);
    _work.resize(sz);

    // factorize
    _w.resize(_mat.nrows());
    auto info = lapack_kernel::syev('L', _mat, _w, _work.data(), _work.size());
    if (info > 0)
      hif_error("?syev failed to converge with info=%d.", (int)info);
    else if (info < 0)
      hif_error("?syev's %d-th arg is illegal.", (int)-info);
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

  /// \brief solve \f$\mathbf{Q\Lambda Q}^T\mathbf{x}=\mathbf{x}\f$
  /// \param[in,out] x input rhs, output solution
  inline void solve(Array<value_type> &x,
                    const size_type /* rank */ = 0) const {
    hif_error_if(
        _mat.empty() || _w.empty(),
        "either the matrix is not set or the factorization has not yet done!");
    hif_error_if(x.size() != _mat.nrows(),
                 "unmatched sizes between system and rhs");
    const auto n = x.size();
    _y.resize(n);
    // std::copy(x.cbegin(), x.cend(), _y.begin());
    // step 1, compute y=Q^T*x
    lapack_kernel::gemv('T', value_type(1), _mat, x, value_type(0), _y);
    // step 2, solve inv(lambda)*y
    for (size_type i(0); i < n; ++i) _y[i] /= _w[i];
    // step 3, compute x=Q*y
    lapack_kernel::gemv('N', value_type(1), _mat, _y, value_type(0), x);
  }

  /// \brief wrapper if \a value_type is different from input's
  template <class ArrayType>
  inline void solve(ArrayType &x, const size_type rank = 0) const {
    _base::_x.resize(x.size());
    std::copy(x.cbegin(), x.cend(), _base::_x.begin());
    solve(_base::_x, rank);
    std::copy(_base::_x.cbegin(), _base::_x.cend(), x.begin());
  }

 protected:
  using _base::_mat;                ///< matrix
  using _base::_mat_backup;         ///< matrix backup
  using _base::_rank;               ///< rank
  Array<value_type>         _w;     ///< Eigenvalues
  mutable Array<value_type> _y;     ///< workspace used in solve
  mutable Array<value_type> _work;  ///< work buffer
};

}  // namespace hif

#endif  // _HIF_SMALLSCALE_SYEIG_HPP