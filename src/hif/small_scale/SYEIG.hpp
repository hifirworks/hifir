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
#include <numeric>

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
      hif_info("factorizing dense level by symmetric (%s) eigendecomp...",
               opts.spd > 0 ? "PD" : (opts.spd < 0 ? "ND" : "ID"));

    constexpr static scalar_type EPS =
        std::pow(Const<scalar_type>::EPS, 2. / 3);

    // esimate the workspace
    if (!lapack_kernel::IS_REAL)
      _rwork.resize(std::max(1, int(3 * _mat.nrows() - 2)));
    value_type sz;
    lapack_kernel::syev('L', _mat, _w, &sz, -1, _rwork);
    const auto n = _mat.nrows();
    _work.resize(std::max((size_type)abs(sz), n));

    // factorize
    _w.resize(n);
    auto info =
        lapack_kernel::syev('L', _mat, _w, _work.data(), _work.size(), _rwork);
    if (info > 0)
      hif_error("?syev failed to converge with info=%d.", (int)info);
    else if (info < 0)
      hif_error("?syev's %d-th arg is illegal.", (int)-info);

    // handle truncation
    _trunc_order.resize(n);
    std::iota(_trunc_order.begin(), _trunc_order.end(), 0);
    const scalar_type thres =
        EPS * *std::max_element(_w.cbegin(), _w.cend(),
                                [](const value_type a, const value_type b) {
                                  return std::abs(a) < std::abs(b);
                                });
    _rank = n;
    if (opts.spd > 0) {
      // ensure pd-ness
      for (int i(n - 1); i > -1; --i)
        if (_w[i] <= 0.0 || std::abs(_w[i]) <= thres)
          --_rank;
        else
          break;
    } else if (opts.spd < 0) {
      // ensure nd-ness
      // reverse the order for truncations
      std::reverse(_trunc_order.begin(), _trunc_order.end());
      for (size_type i(0); i < n; ++i)
        if (_w[i] >= 0.0 || std::abs(_w[i]) <= thres)
          --_rank;
        else
          break;
    } else {
      // indefinite
      // sort based on magnitude of the eigenvalues
      std::sort(_trunc_order.begin(), _trunc_order.end(),
                [&](const int a, const int b) {
                  return std::abs(_w[a]) > std::abs(_w[b]);
                });
      for (int i(n - 1); i > -1; --i)
        if (std::abs(_w[_trunc_order[i]]) <= thres)
          --_rank;
        else
          break;
    }
    if (_rank != n)
      hif_warning("\n  System is ill-conditioned with rank=%d, dim=%d.",
                  (int)_rank, (int)n);
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
  inline void solve(Array<value_type> &x, const size_type rank = 0) const {
    hif_error_if(
        _mat.empty() || _w.empty(),
        "either the matrix is not set or the factorization has not yet done!");
    hif_error_if(x.size() != _mat.nrows(),
                 "unmatched sizes between system and rhs");
    const auto      n  = x.size();
    const size_type rk = rank == 0u ? _rank : (rank > n ? n : rank);
    // std::copy(x.cbegin(), x.cend(), _y.begin());
    // step 1, compute y=Q^T*x
    lapack_kernel::gemv('C', value_type(1), _mat, x, value_type(0), _work);
    // step 2, solve inv(lambda)*y with truncation
    for (size_type i(0); i < rk; ++i)
      _work[_trunc_order[i]] /= _w[_trunc_order[i]];
    for (size_type i(rk); i < n; ++i) _work[_trunc_order[i]] = 0;
    // step 3, compute x=Q*y
    lapack_kernel::gemv('N', value_type(1), _mat, _work, value_type(0), x);
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
  using _base::_mat;                        ///< matrix
  using _base::_mat_backup;                 ///< matrix backup
  using _base::_rank;                       ///< rank
  Array<scalar_type>         _w;            ///< Eigenvalues
  mutable Array<value_type>  _work;         ///< work buffer
  mutable Array<scalar_type> _rwork;        ///< rwork buffer
  Array<int>                 _trunc_order;  ///< truncated dimensions
};

}  // namespace hif

#endif  // _HIF_SMALLSCALE_SYEIG_HPP