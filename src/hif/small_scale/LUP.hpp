///////////////////////////////////////////////////////////////////////////////
//                  This file is part of the HIFIR library                   //
///////////////////////////////////////////////////////////////////////////////

/*!
 * \file hif/small_scale/LUP.hpp
 * \brief Small scale solver with dense LU with partial pivoting
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

#ifndef _HIF_SMALLSCALE_LUP_HPP
#define _HIF_SMALLSCALE_LUP_HPP

#include <algorithm>
#include <array>

#include "hif/Options.h"
#include "hif/ds/Array.hpp"
#include "hif/ds/DenseMatrix.hpp"
#include "hif/small_scale/lapack.hpp"

namespace hif {

/// \class LUP
/// \brief LU with partial pivoting for small scale solver
/// \tparam ValueType value type, e.g. \a double
/// \ingroup sss
template <class ValueType>
class LUP {
 public:
  typedef ValueType                      value_type;     ///< value type
  typedef DenseMatrix<value_type>        dense_type;     ///< dense matrix
  typedef Lapack<value_type>             lapack_kernel;  ///< lapack backend
  typedef typename dense_type::size_type size_type;      ///< size type

  /// \brief get the solver type
  inline static const char *method() { return "LUP"; }

  /// \brief default constructor
  LUP() { _rank = 0u; }

  // utilities
  inline bool              empty() const { return _mat.empty(); }
  inline size_type         rank() const { return _rank; }
  inline const dense_type &mat() const { return _mat; }
  inline dense_type &      mat() { return _mat; }
  inline const dense_type &mat_backup() const { return _mat_backup; }
  inline dense_type &      mat_backup() { return _mat_backup; }
  inline bool              is_squared() const { return _mat.is_squared(); }
  inline bool full_rank() const { return _rank != 0u && _rank == _mat.ncols(); }

  /// \brief set operator
  /// \tparam CsType compressed storage type, see \ref CCS and \ref CRS
  /// \param[in] cs compressed storage
  template <class CsType>
  inline void set_matrix(const CsType &cs) {
    _mat = dense_type::from_sparse(cs);
    _mat_backup.resize(_mat.nrows(), _mat.ncols());
    std::copy(_mat.array().cbegin(), _mat.array().cend(),
              _mat_backup.array().begin());
  }

  /// \brief set a dense operator, this is needed for H version
  /// \param[in,out] mat input matrix, the data is \b destroyed upon output
  inline void set_matrix(dense_type &&mat) {
    _mat = std::move(mat);
    _mat_backup.resize(_mat.nrows(), _mat.ncols());
    std::copy(_mat.array().cbegin(), _mat.array().cend(),
              _mat_backup.array().begin());
  }

  /// \brief set a dense operator from other data type
  /// \tparam T value type
  template <class T>
  inline void set_matrix(const DenseMatrix<T> &mat) {
    _mat = dense_type(mat);
    _mat_backup.resize(_mat.nrows(), _mat.ncols());
    std::copy(_mat.array().cbegin(), _mat.array().cend(),
              _mat_backup.array().begin());
  }

  /// \brief perform LU with partial pivoting
  /// \param[in] opts control parameters, see \ref Options
  inline void factorize(const Options &opts) {
    hif_error_if(_mat.empty(), "matrix is still empty!");
    hif_error_if(!is_squared(), "the matrix must be squared!");
    if (hif_verbose(INFO, opts)) hif_info("factorizing dense level by LU...");

    _ipiv.resize(_mat.nrows());
    const auto info = lapack_kernel::getrf(_mat, _ipiv);
    if (info < 0)
      hif_error("GETRF returned negative info!");
    else if (info > 0) {
      if (hif_verbose(WARN, opts)) {
        hif_warning(
            "GETRF returned positive info, U(%zd,%zd) is exactly zero! "
            "Consider using QRCP for small scale solver!",
            (size_type)info, (size_type)info);
      }
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
  inline void solve(ArrayType &x, const size_type /* rank */ = 0,
                    const bool tran = false) const {
    _x.resize(x.size());
    std::copy(x.cbegin(), x.cend(), _x.begin());
    solve(_x, size_type(0), tran);
    std::copy(_x.cbegin(), _x.cend(), x.begin());
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
    _mrhs.resize(x.size(), Nrhs);
    for (size_type j = 0; j < Nrhs; ++j)
      for (size_type i(0); i < x.size(); ++i) _mrhs(i, j) = x[i][j];
    if (lapack_kernel::getrs(_mat, _ipiv, _mrhs, tran ? 'T' : 'N') < 0)
      hif_error("GETRS returned negative info!");
    for (size_type j = 0; j < Nrhs; ++j)
      for (size_type i(0); i < x.size(); ++i) x[i][j] = _mrhs(i, j);
  }

  /// \brief compute \f$\mathbf{y}=\mathbf{P}^{T}\mathbf{LUx}\f$
  /// \param[in] x input vector
  /// \param[out] y output vector
  /// \param[in] tran (optional) tranpose flag
  inline void multiply(const Array<value_type> &x, Array<value_type> &y,
                       const size_type /* rank */ = 0,
                       const bool tran            = false) const {
    hif_error_if(x.size() != _mat_backup.nrows(),
                 "unmatched sizes between system and x");
    hif_error_if(x.size() != y.size(), "unmatched sizes x and y");
    lapack_kernel::gemv(tran ? 'C' : 'N', value_type(1), _mat_backup, x,
                        value_type(0), y);
  }

  /// \brief wrapper if \a value_type is different from input's
  template <class ArrayIn, class ArrayOut>
  inline void multiply(const ArrayIn &x, ArrayOut &y,
                       const size_type /* rank */ = 0,
                       const bool tran            = false) const {
    _x.resize(x.size());
    _y.resize(y.size());
    std::copy(x.cbegin(), x.cend(), _x.begin());
    multiply(_x, _y, size_type(0), tran);
    std::copy(_y.cbegin(), _y.cend(), y.begin());
  }

 protected:
  dense_type                _mat;         ///< matrix
  dense_type                _mat_backup;  ///< backup matrix
  size_type                 _rank;        ///< rank
  mutable Array<value_type> _x;     ///< buffer for handling solve in derived
  mutable Array<value_type> _y;     ///< buffer for handling mv in derived
  mutable dense_type        _mrhs;  ///< multiple RHS buffer
  Array<hif_lapack_int>     _ipiv;  ///< row pivoting array
};

}  // namespace hif

#endif  // _HIF_SMALLSCALE_LUP_HPP
