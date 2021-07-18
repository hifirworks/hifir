///////////////////////////////////////////////////////////////////////////////
//                  This file is part of the HIFIR library                   //
///////////////////////////////////////////////////////////////////////////////

/*!
 * \file hif/small_scale/SYEIG.hpp
 * \brief Small scale solver with symmetric eigendecomposition
 * \author Qiao Chen

\verbatim
Copyright (C) 2020 NumGeom Group at Stony Brook University

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

#ifndef _HIF_SMALLSCALE_SYEIG_HPP
#define _HIF_SMALLSCALE_SYEIG_HPP

#include <algorithm>
#include <array>
#include <numeric>

#include "hif/Options.h"
#include "hif/ds/Array.hpp"
#include "hif/ds/DenseMatrix.hpp"
#include "hif/small_scale/lapack.hpp"
#include "hif/utils/common.hpp"
#include "hif/utils/math.hpp"

namespace hif {

/// \class SYEIG
/// \brief Symmetric eigen-decompsition for small scale solver
/// \tparam ValueType value type, e.g. \a double
/// \ingroup sss
template <class ValueType>
class SYEIG {
 public:
  typedef ValueType                      value_type;     ///< value type
  typedef DenseMatrix<value_type>        dense_type;     ///< dense type
  typedef typename dense_type::size_type size_type;      ///< size type
  typedef Lapack<value_type>             lapack_kernel;  ///< lapack backend
  typedef typename ValueTypeTrait<value_type>::value_type scalar_type;
  ///< scalar type

  /// \brief get the solver type
  inline static const char *method() { return "SYEIG"; }

  /// \brief default constructor
  SYEIG() { _rank = 0u; }

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

  /// \brief compute decomposition and determine the rank
  /// \param[in] opts control parameters, see \ref Options
  inline void factorize(const Options &opts) {
    hif_error_if(_mat.empty(), "matrix is still empty!");
    hif_error_if(!is_squared(), "the matrix must be squared!");
    if (hif_verbose(INFO, opts))
      hif_info("factorizing dense level by symmetric (%s) eigendecomp...",
               opts.spd > 0 ? "PD" : (opts.spd < 0 ? "ND" : "ID"));

    const static scalar_type EPS = std::pow(Const<scalar_type>::EPS, 2. / 3);

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

  /// \brief solve \f$\mathbf{Q\Lambda Q}^H\mathbf{x}=\mathbf{b}\f$
  /// \param[in,out] x input rhs, output solution
  /// \param[in] rank (optional) numerical rank
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
    _x.resize(x.size());
    std::copy(x.cbegin(), x.cend(), _x.begin());
    solve(_x, rank);
    std::copy(_x.cbegin(), _x.cend(), x.begin());
  }

  /// \brief solve with multiple RHS
  /// \sa solve
  template <class V, size_type Nrhs>
  inline void solve_mrhs(Array<std::array<V, Nrhs>> &x,
                         const size_type             rank = 0) const {
    hif_error_if(
        _mat.empty() || _w.empty(),
        "either the matrix is not set or the factorization has not yet done!");
    hif_error_if(x.size() != _mat.nrows(),
                 "unmatched sizes between system and rhs");
    const auto      n  = x.size();
    const size_type rk = rank == 0u ? _rank : (rank > n ? n : rank);
    // copy to internal column-major buffer
    _mrhs.resize(x.size(), Nrhs);
    for (size_type j = 0; j < Nrhs; ++j)
      for (size_type i(0); i < x.size(); ++i) _mrhs(i, j) = x[i][j];
    for (size_type i = 0; i < Nrhs; ++i) {
      // step 1, compute y=Q^T*x
      lapack_kernel::gemv('C', _mat.nrows(), _mat.ncols(), value_type(1),
                          _mat.data(), _mat.nrows(), &_mrhs(0, i * Nrhs),
                          value_type(0), _work.data());
      // step 2, solve inv(lambda)*y with truncation
      for (size_type i(0); i < rk; ++i)
        _work[_trunc_order[i]] /= _w[_trunc_order[i]];
      for (size_type i(rk); i < n; ++i) _work[_trunc_order[i]] = 0;
      // step 3, compute x=Q*y
      lapack_kernel::gemv('N', _mat.nrows(), _mat.ncols(), value_type(1),
                          _mat.data(), _mat.nrows(), _work.data(),
                          value_type(0), &_mrhs(0, i * Nrhs));
    }
    // copy back to the application
    for (int j = 0; j < Nrhs; ++j)
      for (size_type i(0); i < x.size(); ++i) x[i][j] = _mrhs(i, j);
  }

  /// \brief matrix-vector \f$\mathbf{Q\Lambda Q}^H\mathbf{x}\f$
  /// \param[in,out] x input rhs, output solution
  /// \param[in] rank (optional) numerical rank
  inline void multiply(Array<value_type> &x, const size_type rank = 0) const {
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
    // step 2, compute lambda*y with truncation
    for (size_type i(0); i < rk; ++i)
      _work[_trunc_order[i]] *= _w[_trunc_order[i]];
    for (size_type i(rk); i < n; ++i) _work[_trunc_order[i]] = 0;
    // step 3, compute x=Q*y
    lapack_kernel::gemv('N', value_type(1), _mat, _work, value_type(0), x);
  }

  /// \brief wrapper if \a value_type is different from input's for mv
  template <class ArrayType>
  inline void multiply(ArrayType &x, const size_type rank = 0) const {
    _x.resize(x.size());
    std::copy(x.cbegin(), x.cend(), _x.begin());
    multiply(_x, rank);
    std::copy(_x.cbegin(), _x.cend(), x.begin());
  }

  /// \brief matrix-vector with multiple RHS
  /// \sa multiply
  template <class V, size_type Nrhs>
  inline void multiply_mrhs(Array<std::array<V, Nrhs>> &x,
                            const size_type             rank = 0) const {
    hif_error_if(
        _mat.empty() || _w.empty(),
        "either the matrix is not set or the factorization has not yet done!");
    hif_error_if(x.size() != _mat.nrows(),
                 "unmatched sizes between system and rhs");
    const auto      n  = x.size();
    const size_type rk = rank == 0u ? _rank : (rank > n ? n : rank);
    // copy to internal column-major buffer
    _mrhs.resize(x.size(), Nrhs);
    for (size_type j = 0; j < Nrhs; ++j)
      for (size_type i(0); i < x.size(); ++i) _mrhs(i, j) = x[i][j];
    for (size_type i = 0; i < Nrhs; ++i) {
      // step 1, compute y=Q^T*x
      lapack_kernel::gemv('C', _mat.nrows(), _mat.ncols(), value_type(1),
                          _mat.data(), _mat.nrows(), &_mrhs(0, i * Nrhs),
                          value_type(0), _work.data());
      // step 2, compute lambda*y with truncation
      for (size_type i(0); i < rk; ++i)
        _work[_trunc_order[i]] *= _w[_trunc_order[i]];
      for (size_type i(rk); i < n; ++i) _work[_trunc_order[i]] = 0;
      // step 3, compute x=Q*y
      lapack_kernel::gemv('N', _mat.nrows(), _mat.ncols(), value_type(1),
                          _mat.data(), _mat.nrows(), _work.data(),
                          value_type(0), &_mrhs(0, i * Nrhs));
    }
    // copy back to the application
    for (int j = 0; j < Nrhs; ++j)
      for (size_type i(0); i < x.size(); ++i) x[i][j] = _mrhs(i, j);
  }

 protected:
  dense_type                 _mat;         ///< matrix
  dense_type                 _mat_backup;  ///< backup matrix
  size_type                  _rank;        ///< rank
  mutable Array<value_type>  _x;      ///< buffer for handling solve in derived
  mutable Array<value_type>  _y;      ///< buffer for handling mv in derived
  mutable dense_type         _mrhs;   ///< multiple RHS buffer
  Array<scalar_type>         _w;      ///< Eigenvalues
  mutable Array<value_type>  _work;   ///< work buffer
  mutable Array<scalar_type> _rwork;  ///< rwork buffer
  Array<int>                 _trunc_order;  ///< truncated dimensions
};

}  // namespace hif

#endif  // _HIF_SMALLSCALE_SYEIG_HPP