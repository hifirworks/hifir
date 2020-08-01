///////////////////////////////////////////////////////////////////////////////
//                This file is part of HILUCSI project                       //
///////////////////////////////////////////////////////////////////////////////

/*!
 * \file hilucsi/small_scale/QRCP.hpp
 * \brief Small scale solver with dense QR with column pivoting
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

#ifndef _HILUCSI_SMALLSCALE_QRCP_HPP
#define _HILUCSI_SMALLSCALE_QRCP_HPP

#include <algorithm>

#include "hilucsi/ds/Array.hpp"
#include "hilucsi/small_scale/SmallScaleBase.hpp"
#include "hilucsi/small_scale/lapack.hpp"
#include "hilucsi/utils/common.hpp"
#include "hilucsi/utils/math.hpp"

namespace hilucsi {

/// \class QRCP
/// \brief QR with column pivoting for small scale solver
/// \tparam ValueType value type, e.g. \a double
/// \ingroup sss
template <class ValueType>
class QRCP : public internal::SmallScaleBase<ValueType> {
  using _base = internal::SmallScaleBase<ValueType>;

 public:
  typedef ValueType                 value_type;     ///< value type
  typedef Lapack<value_type>        lapack_kernel;  ///< lapack backend
  typedef typename _base::size_type size_type;      ///< size type
  typedef typename ValueTypeTrait<value_type>::value_type scalar_type;
  ///< scalar type

  /// \brief get the solver type
  inline static const char *method() { return "QRCP"; }

  /// \brief default constructor
  QRCP() = default;

  /// \brief compute decomposition and determine the rank
  inline void factorize() {
    // get tolerance
    constexpr static scalar_type diag_tol = std::sqrt(Const<scalar_type>::EPS);
    constexpr static scalar_type cond_tol =
        std::sqrt(1 / Const<scalar_type>::EPS);

    hilucsi_error_if(_mat.empty(), "matrix is still empty!");
    hilucsi_error_if(_mat.nrows() < _mat.ncols(),
                     "matrix must have no smaller row size");

    _jpvt.resize(_mat.ncols());
    // NOTE we need to initialize jpvt to zero
    std::fill(_jpvt.begin(), _jpvt.end(), 0);
    _tau.resize(std::min(_mat.nrows(), _mat.ncols()));

    // query the buffer size
    value_type lwork;
    lapack_kernel::geqp3(_mat.nrows(), _mat.ncols(), _mat.data(), _mat.nrows(),
                         _jpvt.data(), _tau.data(), &lwork, -1);
    _work.resize((size_type)lwork);

    const auto info = lapack_kernel::geqp3(
        _mat.nrows(), _mat.ncols(), _mat.data(), _mat.nrows(), _jpvt.data(),
        _tau.data(), _work.data(), (size_type)lwork);
    hilucsi_error_if(info < 0, "GEQP3 returned negative info");

    // fast filtering to test diagonal entries
    bool            do_cond_test = false;
    const size_type N            = _tau.size();  // min (row, col)
    // we first check if we have too small diagonals, use this as a safe guard
    // to see if we need to call more expensive condition # estimator
    // weighted by R(0,0)
    const scalar_type diag_eps = diag_tol * std::abs(_mat[0]);
    for (size_type i = N; i != 0u; --i)
      if (std::abs(_mat(i - 1, i - 1)) < diag_eps) {
        hilucsi_warning(
            "System ill-conditioned (diagonal %zd is smaller than tolerance "
            "%g), switch to condition number estimator to "
            "determine the final rank",
            i, (double)diag_eps);
        do_cond_test = true;
        break;
      }
    // initial rank to be full rank
    _rank = _mat.ncols();
    if (do_cond_test) {
#if 0
      // allocate buffer
      _work.resize(3 * _mat.ncols());
      _iwork.resize(_mat.ncols());
      scalar_type rcond;  // reciprocal of condition number
      for (;;) {
        // use 1 norm
        const auto info = lapack_kernel::trcon('1', 'U', 'N', _rank,
                                               _mat.data(), _mat.nrows(), rcond,
                                               _work.data(), _iwork.data());
        hilucsi_error_if(info < 0, "TRCON returned negative info!");
        if (rcond * cond_tol >= scalar_type(1)) break;
        if (_rank == 0u) break;
        --_rank;
      }
#else
      // _est_rank(cond_tol);
      _est_rank_2norm(cond_tol);
#endif
      hilucsi_warning_if(
          _rank != _mat.ncols(),
          "The system is rank deficient with rank=%zd, the tolerance used "
          "for thresholding reciprocal of 1-norm based condition number was %g",
          _rank, (double)cond_tol);
    }
  }

  /// \brief solve system
  /// \param[in,out] x input rhs, output solution
  ///
  /// First, QRCP returns \f$\mathbf{AP}=\mathbf{QR}\f$, thus when we
  /// have \f$\mathbf{Ax}=\mathbf{b}\f$, the derivation is:
  ///
  /// \f{eqnarray*}{
  ///   \mathbf{QRP}^T\mathbf{x}&=&\mathbf{b} \\
  ///   \hookrightarrow\mathbf{RP}^T\mathbf{x}&=&\mathbf{Q}^T\mathbf{b} \\
  ///   \hookrightarrow\mathbf{P}^T\mathbf{x}&=&\mathbf{R}^{-1}
  ///     \mathbf{Q}^T\mathbf{b} \\
  ///   \hookrightarrow\mathbf{x}&=&\mathbf{PR}^{-1}\mathbf{Q}^T\mathbf{b}
  /// \f}
  ///
  /// Notice that \f$\mathbf{R}^{-1}\mathbf{Q}^T\f$ is the
  /// pseudo-inverse of \f$\mathbf{AP}\f$.
  inline void solve(Array<value_type> &x, const bool tran = false) const {
    using v_t = scalar_type;

    hilucsi_error_if(tran, "QRCP does not support transpose solve!");
    hilucsi_error_if(
        _mat.empty() || _jpvt.empty(),
        "either the matrix is not set or the factorization has not yet done!");
    hilucsi_error_if(x.size() != _mat.nrows(),
                     "unmatched sizes between system and rhs");
    // query the optimal work space
    v_t lwork;
    lapack_kernel::ormqr('L', 'T', x.size(), 1, _rank, _mat.data(),
                         _mat.nrows(), _tau.data(), x.data(), x.size(), &lwork,
                         -1);
    _work.resize((size_type)lwork);

    // compute x=Q(:,1:_rank)'*x
    const auto info = lapack_kernel::ormqr(
        'L', 'T', x.size(), 1, _rank, _mat.data(), _mat.nrows(), _tau.data(),
        x.data(), x.size(), _work.data(), (size_type)lwork);
    hilucsi_error_if(info < 0, "ORMQR returned negative info (Q'*x)");

    // compute x(1:_rank)=inv(R(1:_rank,1:_rank))*x(1:_rank)
    lapack_kernel::trsv('U', 'N', 'N', _rank, _mat.data(), _mat.nrows(),
                        x.data(), 1);

    // permutation, need to loop thru all entries in jpvt. Also, note that jpvt
    // is column pivoting, thus we actually need the inverse mapping while
    // performing P*x. It's worth noting that, unlike ipiv in LU routine, jpvt
    // is a real permutation vector, i.e. not in-place swapable. Therefore, we
    // need a buffer.
    //
    // for normal mapping, we have x=b(p), for inverse, we need x(p)=b
    const size_type n = _jpvt.size();
    _work.resize(x.size());
    for (size_type i = 0u; i < n; ++i) _work[_jpvt[i] - 1] = x[i];
    std::copy_n(_work.cbegin(), n, x.begin());
  }

  /// \brief wrapper if input is different from \a value_type
  template <class ArrayType>
  inline void solve(ArrayType &x, const bool tran = false) const {
    _x.resize(x.size());
    std::copy(x.cbegin(), x.cend(), _x.begin());
    solve(_x, tran);
    std::copy(_x.cbegin(), _x.cend(), x.begin());
  }

 protected:
  /// \brief Incrementally estimate the condition number of R
  /// \param[in] cond_tol condition number threshold
  /// \note We incrementally check based on \f$\mathbf{R}^{T}\f$ by 1-norm
  inline void _est_rank(const scalar_type cond_tol) {
    const size_type n = _mat.ncols();
    _work.resize(n);  // allocate buffer, used in _est_cond1
    scalar_type nrm, inrm;
    for (_rank = 0u; _rank < n; ++_rank) {
      // esimate norm of the current leading block
      if (_mat(_rank, _rank) == 0) break;
      _est_cond1(inrm, nrm);
      if (inrm * nrm >= cond_tol) break;
    }
  }

  /// \brief helper function to estimate the norms of \f$\mathbf{R}^T\f$ and
  ///        \f$\mathbf{R}^{-T}\f$.
  /// \param[in,out] inrm invser norm at step _rank
  /// \param[in,out] nrm norm at step _rank
  inline void _est_cond1(scalar_type &inrm, scalar_type &nrm) {
    // NOTE, the upper part in _mat is R
    // NOTE, keep in mind that we are dealing with $R^{T}$
    if (_rank == 0u) {
      _work[0] = value_type(1) / _mat(0, 0);
      inrm     = abs(_work[0]);
      nrm      = abs(_mat(0, 0));
    } else {
      value_type  s(0);
      scalar_type nn(0);
      // backsolve
      for (size_type j(0); j < _rank; ++j) {
        s += _work[j] * _mat(j, _rank);
        nn += abs(_mat(j, _rank));
      }
      const value_type k1 = value_type(1) - s, k2 = -value_type(1) - s;
      _work[_rank] =
          abs(k1) < abs(k2) ? k2 / _mat(_rank, _rank) : k1 / _mat(_rank, _rank);
      // compute the invser 1-norm for R^T_{0:rank,0:rank}
      inrm = std::max(inrm, abs(_work[_rank]));
      // compute the 1-norm for R^T_{0:rank,0:rank}
      nrm = std::max(nrm, nn + abs(_mat(_rank, _rank)));
    }
  }

  /// \brief esimate rank based on 2-norm condition numbers
  /// \param[out] cond_tol condition number threshold
  /// \note This is calling after QRCP by using \a ?laic1 auxiliary routine
  /// \note This implemention is adapted based on ?gelsy driver routine
  inline void _est_rank_2norm(const scalar_type cond_tol) {
    static constexpr typename lapack_kernel::int_type JOB_MIN = 2, JOB_MAX = 1;

    const size_type n = _mat.ncols();
    _work.resize(2 * n);                 // allocate buffer
    auto *x = _work.data(), *y = x + n;  // used in ?laic1
    x[0] = y[0] = value_type(1);
    value_type smax(abs(_mat(0, 0))), smin(smax);  // initial singular values
    value_type s1, c1, s2, c2, sminpr, smaxpr;     // needed in ?laic1
    for (_rank = 0u; _rank < n; ++_rank) {
      // esimate smaller
      lapack_kernel::laic1(JOB_MIN, _rank, x, smin, &_mat(0, _rank),
                           _mat(_rank, _rank), sminpr, s1, c1);
      // esimate larger
      lapack_kernel::laic1(JOB_MAX, _rank, y, smax, &_mat(0, _rank),
                           _mat(_rank, _rank), smaxpr, s2, c2);
      if (smaxpr <= sminpr * cond_tol) {
        // still well-conditioned
        for (size_type i(0); i < _rank; ++i) {
          x[i] *= s1;
          y[i] *= s2;
        }
        x[_rank] = c1;
        y[_rank] = c2;
        smin     = sminpr;
        smax     = smaxpr;
        continue;
      }
      break;
    }
  }

 protected:
  using _base::_mat;                         ///< matrix
  using _base::_rank;                        ///< rank
  Array<hilucsi_lapack_int>         _jpvt;   ///< column pivoting array
  Array<value_type>                 _tau;    ///< tau array
  mutable Array<value_type>         _work;   ///< work buffer
  mutable Array<hilucsi_lapack_int> _iwork;  ///< integer work buffer
  mutable Array<value_type>         _x;      ///< buffer for different type
};

}  // namespace hilucsi

#endif  // _HILUCSI_SMALLSCALE_QRCP_HPP