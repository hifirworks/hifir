//@HEADER
//----------------------------------------------------------------------------
//                Copyright (C) 2019 The PSMILU AUTHORS
//----------------------------------------------------------------------------
//@HEADER

/// \file psmilu_small_scale/QRCP.hpp
/// \brief Small scale solver with dense QR with column pivoting
/// \authors Qiao,

#ifndef _PSMILU_SMALLSCALE_QRCP_HPP
#define _PSMILU_SMALLSCALE_QRCP_HPP

#include <algorithm>

#include "SmallScaleBase.hpp"
#include "psmilu_Array.hpp"
#include "psmilu_lapack/lapack.hpp"
#include "psmilu_utils.hpp"

namespace psmilu {

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

  /// \brief get the solver type
  inline static const char *method() { return "QRCP"; }

  /// \brief default constructor
  QRCP() = default;

  /// \brief compute decomposition and determine the rank
  inline void factorize() {
    // get tolerance
    using v_t = typename ValueTypeTrait<value_type>::value_type;
    constexpr static bool IS_DOUBLE = std::is_same<v_t, double>::value;

    constexpr static v_t diag_tol = IS_DOUBLE ? 1e-7 : 1e-4f;
    constexpr static v_t cond_tol = IS_DOUBLE ? 1e-12 : 1e-6f;

    psmilu_error_if(_mat.empty(), "matrix is still empty!");
    psmilu_error_if(_mat.nrows() < _mat.ncols(),
                    "matrix must have no smaller row size");

    _jpvt.resize(_mat.ncols());
    // NOTE we need to initialize jpvt to zero
    std::fill(_jpvt.begin(), _jpvt.end(), 0);
    _tau.resize(std::min(_mat.nrows(), _mat.ncols()));

    // query the buffer size
    v_t lwork;
    lapack_kernel::geqp3(_mat.nrows(), _mat.ncols(), _mat.data(), _mat.nrows(),
                         _jpvt.data(), _tau.data(), &lwork, -1);
    _work.resize((size_type)lwork);

    const auto info = lapack_kernel::geqp3(
        _mat.nrows(), _mat.ncols(), _mat.data(), _mat.nrows(), _jpvt.data(),
        _tau.data(), _work.data(), (size_type)lwork);
    psmilu_error_if(info < 0, "GEQP3 returned negative info");

    // fast filtering to test diagonal entries
    bool            do_cond_test = false;
    const size_type N            = _tau.size();  // min (row, col)
    const v_t       diag_eps     = diag_tol * std::abs(_mat[0]);
    for (size_type i = 0u; i < N; ++i)
      if (std::abs(_mat(i, i)) < diag_eps) {
        psmilu_warning(
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
      const v_t cond_eps = cond_tol * std::abs(_mat[0]);
      // allocate buffer
      _work.resize(3 * _mat.ncols());
      _iwork.resize(_mat.ncols());
      v_t rcond;  // reciprocal of condition number
      for (;;) {
        // use 1 norm
        const auto info = lapack_kernel::trcon('1', 'U', 'N', _rank,
                                               _mat.data(), _mat.nrows(), rcond,
                                               _work.data(), _iwork.data());
        psmilu_error_if(info < 0, "TRCON returned negative info!");
        if (rcond >= cond_eps) break;
        if (_rank == 0u) break;
        --_rank;
      }
      psmilu_warning_if(
          _rank != _mat.ncols(),
          "The system is rank deficient with rank=%zd, the tolerance used "
          "for thresholding reciprocal of 1-norm based condition number was %g",
          _rank, (double)cond_eps);
    }
  }

  /// \brief solve system
  /// \param[in,out] x input rhs, output solution
  ///
  /// First, QRCP returns \f$\boldsymbol{AP}=\boldsymbol{QR}\f$, thus when we
  /// have \f$\boldsymbol{Ax}=\boldsymbol{b}\f$, the derivation is:
  ///
  /// \f[
  ///   \boldsymbol{QRP}^T\boldsymbol{x}&=\boldsymbol{b}
  /// \f]
  /// \f[
  ///   \hookrightarrow\boldsymbol{RP}^T\boldsymbol{x}&=\boldsymbol{Q}^T
  ///     \boldsymbol{b}
  /// \f]
  /// \f[
  ///   \hookrightarrow\boldsymbol{P}^T\boldsymbol{x}&=\boldsymbol{R}^{-1}
  ///     \boldsymbol{Q}^T\boldsymbol{b}
  /// \f]
  /// \f[
  ///   \hookrightarrow\boldsymbol{x}&=\boldsymbol{PR}^{-1}\boldsymbol{Q}^T
  ///     \boldsymbol{b}
  /// \f]
  ///
  /// Notice that \f$\boldsymbol{R}^{-1}\boldsymbol{Q}^T\f$ is the
  /// pseudo-inverse of \f$\boldsymbol{AP}\f$.
  inline void solve(Array<value_type> &x) const {
    using v_t = typename ValueTypeTrait<value_type>::value_type;

    psmilu_error_if(
        _mat.empty() || _jpvt.empty(),
        "either the matrix is not set or the factorization has not yet done!");
    psmilu_error_if(x.size() != _mat.nrows(),
                    "unmatched sizes between system and rhs");
    // query the optimal work space
    v_t lwork;
    lapack_kernel::ormqr('L', 'T', x.size(), 1, _rank, _mat.data(),
                         _mat.nrows(), _tau.data(), x.data(), _tau.size(),
                         &lwork, -1);
    _work.resize((size_type)lwork);

    // compute x=Q(:,1:_rank)'*x
    const auto info = lapack_kernel::ormqr(
        'L', 'T', x.size(), 1, _rank, _mat.data(), _mat.nrows(), _tau.data(),
        x.data(), _tau.size(), _work.data(), (size_type)lwork);
    psmilu_error_if(info < 0, "ORMQR returned negative info (Q'*x)");

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
    std::copy_n(_iwork.cbegin(), n, x.begin());
  }

 protected:
  using _base::_mat;                ///< matrix
  using _base::_rank;               ///< rank
  Array<psmilu_lapack_int> _jpvt;   ///< column pivoting array
  Array<value_type>        _tau;    ///< tau array
  Array<value_type>        _work;   ///< work buffer
  Array<psmilu_lapack_int> _iwork;  ///< integer work buffer
};

}  // namespace psmilu

#endif  // _PSMILU_SMALLSCALE_QRCP_HPP