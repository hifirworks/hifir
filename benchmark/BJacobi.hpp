//@HEADER
//----------------------------------------------------------------------------
//                Copyright (C) 2019 The PSMILU AUTHORS
//----------------------------------------------------------------------------
//@HEADER

// This is a template implementation block Jacobi iteration

#pragma once

#include "GMRES.hpp"

namespace psmilu {
namespace bench {

enum {
  BJACOBI_UNKNOWN_ERROR = -2,
  BJACOBI_INVALID_PARS  = -1,
  BJACOBI_SUCCESS       = 0,
  BJACOBI_DIVERGED      = 1,
};

namespace internal {

template <class Child, class BlockDiag>
class BJacobiBase {
  inline const Child &_this() const {
    return static_cast<const Child &>(*this);
  }

 public:
  typedef BlockDiag                                        block_diag_type;
  typedef typename block_diag_type::array_type             array_type;
  typedef typename array_type::value_type                  value_type;
  typedef typename array_type::size_type                   size_type;
  typedef typename internal::ScalarTrait<value_type>::type scalar_type;

  BJacobiBase() = delete;

  explicit BJacobiBase(const block_diag_type &D, const size_type max_it = 0u,
                       const scalar_type tol = 0)
      : _D(D),
        _xk(),
        _r(),
        maxit(max_it ? max_it : 500u),
        rtol(tol > scalar_type(0) ? tol : scalar_type(1e-6)) {}

  template <class Operator>
  inline std::pair<int, size_type> solve_with_guess(
      const Operator &A, const array_type &b, array_type &x0,
      const bool verbose = true) const {
    return verbose ? _solve(A, b, x0, Cout, Cerr)
                   : _solve(A, b, x0, Dummy_streamer, Dummy_streamer);
  }

  template <class Operator>
  inline std::pair<int, size_type> solve(const Operator &A, const array_type &b,
                                         array_type &x,
                                         const bool  verbose = true) const {
    std::fill(x.begin(), x.end(), value_type(0));
    return solve_with_guess(A, b, x, verbose);
  }

 protected:
  // The Jacobi iteration reads:
  //    A*x=b
  //    (D+A-D)*x=b
  //    D*x_{k+1}=b-(A-D)*x_k
  //    x_{k+1}=inv(D)*(b-(A-D)*x_k)
  //    x_{k+1}=inv(D)*(b-A*x_k+D*x_k)
  //    x_{k+1}=inv(D)*res_k+x_k
  template <class Operator, class CoutStream, class CerrStream>
  inline std::pair<int, size_type> _solve(const Operator &  A,
                                          const array_type &b, array_type &x0,
                                          const CoutStream &Cout_,
                                          const CerrStream &Cerr_) const {
    int       flag(BJACOBI_INVALID_PARS);
    size_type iters(0);
    if (b.size() != x0.size() || _D.empty()) return std::make_pair(flag, iters);
    try {
      _xk.resize(b.size());
      _r.resize(b.size());
      _this()._prepare(A, b, x0);
    } catch (const std::exception &e) {
      Cerr_("resize buffer error with exception:\n%s\n", e.what());
      flag = BJACOBI_UNKNOWN_ERROR;
      return std::make_pair(flag, iters);
    }
    const scalar_type beta = std::sqrt(
        std::inner_product(b.cbegin(), b.cend(), b.cbegin(), value_type(0)));
    // iteration
    // x=inv(D)*(b-A*xk)+xk
    flag = BJACOBI_DIVERGED;
    if (beta != scalar_type(0)) {
      const size_type n(b.size());
      const auto      beta_inv = 1. / beta;
      auto &          x        = x0;
      try {
        for (; iters < maxit; ++iters) {
          std::copy(x.cbegin(), x.cend(), _xk.begin());
          // compute residual, b-A*xk, store to x
          A.mv(_xk, x);
          for (size_type i(0); i < n; ++i) x[i] = b[i] - x[i];
#ifdef PSMILU_BJACOBI_REPORT_RES
          const auto res =
              std::sqrt(std::inner_product(x.cbegin(), x.cend(), x.cbegin(),
                                           value_type(0))) *
              beta_inv;
          Cout_(" At iteration %zd, residual is %g\n", iters + 1, (double)res);
#endif  // PSMILU_BJACOBI_REPORT_RES
        // compute inv(D)*res
          _D.solve(x, _r);
          // for (size_type i(0); i < n; ++i) x[i] = _r[i] + _xk[i];
          _this()._post_process(iters, x);
          const auto err =
              std::sqrt(std::inner_product(_r.cbegin(), _r.cend(), _r.cbegin(),
                                           value_type(0)) /
                        std::inner_product(x.cbegin(), x.cend(), x.cbegin(),
                                           value_type(0)));
          Cout_(
              " At iteration %zd, relative error |x(%zd)-x(%zd)|/|x(%zd)| is "
              "%g\n",
              iters + 1, iters + 1, iters, iters + 1, err);
          if (err <= rtol) {
            flag = BJACOBI_SUCCESS;
            break;
          }
        }
        if (flag != BJACOBI_SUCCESS)
          Cerr_("Could not find the solution within %zd iterations\n", maxit);
        return std::make_pair(flag, iters);
      } catch (const std::exception &e) {
        Cerr_(
            "Exception thrown inside loop body (iter=%zd) with message:\n%s\n",
            iters, e.what());
        flag = BJACOBI_UNKNOWN_ERROR;
        return std::make_pair(flag, iters);
      }
    } else {
      Cout_("exactly zero rhs detected\n");
      std::fill(x0.begin(), x0.end(), value_type(0));
      flag = BJACOBI_SUCCESS;
      return std::make_pair(flag, iters);
    }
  }

 protected:
  const block_diag_type &_D;
  mutable array_type     _xk;
  mutable array_type     _r;

 public:
  size_type   maxit;
  scalar_type rtol;
};
}  // namespace internal

template <class BlockDiag>
class BJacobi : public internal::BJacobiBase<BJacobi<BlockDiag>, BlockDiag> {
  using _base = internal::BJacobiBase<BJacobi<BlockDiag>, BlockDiag>;
  friend _base;

 public:
  typedef typename _base::block_diag_type block_diag_type;
  typedef typename _base::array_type      array_type;
  typedef typename _base::size_type       size_type;
  typedef typename _base::scalar_type     scalar_type;

  BJacobi() = delete;
  explicit BJacobi(const block_diag_type &D, const size_type max_it = 0u,
                   const scalar_type tol = 0)
      : _base(D, max_it, tol) {}

 protected:
  using _base::_r;
  using _base::_xk;
  template <class Operator>
  inline void _prepare(const Operator &, const array_type &,
                       array_type &) const {}
  inline void _post_process(const size_type, array_type &x) const {
    const size_type n = x.size();
    for (size_type i(0); i < n; ++i) x[i] = _r[i] + _xk[i];
  }
};

// with Chebyshev polynomial acceleration
template <class BlockDiag>
class CBJacobi : public internal::BJacobiBase<CBJacobi<BlockDiag>, BlockDiag> {
  using _base = internal::BJacobiBase<CBJacobi<BlockDiag>, BlockDiag>;
  friend _base;

 public:
  typedef typename _base::block_diag_type block_diag_type;
  typedef typename _base::array_type      array_type;
  typedef typename _base::size_type       size_type;
  typedef typename _base::value_type      value_type;
  typedef typename _base::scalar_type     scalar_type;

  CBJacobi() = delete;
  explicit CBJacobi(const block_diag_type &D, const value_type lambda1 = 0.9,
                    const value_type lambda2 = -0.9, const size_type max_it = 0u,
                    const scalar_type tol = 0)
      : _base(D, max_it, tol), _rho(2), _xkk() {
    _compute_coeffs(lambda1, lambda2);
  }

 protected:
  mutable value_type _rho;
  mutable array_type _xkk;
  value_type         _gamma;
  value_type         _sigma2;
  using _base::_r;
  using _base::_xk;

 protected:
  inline void _compute_coeffs(const value_type l1, const value_type l2) {
    _gamma  = 2. / (2. - l1 - l2);
    _sigma2 = 0.5 * _gamma * (l1 - l2);
    _sigma2 = _sigma2 * _sigma2;
  }
  template <class Operator>
  inline void _prepare(const Operator &, const array_type &b,
                       array_type &) const {
    _xkk.resize(b.size());
  }

  inline void _post_process(const size_type iter, array_type &x) const {
    const size_type n = x.size();
    for (size_type i(0); i < n; ++i) x[i] = _xk[i] + _gamma * _r[i];
    if (iter) {
      // NOTE _rho is initialized as 2, thus it valids for iter==1
      _rho            = 1. / (1. - 0.25 * _sigma2 * _rho);
      const auto beta = 1. - _rho;
      for (size_type i(0); i < n; ++i) x[i] = _rho * x[i] + beta * _xkk[i];
    }
    // update error
    for (size_type i(0); i < n; ++i) _r[i] = x[i] - _xk[i];
    std::copy(_xk.cbegin(), _xk.cend(), _xkk.begin());
  }
};
}  // namespace bench
}  // namespace psmilu
