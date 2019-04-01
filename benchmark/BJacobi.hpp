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

template <class BlockDiag>
class BJacobi {
 public:
  typedef BlockDiag                                        block_diag_type;
  typedef typename block_diag_type::array_type             array_type;
  typedef typename array_type::value_type                  value_type;
  typedef typename array_type::size_type                   size_type;
  typedef typename internal::ScalarTrait<value_type>::type scalar_type;

 public:
  BJacobi() = delete;

  explicit BJacobi(const block_diag_type &D, const size_type max_it = 0u,
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
    return verbose ? _solve(A, b, x0, internal::Cout, internal::Cerr)
                   : _solve(A, b, x0, internal::Dummy_streamer,
                            internal::Dummy_streamer);
  }

  template <class Operator>
  inline std::pair<int, size_type> solve(const Operator &A, const array_type &b,
                                         array_type &x,
                                         const bool  verbose = true) const {
    std::fill(x.begin(), x.end(), value_type(0));
    return solve_with_guess(A, b, x, verbose);
  }

 protected:
  template <class Operator, class CoutStream, class CerrStream>
  inline std::pair<int, size_type> _solve(const Operator &  A,
                                          const array_type &b, array_type &x0,
                                          const CoutStream &Cout,
                                          const CerrStream &Cerr) const {
    int       flag(BJACOBI_INVALID_PARS);
    size_type iters(0);
    if (b.size() != x0.size() || _D.empty()) return std::make_pair(flag, iters);
    try {
      _xk.resize(b.size());
      _r.resize(b.size());
    } catch (const std::exception &e) {
      Cerr("resize buffer error with exception:\n%s\n", e.what());
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
          Cout(" At iteration %zd, residual is %g\n", iters + 1, (double)res);
#endif  // PSMILU_BJACOBI_REPORT_RES
        // compute inv(D)*res
          _D.solve(x, _r);
          for (size_type i(0); i < n; ++i) x[i] = _r[i] + _xk[i];
          const auto err =
              std::sqrt(std::inner_product(_r.cbegin(), _r.cend(), _r.cbegin(),
                                           value_type(0)) /
                        std::inner_product(x.cbegin(), x.cend(), x.cbegin(),
                                           value_type(0)));
          Cout(
              " At iteration %zd, relative error |x(%zd)-x(%zd)|/|x(%zd)| is "
              "%g\n",
              iters + 1, iters + 1, iters, iters + 1, err);
          if (err <= rtol) {
            flag = BJACOBI_SUCCESS;
            break;
          }
        }
        if (flag != BJACOBI_SUCCESS)
          Cerr("Could not find the solution within %zd iterations\n", maxit);
        return std::make_pair(flag, iters);
      } catch (const std::exception &e) {
        Cerr("Exception thrown inside loop body (iter=%zd) with message:\n%s\n",
             iters, e.what());
        flag = BJACOBI_UNKNOWN_ERROR;
        return std::make_pair(flag, iters);
      }
    } else {
      Cout("exactly zero rhs detected\n");
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

}  // namespace bench
}  // namespace psmilu