//@HEADER
//----------------------------------------------------------------------------
//                Copyright (C) 2019 The PSMILU AUTHORS
//----------------------------------------------------------------------------
//@HEADER

// This is a template implementation of FGMRES with Jacobi iteration

#pragma once

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdio>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace psmilu {

namespace internal {
template <class T>
inline T conj(const T &v) {
  return std::conj(v);
}

template <>
inline double conj<double>(const double &v) {
  return v;
}

template <>
inline float conj<float>(const float &v) {
  return v;
}

template <class T>
struct ScalarTrait {
  using type = T;
};

template <class T>
struct ScalarTrait<std::complex<T>> {
  using type = T;
};

const static struct {
  template <class... Args>
  inline void operator()(const char *f, Args... args) const {
    std::fprintf(stdout, f, args...);
  }
} Cout;

const static struct {
  template <class... Args>
  inline void operator()(const char *f, Args... args) const {
    std::fprintf(stderr, f, args...);
  }
} Cerr;

const static struct {
  template <class... Args>
  inline void operator()(const char *, Args...) const {}
} Dummy_streamer;

template <class Child, class Diag>
class JacobiBase {
  inline const Child &_this() const {
    return static_cast<const Child &>(*this);
  }

 public:
  typedef Diag                                   diag_type;
  typedef typename diag_type::array_type         array_type;
  typedef typename array_type::value_type        value_type;
  typedef typename array_type::size_type         size_type;
  typedef typename ScalarTrait<value_type>::type scalar_type;

  JacobiBase() = delete;

  explicit JacobiBase(const diag_type &D, const size_type max_it = 0u,
                      const scalar_type tol = 0)
      : _D(D),
        _xk(),
        _r(),
        maxit(max_it ? max_it : 500u),
        rtol(tol > scalar_type(0) ? tol : scalar_type(1e-6)) {}

  template <class Operator>
  inline int solve(const Operator &A, const array_type &b, const size_type N,
                   array_type &x0) const {
    return _solve(A, b, N, x0, Dummy_streamer, Dummy_streamer);
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
  inline int _solve(const Operator &A, const array_type &b, const size_type N,
                    array_type &      x0, const CoutStream &,
                    const CerrStream &Cerr_) const {
    size_type iters(0);
    if (b.size() != x0.size() || _D.empty()) return 1;
    try {
      _xk.resize(b.size());
      _r.resize(b.size());
      _this()._prepare(A, b, x0);
    } catch (const std::exception &e) {
      Cerr_("resize buffer error with exception:\n%s\n", e.what());
      return 1;
    }
    // iteration
    // x=inv(D)*(b-A*xk)+xk
    const size_type n(b.size());
    auto &          x = x0;
    try {
      // // copy rhs to x as initial guess
      std::fill(x.begin(), x.end(), value_type(0));
      for (; iters < N; ++iters) {
        std::copy(x.cbegin(), x.cend(), _xk.begin());
        // compute residual, b-A*xk, store to x
        A.mv(_xk, x);
        for (size_type i(0); i < n; ++i) x[i] = b[i] - x[i];
        // compute inv(D)*res
        _D.solve(x, _r);
        _this()._post_process(iters, x);
      }
      return 0;
    } catch (const std::exception &e) {
      Cerr_("Exception thrown inside loop body (iter=%zd) with message:\n%s\n",
            iters, e.what());
      return 1;
    }
  }

 protected:
  const diag_type &  _D;
  mutable array_type _xk;
  mutable array_type _r;
  mutable array_type _v;

 public:
  size_type   maxit;
  scalar_type rtol;
};
}  // namespace internal

template <class Diag>
class Jacobi : public internal::JacobiBase<Jacobi<Diag>, Diag> {
  using _base = internal::JacobiBase<Jacobi<Diag>, Diag>;
  friend _base;

 public:
  typedef typename _base::diag_type   diag_type;
  typedef typename _base::array_type  array_type;
  typedef typename _base::size_type   size_type;
  typedef typename _base::scalar_type scalar_type;

  Jacobi() = delete;
  explicit Jacobi(const diag_type &D, const size_type max_it = 0u,
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
template <class Diag>
class CJacobi : public internal::JacobiBase<CJacobi<Diag>, Diag> {
  using _base = internal::JacobiBase<CJacobi<Diag>, Diag>;
  friend _base;

 public:
  typedef typename _base::diag_type   diag_type;
  typedef typename _base::array_type  array_type;
  typedef typename _base::size_type   size_type;
  typedef typename _base::value_type  value_type;
  typedef typename _base::scalar_type scalar_type;

  CJacobi() = delete;
  explicit CJacobi(const diag_type &D, const value_type lambda1 = 0.9,
                   const value_type lambda2 = -0.9, const size_type max_it = 0u,
                   const scalar_type tol = 0)
      : _base(D, max_it, tol), _rho(2), _xkk() {
    const value_type l2 =
        *std::min_element(D.prec(0).d_B.cbegin(), D.prec(0).d_B.cend());
    _compute_coeffs(lambda1, l2);
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

template <class Psmilu>
class PrecWrap {
 public:
  typedef Psmilu                                           diag_type;
  typedef typename diag_type::array_type                   array_type;
  typedef typename array_type::value_type                  value_type;
  typedef typename array_type::size_type                   size_type;
  typedef typename internal::ScalarTrait<value_type>::type scalar_type;

  PrecWrap() = delete;
  explicit PrecWrap(const diag_type &M) : _M(M) {}

  template <class Operator>
  inline int solve(const Operator &, const array_type &b, const size_type,
                   array_type &x0) const {
    _M.solve(b, x0);
    return 0;
  }

 protected:
  const diag_type &_M;
};

enum {
  GMRES_UNKNOWN_ERROR     = -3,
  GMRES_INVALID_FUNC_PARS = -2,
  GMRES_INVALID_PARS      = -1,
  GMRES_SUCCESS           = 0,
  GMRES_DIVERGED          = 1,
  GMRES_STAGNATED         = 2
};

template <class Psmilu>
class FGMRES {
 public:
  typedef Psmilu                                           prec_operator_type;
  typedef typename prec_operator_type::array_type          array_type;
  typedef typename array_type::size_type                   size_type;
  typedef typename array_type::value_type                  value_type;
  typedef typename internal::ScalarTrait<value_type>::type scalar_type;

 protected:
  constexpr static int _D = std::numeric_limits<scalar_type>::digits10 / 2 + 1;

 public:
  static_assert(std::is_floating_point<scalar_type>::value,
                "must be floating point type");

  scalar_type rtol    = sizeof(scalar_type) == sizeof(double) ? 1e-6 : 1e-4;
  int         restart = 30;
  size_type   maxit   = 500u;
  size_type   max_inner_steps = 8u;
  FGMRES()                    = delete;

  explicit FGMRES(const prec_operator_type &M) : _M(M) {}

  inline const array_type &resids() const { return _resids; }

  template <class Matrix>
  inline std::pair<int, size_type> solve_pre(const Matrix &    A,
                                             const array_type &b, array_type &x,
                                             const bool verbose = 1) const {
    std::fill(x.begin(), x.end(), value_type());
    const PrecWrap<prec_operator_type> M(_M);
    return verbose ? _solve(A, M, b, x, internal::Cout, internal::Cerr, true)
                   : _solve(A, M, b, x, internal::Dummy_streamer,
                            internal::Dummy_streamer, true);
  }

  template <class Matrix>
  inline std::pair<int, size_type> solve_jacobi(const Matrix &    A,
                                                const array_type &b,
                                                array_type &      x,
                                                const bool verbose = 1) const {
    std::fill(x.begin(), x.end(), value_type());
    const Jacobi<prec_operator_type> M(_M);
    return verbose ? _solve(A, M, b, x, internal::Cout, internal::Cerr, true)
                   : _solve(A, M, b, x, internal::Dummy_streamer,
                            internal::Dummy_streamer, true);
  }

  template <class Matrix>
  inline std::pair<int, size_type> solve_cjacobi(
      const Matrix &A, const array_type &b, array_type &x,
      const bool verbose = 1, const value_type lambda1 = 0.9,
      const value_type lambda2 = -0.9) const {
    std::fill(x.begin(), x.end(), value_type());
    const CJacobi<prec_operator_type> M(_M, lambda1, lambda2);
    return verbose ? _solve(A, M, b, x, internal::Cout, internal::Cerr, true)
                   : _solve(A, M, b, x, internal::Dummy_streamer,
                            internal::Dummy_streamer, true);
  }

 protected:
  const prec_operator_type &_M;
  mutable array_type        _y;
  mutable array_type        _R;
  mutable array_type        _Q;
  mutable array_type        _Z;
  mutable array_type        _J;
  mutable array_type        _v;
  mutable array_type        _resids;
  mutable array_type        _w;

 protected:
  inline void _check_pars() {
    if (rtol <= 0) rtol = sizeof(scalar_type) == sizeof(double) ? 1e-6 : 1e-4;
    if (restart <= 0) restart = 30;
    if (maxit <= 0) maxit = 500;
  }

  inline void _ensure_data_capacities(const size_type n) const {
    _y.resize(restart + 1);
    _R.resize(restart * (restart + 1) / 2);  // packed storage
    _Q.resize(n * restart);
    _Z.resize(_Q.size());
    _J.resize(2 * restart);
    _v.resize(n);
    _resids.reserve(maxit);
    _resids.resize(0);
    _w.resize(_v.size());
  }

  template <class Matrix, class Operator, class CoutStreamer,
            class CerrStreamer>
  std::pair<int, size_type> _solve(const Matrix &A, const Operator &M,
                                   const array_type &b, array_type &x0,
                                   const CoutStreamer &Cout,
                                   const CerrStreamer &Cerr,
                                   const bool          with_guess) const {
    using internal::conj;
    const static scalar_type _stag_eps =
        std::pow(scalar_type(10), -(scalar_type)_D);
    size_type iter(0);
    try {
      const size_type n = b.size();
      if (n != x0.size()) return std::make_pair(GMRES_INVALID_FUNC_PARS, iter);
      if (rtol <= 0 || restart <= 0 || maxit <= 0 || _M.empty())
        return std::make_pair(GMRES_INVALID_PARS, iter);
      const auto beta0 = std::sqrt(
          std::inner_product(b.cbegin(), b.cend(), b.cbegin(), value_type()));
      // record  time after preconditioner
      _ensure_data_capacities(n);
      const size_type max_outer_iters =
          (size_type)std::ceil((scalar_type)maxit / restart);
      auto &          x    = x0;
      int             flag = GMRES_SUCCESS;
      scalar_type     resid(1);
      const size_type half_restart = restart / 2;
      for (size_type it_outer = 0u; it_outer < max_outer_iters; ++it_outer) {
        Cout("Enter outer iteration %zd.\n", it_outer + 1);
        if (it_outer)
          Cerr("\033[1;33mWARNING!\033[0m Couldn\'t solve with %d restarts.\n",
               restart);
        // initial residual
        if (it_outer > 0u || with_guess) {
          A.mv(x, _v);
          for (size_type i = 0u; i < n; ++i) _v[i] = b[i] - _v[i];
        } else
          std::copy(b.cbegin(), b.cend(), _v.begin());
        const auto beta = std::sqrt(std::inner_product(
            _v.cbegin(), _v.cend(), _v.cbegin(), value_type()));
        _y[0]           = beta;
        do {
          const auto inv_beta = 1. / beta;
          for (size_type i = 0u; i < n; ++i) _Q[i] = _v[i] * inv_beta;
        } while (false);
        size_type       j(0);
        auto            R_itr     = _R.begin();
        const size_type exp_steps = std::min(it_outer + 1, max_inner_steps);
        for (;;) {
          const auto jn = j * n;
          std::copy(_Q.cbegin() + jn, _Q.cbegin() + jn + n, _v.begin());
          if (M.solve(A, _v, exp_steps, _w)) {
            Cerr(
                "\033[1;33mWARNING!\033[0m Failed to call M operator at "
                "iteration "
                "%zd.\n",
                iter);
            flag = GMRES_UNKNOWN_ERROR;
            break;
          }
          std::copy(_w.cbegin(), _w.cend(), _Z.begin() + jn);
          A.mv(_w, _v);
          for (size_type k = 0u; k <= j; ++k) {
            auto itr = _Q.cbegin() + k * n;
            _w[k] =
                std::inner_product(_v.cbegin(), _v.cend(), itr, value_type());
            const auto tmp = _w[k];
            for (size_type i = 0u; i < n; ++i) _v[i] -= tmp * itr[i];
          }
          const auto v_norm2 = std::inner_product(_v.cbegin(), _v.cend(),
                                                  _v.cbegin(), value_type());
          const auto v_norm  = std::sqrt(v_norm2);
          if (j + 1 < (size_type)restart) {
            auto       itr      = _Q.begin() + jn + n;
            const auto inv_norm = 1. / v_norm;
            for (size_type i = 0u; i < n; ++i) itr[i] = inv_norm * _v[i];
          }
          auto J1 = _J.begin(), J2 = J1 + restart;
          for (size_type colJ = 0u; colJ + 1u <= j; ++colJ) {
            const auto tmp = _w[colJ];
            _w[colJ]     = conj(J1[colJ]) * tmp + conj(J2[colJ]) * _w[colJ + 1];
            _w[colJ + 1] = -J2[colJ] * tmp + J1[colJ] * _w[colJ + 1];
          }
          const auto rho        = std::sqrt(_w[j] * _w[j] + v_norm2);
          J1[j]                 = _w[j] / rho;
          J2[j]                 = v_norm / rho;
          _y[j + 1]             = -J2[j] * _y[j];
          _y[j]                 = conj(J1[j]) * _y[j];
          _w[j]                 = rho;
          R_itr                 = std::copy_n(_w.cbegin(), j + 1, R_itr);
          const auto resid_prev = resid;
          resid                 = std::abs(_y[j + 1]) / beta0;
          if (resid >= resid_prev * (1 - _stag_eps)) {
            Cerr(
                "\033[1;33mWARNING!\033[0m Stagnated detected at iteration "
                "%zd.\n",
                iter);
            flag = GMRES_STAGNATED;
            break;
          } else if (iter >= maxit) {
            Cerr(
                "\033[1;33mWARNING!\033[0m Reached maxit iteration limit "
                "%zd.\n",
                maxit);
            flag = GMRES_DIVERGED;
            break;
          }
          ++iter;
          Cout("  At iteration %zd, relative residual is %g.\n", iter, resid);
          _resids.push_back(resid);
          if (resid < rtol || j + 1 >= (size_type)restart) break;
          ++j;
        }  // inf loop
        // backsolve
        if (j) {
          for (int k = j; k > -1; --k) {
            --R_itr;
            _y[k] /= *R_itr;
            const auto tmp = _y[k];
            for (int i = k - 1; i > -1; --i) {
              --R_itr;
              _y[i] -= tmp * *R_itr;
            }
          }
        }
        for (size_type i = 0u; i <= j; ++i) {
          const auto tmp   = _y[i];
          auto       Z_itr = _Z.cbegin() + i * n;
          for (size_type k = 0u; k < n; ++k) x[k] += tmp * Z_itr[k];
        }
        if (resid < rtol || flag != GMRES_SUCCESS) break;
      }
      return std::make_pair(flag, iter);
    } catch (const std::exception &e) {
      Cerr(
          "\033[1;33mWARNING!\033[0m Unexpected termination due to exception "
          "threw at iteration %zd.\n\n\tThe message is:\n\n%s\n",
          iter, e.what());
      return std::make_pair(GMRES_UNKNOWN_ERROR, iter);
    }
  }
};
}  // namespace psmilu