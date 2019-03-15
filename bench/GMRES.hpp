//@HEADER
//----------------------------------------------------------------------------
//                Copyright (C) 2019 The PSMILU AUTHORS
//----------------------------------------------------------------------------
//@HEADER

// This is a template implementation of GMRES with right preconditioner
// for benchmark purpose.

#pragma once

#include <algorithm>
#include <chrono>
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
namespace bench {
namespace internal {

class IdentityPrec {
 public:
  template <class Operator>
  inline void compute(const Operator &, const void * = nullptr) {}

  template <class Vector>
  inline void solve(const Vector &b, Vector &x) const {
    std::copy(b.cbegin(), b.cend(), x.begin());
  }

  inline constexpr bool empty() const { return false; }
};

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

}  // namespace internal

using internal::IdentityPrec;

enum {
  GMRES_UNKNOWN_ERROR     = -3,
  GMRES_INVALID_FUNC_PARS = -2,
  GMRES_INVALID_PARS      = -1,
  GMRES_SUCCESS           = 0,
  GMRES_DIVERGED          = 1,
  GMRES_STAGNATED         = 2
};

inline const char *gmres_repr_flag(const int flag) {
  switch (flag) {
    case GMRES_UNKNOWN_ERROR:
      return "GMRES_UNKNOWN_ERROR";
    case GMRES_INVALID_FUNC_PARS:
      return "GMRES_INVALID_FUNC_PARS";
    case GMRES_INVALID_PARS:
      return "GMRES_INVALID_PARS";
    case GMRES_SUCCESS:
      return "GMRES_SUCCESS";
    case GMRES_DIVERGED:
      return "GMRES_DIVERGED";
    case GMRES_STAGNATED:
      return "GMRES_STAGNATED";
    default:
      return "GMRES_INVALID_FLAG";
  }
}

/// \class GMRES
/// \tparam ValueType value type used, default is \a double
/// \tparam PrecType \b right preconditioner used, default is \ref IdentityPrec
/// \tparam ArrayType internal array type used, e.g. \a std::vector (default)
/// \authors Qiao,
template <class ValueType = double, class PrecType = internal::IdentityPrec,
          class ArrayType = std::vector<ValueType>>
class GMRES {
 public:
  using value_type  = ValueType;    ///< value type
  using size_type   = std::size_t;  ///< size type
  using array_type  = ArrayType;    ///< array used
  using prec_type   = PrecType;     ///< preconditioner used
  using scalar_type = typename internal::ScalarTrait<value_type>::type;
  ///< scalar type, implicitly assume \a std::complex

 protected:
  constexpr static int _D = std::numeric_limits<scalar_type>::digits10 / 2 + 1;

 public:
  static_assert(std::is_floating_point<scalar_type>::value,
                "must be floating point type");

  scalar_type rtol = sizeof(scalar_type) == sizeof(double) ? 1e-6 : 1e-4;
  ///< relative tolerance, default is 1e6 for \a double, 1e-4 for \a float
  int               restart = 30;    ///< restart, deafult is 30
  size_type         maxit   = 500u;  ///< max iteration, default is 500
  mutable prec_type M;               ///< preconditioner

  /// \brief default constructor
  GMRES() = default;

  GMRES(const scalar_type rel_tol, int restart_ = 0, int max_iters = 0)
      : rtol(rel_tol), restart(restart_), maxit(max_iters) {
    _check_pars();
  }

  inline void set_parameter(const std::string &par_name, const scalar_type v) {
    if (par_name == "rtol")
      rtol = v;
    else if (par_name == "restart")
      restart = (int)v;
    else if (par_name == "maxit")
      maxit = (int)v;
  }

  inline const array_type &resids() const { return _resids; }

  template <class Operator, class VectorType>
  std::tuple<int, size_type, double> solve_with_guess(
      const Operator &A, const VectorType &b, VectorType &x0,
      const bool verbose = true) const {
    return verbose ? _solve_kernel(A, b, x0, internal::Cout, internal::Cerr)
                   : _solve_kernel(A, b, x0, internal::Dummy_streamer,
                                   internal::Dummy_streamer);
  }

  template <class Operator, class VectorType>
  std::tuple<int, size_type, double> solve(const Operator &  A,
                                           const VectorType &b, VectorType &x,
                                           const bool verbose = true) const {
    std::fill(x.begin(), x.end(), value_type());
    return solve_with_guess(A, b, x, verbose);
  }

 protected:
  mutable array_type _y;
  mutable array_type _R;
  mutable array_type _Q;
  mutable array_type _Z;
  mutable array_type _J;
  mutable array_type _v;
  mutable array_type _resids;
  mutable array_type _w;

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

  template <class Operator, class VectorType, class CoutStreamer,
            class CerrStreamer>
  std::tuple<int, size_type, double> _solve_kernel(
      const Operator &A, const VectorType &b, VectorType &x0,
      const CoutStreamer &Cout, const CerrStreamer &Cerr) const {
    using internal::conj;
    const static scalar_type _stag_eps =
        std::pow(scalar_type(10), -(scalar_type)_D);
    size_type iter(0);
    try {
      const size_type n = b.size();
      if (n != x0.size())
        return std::make_tuple(GMRES_INVALID_FUNC_PARS, iter, 0.0);
      if (rtol <= 0 || restart <= 0 || maxit <= 0)
        return std::make_tuple(GMRES_INVALID_PARS, iter, 0.0);
      const auto beta0 = std::sqrt(
          std::inner_product(b.cbegin(), b.cend(), b.cbegin(), value_type()));
      if (M.empty()) M.compute(A);  // default compute
      // record  time after preconditioner
      _ensure_data_capacities(n);
      auto            time_start = std::chrono::high_resolution_clock::now();
      const size_type max_outer_iters =
          (size_type)std::ceil((scalar_type)maxit / restart);
      auto &      x    = x0;
      int         flag = GMRES_SUCCESS;
      scalar_type resid(1);
      for (size_type it_outer = 0u; it_outer < max_outer_iters; ++it_outer) {
        Cout("Enter outer iteration %zd.\n", it_outer + 1);
        if (it_outer)
          Cerr("\033[1;33mWARNING!\033[0m Couldn\'t solve with %d restarts.\n",
               restart);
        // initial residual
        if (it_outer > 0u || std::inner_product(x.cbegin(), x.cend(),
                                                x.cbegin(), value_type()) > 0) {
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
        size_type j(0);
        auto      R_itr = _R.begin();
        for (;;) {
          const auto jn = j * n;
          std::copy(_Q.cbegin() + jn, _Q.cbegin() + jn + n, _v.begin());
          M.solve(_v, _w);
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
            flag = GMRES_STAGNATED;
            Cerr(
                "\033[1;33mWARNING!\033[0m Stagnated detected at iteration "
                "%zd.\n",
                iter);
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
      auto time_end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> period = time_end - time_start;
      return std::make_tuple(flag, iter, period.count());
    } catch (const std::exception &e) {
      Cerr(
          "\033[1;33mWARNING!\033[0m Unexpected termination due to exception "
          "threw at iteration %zd.\n\n\tThe message is:\n\n%s\n",
          iter, e.what());
      return std::make_tuple(GMRES_UNKNOWN_ERROR, iter, 0.0);
    }
  }
};

}  // namespace bench
}  // namespace psmilu
