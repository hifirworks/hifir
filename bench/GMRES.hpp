//@HEADER
//----------------------------------------------------------------------------
//                Copyright (C) 2019 The PSMILU AUTHORS
//----------------------------------------------------------------------------
//@HEADER

// This is a template implementation of GMRES with right preconditioner
// for benchmark purpose.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <numeric>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace psmilu {
namespace bench {
namespace internal {

template <class ValueType>
class IdentityPrec {
 public:
  using value_type = ValueType;

  template <class Operator>
  inline void compute(const Operator &, const void * = nullptr) {}

  inline constexpr bool empty() const { return false; }
};
}  // namespace internal

/// \class GMRES
/// \tparam ValueType value type used, default is \a double
/// \tparam PrecType \b right preconditioner used, default is \ref IdentityPrec
/// \tparam ArrayType internal array type used, e.g. \a std::vector (default)
/// \authors Qiao,
template <class ValueType = double,
          class PrecType  = internal::IdentityPrec<ValueType>,
          class ArrayType = std::vector<ValueType>>
class GMRES {
 public:
  using value_type = ValueType;    ///< value type
  using size_type  = std::size_t;  ///< size type
  using array_type = ArrayType;    ///< array used
  using prec_type  = PrecType;     ///< preconditioner used
  using scalar_type =
      typename std::conditional<std::is_class<value_type>::value,
                                typename value_type::value_type,
                                value_type>::type;
  ///< scalar type, implicitly assume \a std::complex

  static_assert(std::is_floating_point<scalar_type>::value,
                "must be floating point type");

  scalar_type rtol = sizeof(scalar_type) == sizeof(double) ? 1e-6 : 1e-4;
  ///< relative tolerance, default is 1e6 for \a double, 1e-4 for \a float
  int       restart = 30;    ///< restart, deafult is 30
  size_type maxit   = 500u;  ///< max iteration, default is 500

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

  template <class Operator, class VectorType>
  std::pair<int, size_type> solve_with_guess(const Operator &  A,
                                             const VectorType &b,
                                             VectorType &      x0,
                                             const bool = true) const {
    const size_type n = b.size();
    size_type       iter(0);
    if (n != x0.size()) return std::make_pair(-1, iter);
    if (rtol <= 0 || restart <= 0 || maxit <= 0)
      return std::make_pair(-2, iter);
    if (_M.empty()) _M.compute(A);
    _ensure_data_capacities(n);
    const size_type max_outer_iters =
        (size_type)std::ceil((scalar_type)maxit / restart);
    auto &x    = x0;
    int   flag = 0;
    for (size_type it_outer = 0u; it_outer < max_outer_iters; ++it_outer) {
      // initial residual
      if (it_outer > 1u || std::inner_product(x.cbegin(), x.cend(), x.cbegin(),
                                              value_type()) > 0) {
        A.mv(x, _v);
        for (size_type i = 0u; i < n; ++i) _v[i] = b[i] - _v[i];
      } else
        std::copy(b.cbegin(), b.cend(), _v.begin());
      const auto beta = std::sqrt(std::inner_product(
          _v.cbegin(), _v.cend(), _v.cbegin(), value_type()));
      _y[0]           = beta;
      {
        const auto inv_beta = 1. / beta;
        for (size_type i = 0u; i < n; ++i) _Q[i] = _v[i] * inv_beta;
      }
      size_type j(0);
      for (;;) {
        const auto jn = j * n;
        std::copy(_Q.cbegin() + jn, _Q.cbegin() + jn + n, _w.begin());
        _M.solve(_w, _v);
        std::copy(_v.cbegin(), _v.cend(), _Z.begin() + jn);
        A.mv(_v, _w);
        {
          auto itr = _R.begin() + restart * j;
          for (size_type k = 0u; k <= j; ++k)
            itr[k] = std::inner_product(_w.cbegin(), _w.cend(),
                                        _Q.cbegin() + k * n, value_type());
        }
      }
    }
    return std::make_pair(flag, iter);
  }

  template <class Operator, class VectorType>
  std::pair<int, size_type> solve(const Operator &A, const VectorType &b,
                                  VectorType &x,
                                  const bool  verbose = true) const {
    std::fill(x.begin(), x.end(), value_type());
    return solve_with_guess(A, b, x, verbose);
  }

 protected:
  mutable prec_type  _M;  ///< preconditioner
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

  inline void _ensure_data_capacities(const size_type n) {
    _y.resize(n);
    _R.resize(restart * restart);
    _Q.resize(n * restart);
    _Z.resize(_Q.size());
    _J.resize(2 * restart);
    _v.resize(_y.size());
    _resids.resize(maxit);
    _w.resize(_v.size());
  }
};

}  // namespace bench
}  // namespace psmilu
