///////////////////////////////////////////////////////////////////////////////
//                This file is part of HILUCSI project                       //
///////////////////////////////////////////////////////////////////////////////

/*!
 * \file hilucsi/ksp/QMRCGSTAB_Null.hpp
 * \brief QMRCGSTAB implementation for solving left null space component(s)
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

#ifndef _HILUCSI_KSP_QMRCGSTAB_NULL_HPP
#define _HILUCSI_KSP_QMRCGSTAB_NULL_HPP

#include <cmath>
#include <functional>
#include <limits>
#include <utility>

#include "hilucsi/ksp/common.hpp"
#include "hilucsi/utils/math.hpp"

namespace hilucsi {
namespace ksp {

/// \class QMRCGSTAB_Null
/// \tparam MType preconditioner type, see \ref HILUCSI
/// \tparam ValueType if not given, i.e. \a void, then use value in \a MType
/// \brief Flexible QMRCGSTAB implementation
/// \ingroup qmrcgstab
template <class MType, class ValueType = void>
class QMRCGSTAB_Null
    : public internal::KSP<QMRCGSTAB_Null<MType, ValueType>, MType, ValueType> {
 protected:
  using _base =
      internal::KSP<QMRCGSTAB_Null<MType, ValueType>, MType, ValueType>;
  ///< base
  // grant friendship
  friend _base;

 public:
  typedef MType                           M_type;      ///< preconditioner
  typedef typename _base::array_type      array_type;  ///< value array
  typedef typename array_type::size_type  size_type;   ///< size type
  typedef typename array_type::value_type value_type;  ///< value type
  typedef typename DefaultSettings<value_type>::scalar_type scalar_type;
  ///< scalar type from value_type

  static_assert(std::is_floating_point<scalar_type>::value,
                "must be floating point type");

  /// \brief get the solver name
  inline static const char *repr() { return "QMRCGSTAB_Null"; }

  using _base::rtol;

  using _base::maxit;

  using _base::inner_steps;
  using _base::lamb1;
  using _base::lamb2;

  QMRCGSTAB_Null() = default;

  /// \brief constructor with all essential parameters
  /// \param[in] M multilevel ILU preconditioner
  /// \param[in] rel_tol relative tolerance for convergence (1e-6 for double)
  /// \param[in] max_iters maximum number of iterations
  /// \param[in] inner_steps maximum inner iterations for jacobi kernels
  explicit QMRCGSTAB_Null(
      std::shared_ptr<M_type> M, const scalar_type rel_tol = 1e-13,
      const size_type max_iters   = DefaultSettings<value_type>::max_iters,
      const size_type inner_steps = DefaultSettings<value_type>::inner_steps)
      : _base(M, rel_tol, max_iters, inner_steps) {
    if (_M && _M->nrows()) _ensure_data_capacities(_M->nrows());
  }

 protected:
  /// \name workspace
  /// Workspace for QMRCGSTAB
  /// @{
  mutable array_type _Ax;
  mutable array_type _r0;
  mutable array_type _p;
  mutable array_type _ph;
  mutable array_type _v;
  mutable array_type _r;
  mutable array_type _s;
  mutable array_type _t;
  mutable array_type _d;
  mutable array_type _d2;
  mutable array_type _x2;
  mutable array_type _sh;
  mutable array_type _xk;  ///< previous solution
  /// @}
  using _base::_M;
  using _base::_resids;

 protected:
  /// \brief ensure internal buffer sizes
  /// \param[in] n right-hand size array size
  inline void _ensure_data_capacities(const size_type n) const {
    _Ax.resize(n);
    _r0.resize(n);
    _p.resize(n);
    _ph.resize(n);
    _v.resize(n);
    _r.resize(n);
    _s.resize(n);
    _t.resize(n);
    _d.resize(n);
    _d2.resize(n);
    _x2.resize(n);
    _sh.resize(n);
    // _xk.resize(n);
    _base::_init_resids();
  }

  /// \brief helper function for applying matrix vector transpose
  template <class Matrix>
  static void _apply_mv_t(const Matrix &A, const array_type &x, array_type &y) {
    A.mv_t(x, y);
  }

  static void _apply_mv_t(
      const std::function<void(const array_type &, array_type &)> &A,
      const array_type &x, array_type &y) {
    A(x, y);
  }

  /// \brief low level solve kernel
  /// \tparam UseIR flag indicates whether or not enabling iterative refine
  /// \tparam Matrix user input matrix type, see \ref CRS and \ref CCS
  /// \tparam Operator "preconditioner" operator type, see \ref HILUCSI
  /// \tparam StreamerCout cout streamer type
  /// \tparam StreamerCerr cerr streamer type
  /// \param[in] A user matrix
  /// \param[in] M "preconditioner" operator
  /// \param[in] b right hard size
  /// \param[in] innersteps inner steps used
  /// \param[in] zero_start flag to indicate \a x0 starts with all zeros
  /// \param[in,out] x0 initial guess and solution on output
  /// \param[in] Cout "stdout" streamer
  /// \param[in] Cerr "stderr" streamer
  template <bool UseIR, class Matrix, class Operator, class StreamerCout,
            class StreamerCerr>
  std::pair<int, size_type> _solve(const Matrix &A, const Operator &M,
                                   const array_type &b,
                                   const size_type   innersteps,
                                   const bool zero_start, array_type &x0,
                                   const StreamerCout &Cout,
                                   const StreamerCerr &Cerr) const {
    const size_type n = b.size();
    // record  time after preconditioner
    _ensure_data_capacities(n);
    const auto normb = norm2(b);
    auto &     x     = x0;
    int        flag  = SUCCESS;
    if (normb == 0) {
      std::fill_n(x.begin(), n, value_type(0));
      return std::make_pair(flag, size_type(0));
    }
    if (zero_start)
      std::copy(b.cbegin(), b.cend(), _r0.begin());
    else {
      // mt::mv_nt(A, x, _Ax);
      QMRCGSTAB_Null::_apply_mv_t(A, x, _Ax);
      // A.mv(x, _Ax);
      for (size_type i(0); i < n; ++i) _r0[i] = b[i] - _Ax[i];
    }
    const auto &r0  = _r0;
    auto        tau = norm2(r0);
    _resids[0]      = tau;  // starting with size 1
    if (_resids[0] <= rtol * normb) return std::make_pair(flag, size_type(0));
    std::copy(r0.cbegin(), r0.cend(), _p.begin());
    std::copy(r0.cbegin(), r0.cend(), _r.begin());
    // comment out, implicitly handled in the loop
    // std::fill_n(_d.begin(), n, value_type(0));
    value_type eta(0), theta(0), rho1 = tau * tau;

    size_type iter(1);

    // // initialize solution set to be b
    // std::copy_n(b.cbegin(), n, _xk.begin());

    // main loop
    for (; iter <= maxit; ++iter) {
      // if (M.solve(A, _p, innersteps, _ph)) {
      //   Cerr(__HILUCSI_FILE__, __HILUCSI_FUNC__, __LINE__,
      //        "Failed to call M operator at iteration %zd.", iter);
      //   flag = M_SOLVE_ERROR;
      //   break;
      // }
      UseIR ? M.solve(A, _p, innersteps, _ph) : M.solve(_p, _ph);
      QMRCGSTAB_Null::_apply_mv_t(A, _ph, _v);
      // mt::mv_nt(A, _ph, _v);
      // A.mv(_ph, _v);
      auto rho2 = inner(r0, _v);
      if (rho2 == 0) {
        Cerr(__HILUCSI_FILE__, __HILUCSI_FUNC__, __LINE__,
             "Solver break-down detected at iteration %zd.", iter);
        flag = BREAK_DOWN;
        break;
      }
      if (rho1 == 0) {
        Cerr(__HILUCSI_FILE__, __HILUCSI_FUNC__, __LINE__,
             "Stagnated detected at iteration %zd.", iter);
        flag = STAGNATED;
        break;
      }
      const auto alpha = rho1 / rho2;
      for (size_type i(0); i < n; ++i) _s[i] = _r[i] - alpha * _v[i];
      const auto theta2 = norm2(_s) / tau;
      auto       c      = 1. / std::sqrt(1.0 + theta2 * theta2);
      const auto tau2   = tau * theta2 * c;
      const auto eta2   = c * c * alpha;
      if (iter == 1u)
        std::copy(_ph.cbegin(), _ph.cend(), _d2.begin());
      else {
        const auto coeff = theta * theta * eta / alpha;
        for (size_type i(0); i < n; ++i) _d2[i] = _ph[i] + coeff * _d[i];
      }
      for (size_type i(0); i < n; ++i) _x2[i] = x[i] + eta2 * _d2[i];

      // if (M.solve(A, _s, innersteps, _sh)) {
      //   Cerr(__HILUCSI_FILE__, __HILUCSI_FUNC__, __LINE__,
      //        "Failed to call M operator at iteration %zd.", iter);
      //   flag = M_SOLVE_ERROR;
      //   break;
      // }
      UseIR ? M.solve(A, _s, innersteps, _sh) : M.solve(_s, _sh);
      QMRCGSTAB_Null::_apply_mv_t(A, _sh, _t);
      // mt::mv_nt(A, _sh, _t);
      // A.mv(_sh, _t);

      const auto uu = inner(_s, _t), vv = norm2_sq(_t);
      const auto omega = uu / vv;
      if (omega == 0) {
        Cerr(__HILUCSI_FILE__, __HILUCSI_FUNC__, __LINE__,
             "Stagnated detected at iteration %zd.", iter);
        flag = STAGNATED;
        break;
      }
      for (size_type i(0); i < n; ++i) _r[i] = _s[i] - omega * _t[i];

      theta             = norm2(_r) / tau2;
      c                 = 1. / std::sqrt(1. + theta * theta);
      tau               = tau2 * theta * c;
      eta               = c * c * omega;
      const auto _coeff = theta2 * theta2 * eta2 / omega;
      for (size_type i(0); i < n; ++i) {
        const auto tmp = _sh[i] + _coeff * _d2[i];
        _d[i]          = tmp;
        x[i]           = _x2[i] + eta * tmp;
      }

      // update residual
      QMRCGSTAB_Null::_apply_mv_t(A, x, _Ax);
      // mt::mv_nt(A, x, _Ax);
      // A.mv(x, _Ax);
      // for (size_type i(0); i < n; ++i) _Ax[i] = b[i] - _Ax[i];
      // const auto resid_prev = _resids.back();
      _resids.push_back(norm2(_Ax));
      const auto resid = _resids.back() / norm2(x);
      Cout("  At iteration %zd (#Ax:%zd), relative residual is %g.", iter,
           innersteps * 2,
           resid);  // *2 due to called A*x twice
      if (resid <= rtol) break;

      if (std::isnan(resid) || std::isinf(resid)) {
        Cerr(__HILUCSI_FILE__, __HILUCSI_FUNC__, __LINE__,
             "Solver break-down detected at iteration %zd.", iter);
        flag = BREAK_DOWN;
        break;
      }
      if (resid > 100) {
        Cerr(__HILUCSI_FILE__, __HILUCSI_FUNC__, __LINE__,
             "Divergence encountered at iteration %zd.", iter);
        flag = DIVERGED;
        break;
      }
      rho2 = inner(_r, r0);
      if (rho2 == 0) {
        Cerr(__HILUCSI_FILE__, __HILUCSI_FUNC__, __LINE__,
             "Stagnated detected at iteration %zd.", iter);
        flag = STAGNATED;
        break;
      }
      const auto beta = alpha * rho2 / (omega * rho1);
      for (size_type i(0); i < n; ++i)
        _p[i] = _r[i] + beta * (_p[i] - omega * _v[i]);
      rho1 = rho2;
    }  // for

    if (flag == DIVERGED) {
      Cerr(__HILUCSI_FILE__, __HILUCSI_FUNC__, __LINE__,
           "Reached maxit iteration limit %zd.", maxit);
      flag = DIVERGED;
      iter = maxit;
    }

    return std::make_pair(flag, iter);
  }

 protected:
  using _base::restart;
};

}  // namespace ksp
}  // namespace hilucsi

#endif  // _HILUCSI_KSP_QMRCGSTAB_NULL_HPP