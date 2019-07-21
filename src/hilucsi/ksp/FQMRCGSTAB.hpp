//@HEADER
//----------------------------------------------------------------------------
//                Copyright (C) 2019 The HILUCSI AUTHORS
//----------------------------------------------------------------------------
//@HEADER

/// \file hilucsi/ksp/FQMRCGSTAB.hpp
/// \brief QMRCGSTAB implementation
/// \authors Qiao,

#ifndef _HILUCSI_KSP_QMRCGSTAB_HPP
#define _HILUCSI_KSP_QMRCGSTAB_HPP

#include <cmath>
#include <limits>
#include <utility>

#include "hilucsi/ksp/common.hpp"
#include "hilucsi/utils/math.hpp"

namespace hilucsi {
namespace ksp {

/// \class FQMRCGSTAB
/// \tparam MType preconditioner type, see \ref HILUCSI
/// \brief QMRCGSTAB implementation
/// \ingroup qmrcgstab
template <class MType>
class FQMRCGSTAB : public internal::KSP<FQMRCGSTAB<MType>, MType> {
 protected:
  using _base = internal::KSP<FQMRCGSTAB<MType>, MType>;  ///< base
  // grant friendship
  friend _base;

 public:
  typedef MType                           M_type;      ///< preconditioner
  typedef typename M_type::array_type     array_type;  ///< value array
  typedef typename array_type::size_type  size_type;   ///< size type
  typedef typename array_type::value_type value_type;  ///< value type
  typedef typename DefaultSettings<value_type>::scalar_type scalar_type;
  ///< scalar type from value_type

  static_assert(std::is_floating_point<scalar_type>::value,
                "must be floating point type");

  /// \brief get the solver name
  inline static const char *repr() { return "FQMRCGSTAB"; }

  using _base::rtol;

  using _base::maxit;

  using _base::inner_steps;
  using _base::lamb1;
  using _base::lamb2;

  FQMRCGSTAB() = default;

  /// \brief constructor with all essential parameters
  /// \param[in] M multilevel ILU preconditioner
  /// \param[in] rel_tol relative tolerance for convergence (1e-6 for double)
  /// \param[in] max_iters maximum number of iterations
  /// \param[in] inner_steps maximum inner iterations for jacobi kernels
  explicit FQMRCGSTAB(
      std::shared_ptr<M_type> M,
      const scalar_type       rel_tol = DefaultSettings<value_type>::rtol,
      const size_type max_iters       = DefaultSettings<value_type>::max_iters,
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
    _base::_init_resids();
  }

  /// \brief low level solve kernel
  /// \tparam Matrix user input matrix type, see \ref CRS and \ref CCS
  /// \tparam Operator "preconditioner" operator type, see
  ///         \ref internal::DummyJacobi,
  ///         \ref internal::Jacobi, and
  ///         \ref internal::ChebyshevJacobi
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
  template <class Matrix, class Operator, class StreamerCout,
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
      A.mv(x, _Ax);
      for (size_type i(0); i < n; ++i) _r0[i] = b[i] - _Ax[i];
    }
    const auto &r0  = _r0;
    auto        tau = norm2(r0);
    _resids[0]      = tau / normb;  // starting with size 1
    if (_resids[0] <= rtol) return std::make_pair(flag, size_type(0));
    std::copy(r0.cbegin(), r0.cend(), _p.begin());
    if (M.solve(A, _p, innersteps, _ph)) {
      Cerr(__HILUCSI_FILE__, __HILUCSI_FUNC__, __LINE__,
           "Failed to call M operator at iteration %d.", 1);
      flag = M_SOLVE_ERROR;
      return std::make_pair(flag, size_type(1));
    }
    A.mv(_ph, _v);
    std::copy(r0.cbegin(), r0.cend(), _r.begin());
    // comment out, implicitly handled in the loop
    // std::fill_n(_d.begin(), n, value_type(0));
    value_type eta(0), theta(0), rho1 = tau * tau;

    size_type iter(1);

    // main loop
    for (; iter <= maxit; ++iter) {
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

      if (M.solve(A, _s, innersteps, _sh)) {
        Cerr(__HILUCSI_FILE__, __HILUCSI_FUNC__, __LINE__,
             "Failed to call M operator at iteration %zd.", iter);
        flag = M_SOLVE_ERROR;
        break;
      }
      A.mv(_sh, _t);

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
      A.mv(x, _Ax);
      for (size_type i(0); i < n; ++i) _Ax[i] = b[i] - _Ax[i];
      const auto resid_prev = _resids.back();
      _resids.push_back(norm2(_Ax) / normb);
      Cout("  At iteration %zd (inner:%zd), relative residual is %g.", iter,
           innersteps, _resids.back());
      if (_resids.back() <= rtol) break;

      if (std::isnan(_resids.back()) || std::isinf(_resids.back())) {
        Cerr(__HILUCSI_FILE__, __HILUCSI_FUNC__, __LINE__,
             "Solver break-down detected at iteration %zd.", iter);
        flag = BREAK_DOWN;
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

      if (M.solve(A, _p, innersteps, _ph)) {
        Cerr(__HILUCSI_FILE__, __HILUCSI_FUNC__, __LINE__,
             "Failed to call M operator at iteration %zd.", iter);
        flag = M_SOLVE_ERROR;
        break;
      }
      A.mv(_ph, _v);
      rho1 = rho2;
    }  // for

    if (flag == SUCCESS && _resids.back() > rtol) {
      Cerr(__HILUCSI_FILE__, __HILUCSI_FUNC__, __LINE__,
           "Reached maxit iteration limit %zd.", maxit);
      flag = DIVERGED;
      iter = maxit;
    }

    return std::make_pair(flag, iter);
  }

 private:
  using _base::restart;
};

}  // namespace ksp
}  // namespace hilucsi

#endif  // _HILUCSI_KSP_QMRCGSTAB_HPP