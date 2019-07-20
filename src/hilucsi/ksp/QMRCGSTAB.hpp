//@HEADER
//----------------------------------------------------------------------------
//                Copyright (C) 2019 The HILUCSI AUTHORS
//----------------------------------------------------------------------------
//@HEADER

/// \file hilucsi/ksp/QMRCGSTAB.hpp
/// \brief QMRCGSTAB implementation
/// \authors Qiao,

#ifndef _HILUCSI_KSP_QMRCGSTAB_HPP
#define _HILUCSI_KSP_QMRCGSTAB_HPP

#include <cmath>
#include <limits>
#include <memory>
#include <utility>

#include "hilucsi/ksp/common.hpp"
#include "hilucsi/utils/log.hpp"
#include "hilucsi/utils/math.hpp"

namespace hilucsi {
namespace ksp {

/// \class FGMRES
/// \tparam MType preconditioner type, see \ref HILUCSI
/// \brief QMRCGSTAB implementation
/// \ingroup qmrcgstab
template <class MType>
class QMRCGSTAB {
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
  inline static const char *repr() { return "QMRCGSTAB"; }

  scalar_type rtol = DefaultSettings<value_type>::rtol;
  ///< relative convergence tolerance
  size_type maxit = DefaultSettings<value_type>::max_iters;
  ///< max numer of iterations

  QMRCGSTAB() = default;

  /// \brief constructor with all essential parameters
  /// \param[in] M multilevel ILU preconditioner
  /// \param[in] rel_tol relative tolerance for convergence (1e-6 for double)
  /// \param[in] max_iters maximum number of iterations
  explicit QMRCGSTAB(
      std::shared_ptr<M_type> M,
      const scalar_type       rel_tol = DefaultSettings<value_type>::rtol,

      const size_type max_iters = DefaultSettings<value_type>::max_iters)
      : _M(M), rtol(rel_tol), maxit(max_iters) {
    _check_pars();
    if (_M && _M->nrows()) _ensure_data_capacities(_M->nrows());
  }

  /// \brief set preconditioner
  /// \param[in] M multilevel ILU preconditioner
  inline void set_M(std::shared_ptr<M_type> M) {
    _M = M;  // increment internal reference counter
    if (_M && _M->nrows()) _ensure_data_capacities(_M->nrows());
  }

  /// \brief get preconditioner
  inline std::shared_ptr<M_type> get_M() const { return _M; }

  /// \brief get residual array
  inline const array_type &resids() const { return _resids; }

  /// \brief solve with \ref _M as traditional preconditioner
  /// \tparam Matrix user input type, see \ref CRS and \ref CCS
  /// \param[in] A user input matrix
  /// \param[in] b right-hand side vector
  /// \param[in,out] x solution
  /// \param[in] with_init_guess if \a false (default), then assign zero to
  ///             \a x as starting values
  /// \param[in] verbose if \a true (default), enable verbose printing
  template <class Matrix>
  inline std::pair<int, size_type> solve(const Matrix &A, const array_type &b,
                                         array_type &x,
                                         const bool  with_init_guess = false,
                                         const bool  verbose = true) const {
    const static hilucsi::internal::StdoutStruct       Cout;
    const static hilucsi::internal::StderrStruct       Cerr;
    const static hilucsi::internal::DummyStreamer      Dummy_streamer;
    const static hilucsi::internal::DummyErrorStreamer Dummy_cerr;

    if (_validate(A, b, x)) return std::make_pair(INVALID_ARGS, size_type(0));
    if (verbose) _show(with_init_guess);
    if (!with_init_guess) std::fill(x.begin(), x.end(), value_type(0));
    return verbose
               ? _solve(A, b, !with_init_guess, x, Cout, Cerr)
               : _solve(A, b, !with_init_guess, x, Dummy_streamer, Dummy_cerr);
  }

 protected:
  std::shared_ptr<M_type> _M;  ///< preconditioner operator
  mutable array_type      _Ax;
  mutable array_type      _r0;
  mutable array_type      _p;
  mutable array_type      _ph;
  mutable array_type      _v;
  mutable array_type      _r;
  mutable array_type      _s;
  mutable array_type      _t;
  mutable array_type      _d;
  mutable array_type      _d2;
  mutable array_type      _x2;
  mutable array_type      _sh;
  mutable array_type      _resids;  ///< residual history

 protected:
  /// \brief check and assign any illegal parameters to default setting
  inline void _check_pars() {
    if (rtol <= 0) rtol = DefaultSettings<value_type>::rtol;
    if (maxit == 0u) maxit = DefaultSettings<value_type>::max_iters;
  }

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
    _resids.reserve(maxit + 1);
    _resids.resize(1);
  }

  /// \brief validation checking
  template <class Matrix>
  inline bool _validate(const Matrix &A, const array_type &b,
                        const array_type &x) const {
    if (!_M || _M->empty()) return true;
    if (_M->nrows() != A.nrows()) return true;
    if (b.size() != A.nrows()) return true;
    if (b.size() != x.size()) return true;
    if (rtol <= 0.0) return true;
    if (maxit == 0u) return true;
    return false;
  }

  /// \brief show information
  /// \param[in] with_init_guess solve with initial guess flag
  inline void _show(const bool with_init_guess) const {
    hilucsi_info("- QMRCGSTAB -\nrtol=%g\nmaxiter=%zd\ninit-guess: %s\n", rtol,
                 maxit, (with_init_guess ? "yes" : "no"));
  }

  /// \brief low level solve kernel
  /// \tparam Matrix user input matrix type, see \ref CRS and \ref CCS
  /// \tparam StreamerCout cout streamer type
  /// \tparam StreamerCerr cerr streamer type
  /// \param[in] A user matrix
  /// \param[in] b right hard size
  /// \param[in,out] x0 initial guess and solution on output
  /// \param[in] Cout "stdout" streamer
  /// \param[in] Cerr "stderr" streamer
  template <class Matrix, class StreamerCout, class StreamerCerr>
  std::pair<int, size_type> _solve(const Matrix &A, const array_type &b,
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
    const auto &M = *_M;
    M.solve(_p, _ph);
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

      M.solve(_s, _sh);
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
      Cout("  At iteration %zd, relative residual is %g.", iter,
           _resids.back());
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

      M.solve(_p, _ph);
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
};

}  // namespace ksp
}  // namespace hilucsi

#endif  // _HILUCSI_KSP_QMRCGSTAB_HPP