//@HEADER
//----------------------------------------------------------------------------
//                Copyright (C) 2019 The HILUCSI AUTHORS
//----------------------------------------------------------------------------
//@HEADER

/// \file hilucsi/ksp/FBICGSTAB.hpp
/// \brief FBICGSTAB implementation
/// \authors Qiao,

#ifndef _HILUCSI_KSP_FBICGSTAB_HPP
#define _HILUCSI_KSP_FBICGSTAB_HPP

#include <cmath>
#include <limits>
#include <utility>

#include "hilucsi/ksp/common.hpp"
#include "hilucsi/utils/math.hpp"

namespace hilucsi {
namespace ksp {

/// \class FBICGSTAB
/// \tparam MType preconditioner type, see \ref HILUCSI
/// \tparam ValueType if not given, i.e. \a void, then use value in \a MType
/// \brief Flexible BICGSTAB implementation
/// \ingroup bicgstab
template <class MType, class ValueType = void>
class FBICGSTAB
    : public internal::KSP<FBICGSTAB<MType, ValueType>, MType, ValueType> {
 protected:
  using _base = internal::KSP<FBICGSTAB<MType, ValueType>, MType, ValueType>;
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
  inline static const char *repr() { return "FBICGSTAB"; }

  using _base::rtol;

  using _base::maxit;

  using _base::inner_steps;
  using _base::lamb1;
  using _base::lamb2;

  FBICGSTAB() = default;

  /// \brief constructor with all essential parameters
  /// \param[in] M multilevel ILU preconditioner
  /// \param[in] rel_tol relative tolerance for convergence (1e-6 for double)
  /// \param[in] max_iters maximum number of iterations
  /// \param[in] inner_steps maximum inner iterations for jacobi kernels
  explicit FBICGSTAB(
      std::shared_ptr<M_type> M,
      const scalar_type       rel_tol = DefaultSettings<value_type>::rtol,
      const size_type max_iters       = DefaultSettings<value_type>::max_iters,
      const size_type inner_steps = DefaultSettings<value_type>::inner_steps)
      : _base(M, rel_tol, max_iters, inner_steps) {
    if (_M && _M->nrows()) _ensure_data_capacities(_M->nrows());
  }

 protected:
  /// \name workspace
  /// Workspace for BICGSTAB
  /// @{
  mutable array_type _r;
  mutable array_type _v;
  mutable array_type _p;
  mutable array_type _p_hat;
  mutable array_type _s;
  mutable array_type _r_tld;
  /// @}
  using _base::_M;
  using _base::_resids;

 protected:
  /// \brief ensure internal buffer sizes
  /// \param[in] n right-hand size array size
  inline void _ensure_data_capacities(const size_type n) const {
    _r.resize(n);
    _v.resize(n);
    _p.resize(n);
    _p_hat.resize(n);
    _s.resize(n);
    _r_tld.resize(n);
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
      std::copy(b.cbegin(), b.cend(), _r.begin());
    else {
      A.mv(x, _r);
      for (size_type i(0); i < n; ++i) _r[i] = b[i] - _r[i];
    }
    if ((_resids[0] = norm2(_r) / normb) <= rtol)
      return std::make_pair(flag, size_type(0));

    value_type  omega(1), alpha(0), rho_1(0);
    const auto &r_tld = _r_tld;
    size_type   iter(1);

    // main loop
    for (; iter <= maxit; ++iter) {
      const auto rho = inner(r_tld, _r);  // direction
      if (rho == 0) {
        Cerr(__HILUCSI_FILE__, __HILUCSI_FUNC__, __LINE__,
             "Stagnated detected at iteration %zd.", iter);
        flag = STAGNATED;
        break;
      }

      if (iter > 1u) {
        const auto beta = (rho / rho_1) * (alpha / omega);
        for (size_type i(0); i < n; ++i)
          _p[i] = _r[i] * beta * (_p[i] - omega * _v[i]);
      } else
        std::copy(_r.cbegin(), _r.cend(), _p.begin());

      // call solve
      if (M.solve(A, _p, innersteps, _p_hat)) {
        Cerr(__HILUCSI_FILE__, __HILUCSI_FUNC__, __LINE__,
             "Failed to call M operator at iteration %zd.", iter);
        flag = M_SOLVE_ERROR;
        break;
      }
      A.mv(_p_hat, _v);
      alpha = rho / inner(r_tld, _v);
      for (size_type i(0); i < n; ++i) {
        x[i] += alpha * _p_hat[i];
        _s[i] = _r[i] - alpha * _v[i];
      }
      const auto resid = norm2(_s) / normb;
      // early convergence checking
      if (resid <= rtol) {
        Cout(
            "  Early convergence detected at iteration %zd, relative residual "
            "is %g.",
            iter, _resids.back());
        _resids.push_back(resid);
        break;
      }

      // call solve
      if (M.solve(A, _s, innersteps, _p_hat)) {
        Cerr(__HILUCSI_FILE__, __HILUCSI_FUNC__, __LINE__,
             "Failed to call M operator at iteration %zd.", iter);
        flag = M_SOLVE_ERROR;
        break;
      }
      A.mv(_p_hat, _v);
      omega = inner(_v, _s) / norm2_sq(_v);
      for (size_type i(0); i < n; ++i) {
        x[i] += omega * _p_hat[i];
        _r[i] = _s[i] - omega * _v[i];
      }
      _resids.push_back(norm2(_r) / normb);
      Cout("  At iteration %zd (inner:%zd), relative residual is %g.", iter,
           innersteps * 2, _resids.back());

      if (_resids.back() <= rtol) break;
      if (std::isnan(_resids.back()) || std::isinf(_resids.back())) {
        Cerr(__HILUCSI_FILE__, __HILUCSI_FUNC__, __LINE__,
             "Solver break-down detected at iteration %zd.", iter);
        flag = BREAK_DOWN;
        break;
      }
      if (_resids.back() > 100) {
        Cerr(__HILUCSI_FILE__, __HILUCSI_FUNC__, __LINE__,
             "Divergence encountered at iteration %zd.", iter);
        flag = DIVERGED;
        break;
      }
      if (omega == 0) {
        Cerr(__HILUCSI_FILE__, __HILUCSI_FUNC__, __LINE__,
             "Solver break-down detected at iteration %zd.", iter);
        flag = BREAK_DOWN;
        break;
      }
      rho_1 = rho;
    }
    if (flag == SUCCESS && _resids.back() > rtol) {
      Cerr(__HILUCSI_FILE__, __HILUCSI_FUNC__, __LINE__,
           "Reached maxit iteration limit %zd.", maxit);
      flag = DIVERGED;
      iter = maxit;
    }

    return std::make_pair(flag, iter);
  }

  /// \brief low level solve kernel
  /// \tparam Matrix user input matrix type, see \ref CRS and \ref CCS
  /// \tparam Operator "preconditioner" operator type, see \ref internal::Jacobi
  /// \tparam StreamerCout cout streamer type
  /// \tparam StreamerCerr cerr streamer type
  /// \param[in] A user matrix
  /// \param[in] M "preconditioner" operator
  /// \param[in] b right hard size
  /// \param[in] zero_start flag to indicate \a x0 starts with all zeros
  /// \param[in,out] x0 initial guess and solution on output
  /// \param[in] Cout "stdout" streamer
  /// \param[in] Cerr "stderr" streamer
  template <class Matrix, class Operator, class StreamerCout,
            class StreamerCerr>
  std::pair<int, size_type> _solve_auto(const Matrix &A, const Operator &M,
                                        const array_type &b,
                                        const bool zero_start, array_type &x0,
                                        const StreamerCout &Cout,
                                        const StreamerCerr &Cerr) const {
    constexpr static int _D =
        std::numeric_limits<scalar_type>::digits10 / 2 + 1;
    const static scalar_type _inc_eps =
        std::pow(scalar_type(10), -(scalar_type)_D);
    constexpr static size_type max_j_steps = 3u;

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
      std::copy(b.cbegin(), b.cend(), _r.begin());
    else {
      A.mv(x, _r);
      for (size_type i(0); i < n; ++i) _r[i] = b[i] - _r[i];
    }
    if ((_resids[0] = norm2(_r) / normb) <= rtol)
      return std::make_pair(flag, size_type(0));

    value_type  omega(1), alpha(0), rho_1(0);
    const auto &r_tld = _r_tld;
    size_type   iter(1);

    unsigned res_inc_flag = 0u;

    // main loop
    for (; iter <= maxit; ++iter) {
      const bool do_j = iter <= max_j_steps;

      const auto rho = inner(r_tld, _r);  // direction
      if (rho == 0) {
        Cerr(__HILUCSI_FILE__, __HILUCSI_FUNC__, __LINE__,
             "Stagnated detected at iteration %zd.", iter);
        flag = STAGNATED;
        break;
      }

      if (iter > 1u) {
        const auto beta = (rho / rho_1) * (alpha / omega);
        for (size_type i(0); i < n; ++i)
          _p[i] = _r[i] * beta * (_p[i] - omega * _v[i]);
      } else
        std::copy(_r.cbegin(), _r.cend(), _p.begin());

      // call solve
      if (do_j || res_inc_flag) {
        if (M.solve(A, _p, inner_steps, _p_hat)) {
          Cerr(__HILUCSI_FILE__, __HILUCSI_FUNC__, __LINE__,
               "Failed to call M operator at iteration %zd.", iter);
          flag = M_SOLVE_ERROR;
          break;
        }
      } else
        M.M().solve(_p, _p_hat);
      A.mv(_p_hat, _v);
      alpha = rho / inner(r_tld, _v);
      for (size_type i(0); i < n; ++i) {
        x[i] += alpha * _p_hat[i];
        _s[i] = _r[i] - alpha * _v[i];
      }
      const auto resid = norm2(_s) / normb;
      // early convergence checking
      if (resid <= rtol) {
        Cout(
            "  Early convergence detected at iteration %zd, relative residual "
            "is %g.",
            iter, _resids.back());
        _resids.push_back(resid);
        break;
      }

      // call solve
      if (do_j || res_inc_flag) {
        if (M.solve(A, _s, inner_steps, _p_hat)) {
          Cerr(__HILUCSI_FILE__, __HILUCSI_FUNC__, __LINE__,
               "Failed to call M operator at iteration %zd.", iter);
          flag = M_SOLVE_ERROR;
          break;
        }
      } else
        M.M().solve(_s, _p_hat);
      A.mv(_p_hat, _v);
      omega = inner(_v, _s) / norm2_sq(_v);
      for (size_type i(0); i < n; ++i) {
        x[i] += omega * _p_hat[i];
        _r[i] = _s[i] - omega * _v[i];
      }
      _resids.push_back(norm2(_r) / normb);
      do {
        const size_type innersteps = do_j || res_inc_flag ? inner_steps * 2 : 2;
        Cout("  At iteration %zd (inner:%zd), relative residual is %g.", iter,
             innersteps, _resids.back());
      } while (false);

      if (_resids.back() <= rtol) break;
      if (std::isnan(_resids.back()) || std::isinf(_resids.back())) {
        Cerr(__HILUCSI_FILE__, __HILUCSI_FUNC__, __LINE__,
             "Solver break-down detected at iteration %zd.", iter);
        flag = BREAK_DOWN;
        break;
      }
      if (_resids.back() > 100) {
        Cerr(__HILUCSI_FILE__, __HILUCSI_FUNC__, __LINE__,
             "Divergence encountered at iteration %zd.", iter);
        flag = DIVERGED;
        break;
      }
      // check if residual increasing
      if (_resids.back() >= _resids[iter - 1] * (1 - _inc_eps)) {
        Cerr(__HILUCSI_FILE__, __HILUCSI_FUNC__, __LINE__,
             "Residual begins to increase at iteration %zd. Turn on inner "
             "refinements.",
             iter);
        res_inc_flag = 2u;
      } else {
        if (res_inc_flag) --res_inc_flag;
      }

      if (omega == 0) {
        Cerr(__HILUCSI_FILE__, __HILUCSI_FUNC__, __LINE__,
             "Solver break-down detected at iteration %zd.", iter);
        flag = BREAK_DOWN;
        break;
      }
      rho_1 = rho;
    }
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

#endif  // _HILUCSI_KSP_FBICGSTAB_HPP