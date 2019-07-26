//@HEADER
//----------------------------------------------------------------------------
//                Copyright (C) 2019 The HILUCSI AUTHORS
//----------------------------------------------------------------------------
//@HEADER

/// \file hilucsi/ksp/FGMRES.hpp
/// \brief Flexible GMRES implementation
/// \authors Qiao,

#ifndef _HILUCSI_KSP_FGMRES_HPP
#define _HILUCSI_KSP_FGMRES_HPP

#include <cmath>
#include <limits>
#include <utility>

#include "hilucsi/ksp/common.hpp"
#include "hilucsi/utils/math.hpp"

namespace hilucsi {
namespace ksp {

/// \class FGMRES
/// \tparam MType preconditioner type, see \ref HILUCSI
/// \tparam ValueType if not given, i.e. \a void, then use value in \a MType
/// \brief flexible GMRES implementation
/// \ingroup gmres
template <class MType, class ValueType = void>
class FGMRES
    : public internal::KSP<FGMRES<MType, ValueType>, MType, ValueType> {
 protected:
  using _base = internal::KSP<FGMRES<MType, ValueType>, MType, ValueType>;
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

 public:
  static_assert(std::is_floating_point<scalar_type>::value,
                "must be floating point type");

  /// \brief get the solver name
  inline static const char *repr() { return "FGMRES"; }

  using _base::rtol;

  using _base::maxit;

  using _base::restart;

  using _base::inner_steps;
  using _base::lamb1;
  using _base::lamb2;

  FGMRES() = default;

  /// \brief constructor with all essential parameters
  /// \param[in] M multilevel ILU preconditioner
  /// \param[in] rel_tol relative tolerance for convergence (1e-6 for double)
  /// \param[in] rs restart, default is 30
  /// \param[in] max_iters maximum number of iterations
  /// \param[in] max_inner_steps maximum inner iterations for jacobi kernels
  explicit FGMRES(
      std::shared_ptr<M_type> M,
      const scalar_type       rel_tol = DefaultSettings<value_type>::rtol,
      const int               rs      = 30,
      const size_type max_iters       = DefaultSettings<value_type>::max_iters,
      const size_type max_inner_steps =
          DefaultSettings<value_type>::inner_steps)
      : _base(M, rel_tol, max_iters, max_inner_steps) {
    restart = rs;
    if (restart <= 0) restart = 30;
    if (_M && _M->nrows()) _ensure_data_capacities(_M->nrows());
  }

 protected:
  /// \name workspace
  /// Workspace for GMRES algorithm
  /// @{
  mutable array_type _y;
  mutable array_type _R;
  mutable array_type _Q;  ///< Q space
  mutable array_type _Z;
  mutable array_type _J;
  mutable array_type _v;
  mutable array_type _w;
  /// @}
  using _base::_M;
  using _base::_resids;

 protected:
  /// \brief check and assign any illegal parameters to default setting
  inline void _check_pars() {
    if (restart <= 0) restart = 30;
    _base::_check_pars();
  }

  /// \brief ensure internal buffer sizes
  /// \param[in] n right-hand size array size
  inline void _ensure_data_capacities(const size_type n) const {
    _y.resize(restart + 1);
    _R.resize(restart * (restart + 1) / 2);  // packed storage
    _Q.resize(n * restart);
    _Z.resize(_Q.size());
    _J.resize(2 * restart);
    _v.resize(n);
    _w.resize(std::max(n, size_type(restart)));
    _base::_init_resids();
  }

  /// \brief validation checking
  template <class Matrix>
  inline bool _validate(const Matrix &A, const array_type &b,
                        const array_type &x) const {
    if (restart <= 0) return true;
    return _base::_validate(A, b, x);
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
  /// \param[in] max_inner_steps maximum inner steps for jacobi-style kernels
  /// \param[in] zero_start flag to indicate \x0 starts with all zeros
  /// \param[in,out] x0 initial guess and solution on output
  /// \param[in] Cout "stdout" streamer
  /// \param[in] Cerr "stderr" streamer
  template <class Matrix, class Operator, class StreamerCout,
            class StreamerCerr>
  std::pair<int, size_type> _solve(const Matrix &A, const Operator &M,
                                   const array_type &b,
                                   const size_type   max_inner_steps,
                                   const bool zero_start, array_type &x0,
                                   const StreamerCout &Cout,
                                   const StreamerCerr &Cerr) const {
    constexpr static int _D =
        std::numeric_limits<scalar_type>::digits10 / 2 + 1;
    const static scalar_type _stag_eps =
        std::pow(scalar_type(10), -(scalar_type)_D);

    size_type       iter(0);
    const size_type n     = b.size();
    const auto      beta0 = norm2(b);
    if (beta0 == 0) {
      std::fill_n(x0.begin(), n, value_type(0));
      return std::make_pair((int)SUCCESS, size_type(0));
    }
    // record  time after preconditioner
    _ensure_data_capacities(n);
    const size_type max_outer_iters =
        (size_type)std::ceil((scalar_type)maxit / restart);
    auto &      x    = x0;
    int         flag = SUCCESS;
    scalar_type resid(1);
    for (size_type it_outer = 0u; it_outer < max_outer_iters; ++it_outer) {
      Cout("Enter outer iteration %zd...", it_outer + 1);
      if (it_outer)
        Cerr(__HILUCSI_FILE__, __HILUCSI_FUNC__, __LINE__,
             "Couldn\'t solve with %d restarts.", restart);
      // initial residual
      if (iter || !zero_start) {
        A.mv(x, _v);
        for (size_type i = 0u; i < n; ++i) _v[i] = b[i] - _v[i];
      } else
        std::copy_n(b.cbegin(), n, _v.begin());
      const auto beta     = norm2(_v);
      _y[0]               = beta;
      const auto inv_beta = 1. / beta;
      if (!it_outer) _resids[0] = beta / beta0;
      for (size_type i = 0u; i < n; ++i) _Q[i] = _v[i] * inv_beta;
      size_type       j(0);
      auto            R_itr     = _R.begin();
      const size_type exp_steps = std::min(it_outer + 1, max_inner_steps);
      for (;;) {
        const auto jn = j * n;
        std::copy(_Q.cbegin() + jn, _Q.cbegin() + jn + n, _v.begin());
        if (n < (size_type)restart) _w.resize(n);
        if (M.solve(A, _v, exp_steps, _w)) {
          Cerr(__HILUCSI_FILE__, __HILUCSI_FUNC__, __LINE__,
               "Failed to call M operator at iteration %zd.", iter);
          flag = M_SOLVE_ERROR;
          break;
        }
        std::copy(_w.cbegin(), _w.cend(), _Z.begin() + jn);
        A.mv(_w, _v);
        if (n < (size_type)restart) _w.resize(restart);
        for (size_type k = 0u; k <= j; ++k) {
          auto itr = _Q.cbegin() + k * n;

          _w[k]          = inner(_v, itr);
          const auto tmp = _w[k];
          for (size_type i = 0u; i < n; ++i) _v[i] -= tmp * itr[i];
        }
        const auto v_norm2 = norm2_sq(_v);
        const auto v_norm  = std::sqrt(v_norm2);
        if (j + 1 < (size_type)restart) {
          auto       itr      = _Q.begin() + jn + n;
          const auto inv_norm = 1. / v_norm;
          for (size_type i = 0u; i < n; ++i) itr[i] = inv_norm * _v[i];
        }
        auto J1 = _J.begin(), J2 = J1 + restart;
        for (size_type colJ = 0u; colJ + 1u <= j; ++colJ) {
          const auto tmp = _w[colJ];
          _w[colJ]       = conj(J1[colJ]) * tmp + conj(J2[colJ]) * _w[colJ + 1];
          _w[colJ + 1]   = -J2[colJ] * tmp + J1[colJ] * _w[colJ + 1];
        }
        const auto rho        = std::sqrt(_w[j] * _w[j] + v_norm2);
        J1[j]                 = _w[j] / rho;
        J2[j]                 = v_norm / rho;
        _y[j + 1]             = -J2[j] * _y[j];
        _y[j]                 = conj(J1[j]) * _y[j];
        _w[j]                 = rho;
        R_itr                 = std::copy_n(_w.cbegin(), j + 1, R_itr);
        const auto resid_prev = resid;
        resid                 = abs(_y[j + 1]) / beta0;
        _resids.push_back(resid);
        if (std::isnan(resid) || std::isinf(resid)) {
          Cerr(__HILUCSI_FILE__, __HILUCSI_FUNC__, __LINE__,
               "Solver break-down detected at iteration %zd.", iter);
          flag = BREAK_DOWN;
          break;
        }
        if (resid >= resid_prev * (1 - _stag_eps)) {
          Cerr(__HILUCSI_FILE__, __HILUCSI_FUNC__, __LINE__,
               "Stagnated detected at iteration %zd.", iter);
          flag = STAGNATED;
          break;
        } else if (iter >= maxit) {
          Cerr(__HILUCSI_FILE__, __HILUCSI_FUNC__, __LINE__,
               "Reached maxit iteration limit %zd.", maxit);
          flag = DIVERGED;
          break;
        }
        ++iter;
        Cout("  At iteration %zd (inner:%zd), relative residual is %g.", iter,
             exp_steps, resid);
        if (resid <= rtol || j + 1 >= (size_type)restart) break;
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
      if (resid <= rtol || flag != SUCCESS) break;
    }
    return std::make_pair(flag, iter);
  }
};
}  // namespace ksp
}  // namespace hilucsi

#endif  // _HILUCSI_KSP_FGMRES_HPP