///////////////////////////////////////////////////////////////////////////////
//                  This file is part of the HIFIR library                   //
///////////////////////////////////////////////////////////////////////////////

/*!
 * \file hif/ksp/GMRES.hpp
 * \brief Right-preconditioned GMRES implementation
 * \author Qiao Chen

\verbatim
Copyright (C) 2020 NumGeom Group at Stony Brook University

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
\endverbatim

 */

#ifndef _HIF_KSP_GMRES_HPP
#define _HIF_KSP_GMRES_HPP

#include <cmath>
#include <limits>
#include <utility>

#include "hif/ksp/common.hpp"
#include "hif/utils/math.hpp"

namespace hif {
namespace ksp {

/// \class GMRES
/// \tparam MType preconditioner type, see \ref HIF
/// \tparam ValueType if not given, i.e. \a void, then use value in \a MType
/// \brief right-preconditioned GMRES implementation
/// \ingroup gmres
template <class MType, class ValueType = void>
class GMRES : public internal::KSP<GMRES<MType, ValueType>, MType, ValueType> {
 protected:
  using _base = internal::KSP<GMRES<MType, ValueType>, MType, ValueType>;
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
  inline static const char *repr() { return "GMRES"; }

  using _base::rtol;

  using _base::maxit;

  using _base::restart;

  using _base::inner_steps;
  using _base::lamb1;
  using _base::lamb2;

  GMRES() = default;

  /// \brief constructor with all essential parameters
  /// \param[in] M multilevel ILU preconditioner
  /// \param[in] rel_tol relative tolerance for convergence (1e-6 for double)
  /// \param[in] rs restart, default is 30
  /// \param[in] max_iters maximum number of iterations
  /// \param[in] max_inner_steps maximum inner iterations for jacobi kernels
  explicit GMRES(
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
  /// \tparam UseIR flag indicates whether or not enabling iterative refine
  /// \tparam Matrix user input matrix type, see \ref CRS and \ref CCS
  /// \tparam Operator "preconditioner" operator type, see \ref HIF
  /// \tparam StreamerCout cout streamer type
  /// \tparam StreamerCerr cerr streamer type
  /// \param[in] A user matrix
  /// \param[in] M "preconditioner" operator
  /// \param[in] b right hard size
  /// \param[in] innersteps inner steps for jacobi-style kernels, not used
  /// \param[in] zero_start flag to indicate \a x0 starts with all zeros
  /// \param[in,out] x0 initial guess and solution on output
  /// \param[in] Cout "stdout" streamer
  /// \param[in] Cerr "stderr" streamer
  /// \note This is MGS (i.e., modified Gram-Schmidt) kernel
  template <bool UseIR, class Matrix, class Operator, class StreamerCout,
            class StreamerCerr>
  std::pair<int, size_type> _solve(const Matrix &A, const Operator &M,
                                   const array_type &b,
                                   const size_type   innersteps,
                                   const bool zero_start, array_type &x0,
                                   const StreamerCout &Cout,
                                   const StreamerCerr &Cerr) const {
    constexpr static int _D =
        std::numeric_limits<scalar_type>::digits10 / 2 + 1;
    const static scalar_type _stag_eps =
        std::pow(scalar_type(10), -(scalar_type)_D);

    (void)innersteps;
    // warn that iterative refinement doesn't work for right-prec GMRES
    if (UseIR)
      Cerr(__HIF_FILE__, __HIF_FUNC__, __LINE__,
           "Right-preconditioned GMRES doesn\'t support iterative refinement.");
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
    int         stag_guard(0);
    for (size_type it_outer = 0u; it_outer < max_outer_iters; ++it_outer) {
      Cout("Enter outer iteration %zd...", it_outer + 1);
      if (it_outer)
        Cerr(__HIF_FILE__, __HIF_FUNC__, __LINE__,
             "Couldn\'t solve with %d restarts.", restart);
      // initial residual
      if (iter || !zero_start) {
        // A.multiply(x, _v);
        mt::multiply_nt(A, x, _v);
        for (size_type i = 0u; i < n; ++i) _v[i] = b[i] - _v[i];
      } else
        std::copy_n(b.cbegin(), n, _v.begin());
      const auto beta     = norm2(_v);
      _y[0]               = beta;
      const auto inv_beta = 1. / beta;
      if (!it_outer) _resids[0] = beta;
      for (size_type i = 0u; i < n; ++i) _Q[i] = _v[i] * inv_beta;
      size_type j(0);
      auto      R_itr = _R.begin();
      for (;;) {
        const auto jn = j * n;
        std::copy(_Q.cbegin() + jn, _Q.cbegin() + jn + n, _v.begin());
        if (n < (size_type)restart) _w.resize(n);
        M.solve(_v, _w);
        // A.multiply(_w, _v);
        mt::multiply_nt(A, _w, _v);
        if (n < (size_type)restart) _w.resize(restart);
        for (size_type k = 0u; k <= j; ++k) {
          auto itr       = _Q.cbegin() + k * n;
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
          _w[colJ] =
              conjugate(J1[colJ]) * tmp + conjugate(J2[colJ]) * _w[colJ + 1];
          _w[colJ + 1] = -J2[colJ] * tmp + J1[colJ] * _w[colJ + 1];
        }
        const auto rho        = std::sqrt(_w[j] * _w[j] + v_norm2);
        J1[j]                 = _w[j] / rho;
        J2[j]                 = v_norm / rho;
        _y[j + 1]             = -J2[j] * _y[j];
        _y[j]                 = conjugate(J1[j]) * _y[j];
        _w[j]                 = rho;
        R_itr                 = std::copy_n(_w.cbegin(), j + 1, R_itr);
        const auto resid_prev = resid;
        _resids.push_back(abs(_y[j + 1]));
        resid = _resids.back() / beta0;
        if (std::isnan(resid) || std::isinf(resid)) {
          Cerr(__HIF_FILE__, __HIF_FUNC__, __LINE__,
               "Solver break-down detected at iteration %zd.", iter);
          flag = BREAK_DOWN;
          break;
        }
        const bool is_stag = resid >= resid_prev * (1 - _stag_eps);
        if (is_stag) {
          ++stag_guard;
          if (stag_guard > 1) {
            Cerr(__HIF_FILE__, __HIF_FUNC__, __LINE__,
                 "Stagnated detected at iteration %zd.", iter);
            flag = STAGNATED;
            break;
          }
        } else if (iter >= maxit) {
          Cerr(__HIF_FILE__, __HIF_FUNC__, __LINE__,
               "Reached maxit iteration limit %zd.", maxit);
          flag = DIVERGED;
          break;
        }
        if (!is_stag) stag_guard = 0;
        ++iter;
        Cout("  At iteration %zd, relative residual is %g.", iter, resid);
        if (resid <= rtol || j + 1 >= (size_type)restart) break;
        ++j;
      }  // inf loop
      // backsolve
      for (int k = j; k > -1; --k) {
        --R_itr;
        _y[k] /= *R_itr;
        const auto tmp = _y[k];
        for (int i = k - 1; i > -1; --i) _y[i] -= tmp * *(--R_itr);
      }
      // compute Q*y
      std::fill_n(_v.begin(), n, 0);
      for (size_type i = 0u; i <= j; ++i) {
        const auto tmp   = _y[i];
        auto       Q_itr = _Q.cbegin() + i * n;
        for (size_type k = 0u; k < n; ++k) _v[k] += tmp * Q_itr[k];
      }
      // compute M solve
      _w.resize(n);
      M.solve(_v, _w);
      for (size_type k(0); k < n; ++k) x[k] += _w[k];  // accumulate sol
      if (resid <= rtol || flag != SUCCESS) break;
    }
    return std::make_pair(flag, iter);
  }
};
}  // namespace ksp
}  // namespace hif

#endif  // _HIF_KSP_FGMRES_HPP