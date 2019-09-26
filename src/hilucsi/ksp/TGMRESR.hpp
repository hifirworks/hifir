///////////////////////////////////////////////////////////////////////////////
//                This file is part of HILUCSI project                       //
///////////////////////////////////////////////////////////////////////////////

/*!
 * \file hilucsi/ksp/TGMRESR.hpp
 * \brief GMRESR with truncated cycling implementation
 * \author Qiao Chen

\verbatim
Copyright (C) 2019 NumGeom Group at Stony Brook University

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

#ifndef _HILUCSI_KSP_TGMRES_HPP
#define _HILUCSI_KSP_TGMRES_HPP

#include <cmath>
#include <limits>
#include <utility>

#include "hilucsi/ksp/common.hpp"
#include "hilucsi/utils/math.hpp"

namespace hilucsi {
namespace ksp {

/// \class TGMRESR
/// \tparam MType preconditioner type, see \ref HILUCSI
/// \tparam ValueType if not given, i.e. \a void, then use value in \a MType
/// \brief GMRESR implementation with truncated cycling
/// \ingroup gmres
/// \note We reuse \a restart as cycling period
template <class MType, class ValueType = void>
class TGMRESR
    : public internal::KSP<TGMRESR<MType, ValueType>, MType, ValueType> {
 protected:
  using _base = internal::KSP<TGMRESR<MType, ValueType>, MType, ValueType>;
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
  inline static const char *repr() { return "TGMRESR"; }

  using _base::rtol;

  using _base::maxit;

  using _base::restart;

  using _base::inner_steps;
  using _base::lamb1;
  using _base::lamb2;

  TGMRESR() = default;

  /// \brief constructor with all essential parameters
  /// \param[in] M multilevel ILU preconditioner
  /// \param[in] rel_tol relative tolerance for convergence (1e-6 for double)
  /// \param[in] cc truncated cycling frequency, default is 10
  /// \param[in] max_iters maximum number of iterations
  /// \param[in] innersteps maximum inner iterations for jacobi kernels
  explicit TGMRESR(
      std::shared_ptr<M_type> M,
      const scalar_type       rel_tol = DefaultSettings<value_type>::rtol,
      const int               cc      = 10,
      const size_type max_iters       = DefaultSettings<value_type>::max_iters,
      const size_type innersteps = DefaultSettings<value_type>::inner_steps)
      : _base(M, rel_tol, max_iters, innersteps) {
    restart = cc;
    if (restart <= 0) restart = 10;
    if (_M && _M->nrows()) _ensure_data_capacities(_M->nrows());
  }

 protected:
  /// \name workspace
  /// Workspace for GMRES algorithm
  /// @{
  mutable array_type                   _Q;  ///< Q space
  mutable array_type                   _Z;
  mutable array_type                   _v;
  mutable array_type                   _w;
  mutable array_type                   _r;
  mutable typename M_type::iarray_type _perm;
  /// @}
  using _base::_M;
  using _base::_resids;

 protected:
  /// \brief check and assign any illegal parameters to default setting
  inline void _check_pars() {
    if (restart <= 0) restart = 10;
    _base::_check_pars();
  }

  /// \brief ensure internal buffer sizes
  /// \param[in] n right-hand size array size
  inline void _ensure_data_capacities(const size_type n) const {
    _Q.resize(n * restart);
    _Z.resize(_Q.size());
    _v.resize(n);
    _w.resize(n);
    _r.resize(n);
    _perm.resize(maxit + 1);
    // build periodic permutation table
    const size_type buckets = maxit / restart;
    const int       offsets = maxit - buckets * restart;
    for (size_type j(0); j < buckets; ++j) {
      const auto start = j * restart;
      for (int i(0); i < restart; ++i) _perm[i + start] = i;
    }
    if (offsets) {
      const auto start = buckets * restart;
      int        i(0);
      for (int j(0); j < offsets; ++j, ++i) _perm[j + start] = i;
    }
    _base::_init_resids();
  }

  /// \brief validation checking
  template <class Matrix>
  inline bool _validate(const Matrix &A, const array_type &b,
                        const array_type &x) const {
    if (restart <= 0) return true;
    return _base::_validate(A, b, x);
  }
#if 0
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
  /// \param[in] innersteps inner steps for jacobi-style kernels
  /// \param[in] zero_start flag to indicate \x0 starts with all zeros
  /// \param[in,out] x0 initial guess and solution on output
  /// \param[in] Cout "stdout" streamer
  /// \param[in] Cerr "stderr" streamer
  template <class Matrix, class Operator, class StreamerCout,
            class StreamerCerr>
  std::pair<int, size_type> _solve2(const Matrix &A, const Operator &M,
                                    const array_type &b,
                                    const size_type   innersteps,
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
    _ensure_data_capacities(n);
    const size_type cycle = restart;
    auto &          x     = x0;
    int             flag  = SUCCESS;
    // scalar_type     resid(1);
    // initialize residual
    value_type beta = beta0;
    if (!zero_start) {
      A.mv(x, _v);
      for (size_type i = 0u; i < n; ++i) _v[i] = b[i] - _v[i];
      beta = norm2(_v);
    } else
      std::copy_n(b.cbegin(), n, _v.begin());
    _resids[0] = beta / beta0;
    _y[0]      = beta;
    for (size_type i(0); i < n; ++i) _Q[i] = _v[i] / beta;

    for (;;) {
      const size_type j = _perm[iter], jn = j * n, j_next = _perm[iter + 1],
                      start = iter < cycle ? 0 : iter - cycle + 1;
      std::copy(_Q.cbegin() + jn, _Q.cbegin() + jn + n, _v.begin());
      if (n < maxit) _w.resize(n);
      if (M.solve(A, _v, innersteps, _w)) {
        Cerr(__HILUCSI_FILE__, __HILUCSI_FUNC__, __LINE__,
             "Failed to call M operator at iteration %zd.", iter);
        flag = M_SOLVE_ERROR;
        break;
      }
      std::copy_n(_w.cbegin(), n, _Z.begin() + jn);
      A.mv(_w, _v);
      if (n < maxit) _w.resize(maxit);
      for (size_type k = start; k <= iter; ++k) {
        const auto perm_k = _perm[k];
        auto       itr    = _Q.cbegin() + perm_k * n;
        const auto tmp    = inner(_v, itr);
        _w[k]             = tmp;
        for (size_type i = 0u; i < n; ++i) _v[i] -= tmp * itr[i];
      }
      const auto v_norm2 = norm2_sq(_v);
      const auto v_norm  = std::sqrt(v_norm2);
      if (iter + 1 < maxit) {
        auto       itr      = _Q.begin() + j_next * n;
        const auto inv_norm = 1. / v_norm;
        for (size_type i = 0u; i < n; ++i) itr[i] = inv_norm * _v[i];
      }
      auto J1 = _J.begin(), J2 = J1 + maxit;
      for (size_type colJ = start; colJ < iter; ++colJ) {
        const auto tmp = _w[colJ];
        _w[colJ]       = conj(J1[colJ]) * tmp + conj(J2[colJ]) * _w[colJ + 1];
        _w[colJ + 1]   = -J2[colJ] * tmp + J1[colJ] * _w[colJ + 1];
      }
      const auto rho = std::sqrt(_w[iter] * _w[iter] + v_norm2);
      J1[iter]       = _w[iter] / rho;
      J2[iter]       = v_norm / rho;
      _y[iter + 1]   = -J2[iter] * _y[iter];
      _y[iter]       = conj(J1[iter]) * _y[iter];
      _w[iter]       = rho;

      // update P, or Z in our case
      auto itr = _Z.begin() + jn;
      do {
        const auto inv_rho = 1. / rho;
        for (size_type k = start; k < iter; ++k) {
          const auto pk   = _perm[k];
          auto       itr2 = _Z.cbegin() + pk * n;
          const auto h_im = _w[k];
          for (size_type i = 0; i < n; ++i) itr[i] -= h_im * itr2[i];
        }
        for (size_type i = 0; i < n; ++i) itr[i] *= inv_rho;
      } while (false);

      // update x
      const auto eta = _y[iter];
      for (size_type i(0); i < n; ++i) x[i] += eta * itr[i];

      const auto resid_prev = _resids[iter];
      if (iter < cycle)
        _resids.push_back(abs(_y[iter + 1]) / beta0);
      else {
        A.mv(x, _v);
        for (size_type i(0); i < n; ++i) _v[i] -= b[i];
        _resids.push_back(norm2(_v) / beta0);
      }
      if (std::isnan(_resids.back()) || std::isinf(_resids.back())) {
        Cerr(__HILUCSI_FILE__, __HILUCSI_FUNC__, __LINE__,
             "Solver break-down detected at iteration %zd.", iter);
        flag = BREAK_DOWN;
        break;
      }
      if (_resids.back() >= resid_prev * (1 - _stag_eps)) {
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
           innersteps, _resids.back());
      if (_resids.back() <= rtol) break;
    }
    return std::make_pair(flag, iter);
  }
#endif
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
  /// \param[in] innersteps inner steps for jacobi-style kernels
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
    _ensure_data_capacities(n);
    const size_type cycle = restart;
    auto &          x     = x0;
    int             flag  = SUCCESS;

    // initialize residual
    value_type beta = beta0;
    if (!zero_start) {
      A.mv(x, _r);
      for (size_type i = 0u; i < n; ++i) _r[i] = b[i] - _r[i];
      beta = norm2(_r);
    } else
      std::copy_n(b.cbegin(), n, _r.begin());
    _resids[0] = beta / beta0;
    auto &u = _w, &c = _v;  // wrap var name to align with the paper
    for (;;) {
      const size_type j = _perm[iter], jn = j * n, j_next = _perm[iter + 1],
                      start = iter < cycle ? 0 : iter - cycle + 1;
      auto c_itr = _Q.begin() + jn, u_itr = _Z.begin() + jn;
      if (M.solve(A, _r, innersteps, u)) {
        Cerr(__HILUCSI_FILE__, __HILUCSI_FUNC__, __LINE__,
             "Failed to call M operator at iteration %zd.", iter);
        flag = M_SOLVE_ERROR;
        break;
      }
      A.mv(u, c);
      for (size_type k = start; k < iter; ++k) {
        const auto p_k   = _perm[k];
        auto       c_k   = _Q.cbegin() + p_k * n;
        auto       u_k   = _Z.cbegin() + p_k * n;
        const auto alpha = inner(c, c_k);
        for (size_type i(0); i < n; ++i) {
          c[i] -= alpha * c_k[i];
          u[i] -= alpha * u_k[i];
        }
      }
      const auto c_norm_inv = 1. / norm2(c);
      for (size_type i(0); i < n; ++i) {
        c_itr[i] = c[i] * c_norm_inv;
        u_itr[i] = u[i] * c_norm_inv;
      }

      // update solution and residual
      const auto eta = inner(_r, c_itr);
      for (size_type i(0); i < n; ++i) {
        x[i] += eta * u_itr[i];
        _r[i] -= eta * c_itr[i];
      }
      const auto resid_prev = _resids[iter];
      _resids.push_back(norm2(_r) / beta0);
      if (std::isnan(_resids.back()) || std::isinf(_resids.back())) {
        Cerr(__HILUCSI_FILE__, __HILUCSI_FUNC__, __LINE__,
             "Solver break-down detected at iteration %zd.", iter);
        flag = BREAK_DOWN;
        break;
      }
      if (_resids.back() >= resid_prev * (1 - _stag_eps)) {
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
           innersteps, _resids.back());
      if (_resids.back() <= rtol) break;
    }
    return std::make_pair(flag, iter);
  }
};
}  // namespace ksp
}  // namespace hilucsi

#endif  // _HILUCSI_KSP_TGMRES_HPP
