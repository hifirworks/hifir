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

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <numeric>
#include <utility>

#include "hilucsi/ksp/common.hpp"
#include "hilucsi/utils/log.hpp"

namespace hilucsi {
namespace ksp {
namespace internal {

/*!
 * \addtogroup gmres
 * @{
 */

/// \class JacobiBase
/// \tparam Child child jacobi iteration object, either \ref Jacobi or
///         \ref ChebyshevJacobi
/// \tparam MType "preconditioner" operator type, see \ref HILUCSI
/// \brief generic base class for Jacobi iterations
template <class Child, class MType>
class JacobiBase {
  /// \brief helper for casting to \a Child
  inline const Child &_this() const {
    return static_cast<const Child &>(*this);
  }

 public:
  typedef MType                           M_type;      ///< preconditioner type
  typedef typename M_type::array_type     array_type;  ///< value array type
  typedef typename array_type::value_type value_type;  ///< value type
  typedef typename array_type::size_type  size_type;   ///< size type

  JacobiBase() = delete;

  /// \brief constructor with reference to preconditioner
  /// \param[in] M preconditioner, see \ref HILUCSI
  explicit JacobiBase(const M_type &M) : _M(M), _xk(M.ncols()), _r(M.nrows()) {
    if (M.nrows() && _xk.empty()) hilucsi_error("memory allocation failed");
    if (M.ncols() && _r.empty()) hilucsi_error("memory allocation failed");
  }

  /// \brief check if the M's address is the same
  /// \param[in] M multilevel ILU
  inline bool is_same_M(const M_type &M) const { return &_M == &M; }

  /// \brief generic implementation of Jacobi iterations with explicit steps
  /// \tparam Matrix matrix type, see \ref CRS or \ref CCS
  /// \param[in] A input matrix
  /// \param[in] b right-hand side vector
  /// \param[in] N number of iterations
  /// \param[out] x0 solution of Jacobi after \a N iterations
  /// \return if return \a true, then solving failed
  template <class Matrix>
  inline bool solve(const Matrix &A, const array_type &b, const size_type N,
                    array_type &x0) const {
    // The Jacobi iteration reads:
    //    A*x=b
    //    (D+A-D)*x=b
    //    D*x_{k+1}=b-(A-D)*x_k
    //    x_{k+1}=inv(D)*(b-(A-D)*x_k)
    //    x_{k+1}=inv(D)*(b-A*x_k+D*x_k)
    //    x_{k+1}=inv(D)*res_k+x_k
    if (_M.nrows() != A.nrows() || _M.ncols() != A.ncols()) return true;
    if (b.size() != A.nrows() || A.ncols() != x0.size()) return true;
    size_type       iters(0);
    const size_type n(b.size());
    auto &          x = x0;
    std::fill(x.begin(), x.end(), value_type(0));
    for (; iters < N; ++iters) {
      // copy rhs to x
      std::copy(x.cbegin(), x.cend(), _xk.begin());
      // compute A*xk=x
      A.mv(_xk, x);
      // compute residual x=b-x
      for (size_type i(0); i < n; ++i) x[i] = b[i] - x[i];
      // compute inv(M)*x=r
      _M.solve(x, _r);
      // call child's post process to update to current solution
      _this()._update(iters, x);
    }
    return false;
  }

 protected:
  const M_type &     _M;   ///< reference to preconditioner
  mutable array_type _xk;  ///< previous solution
  mutable array_type _r;   ///< inv(M)*residual
};

/// \class Jacobi
/// \brief regular Jacobi iterations
/// \tparam MType "preconditioner" operator type, see \ref HILUCSI
template <class MType>
class Jacobi : public internal::JacobiBase<Jacobi<MType>, MType> {
  using _base = internal::JacobiBase<Jacobi<MType>, MType>;  ///< base
  friend _base;

 public:
  typedef typename _base::M_type     M_type;      ///< preconditioner type
  typedef typename _base::array_type array_type;  ///< value array type
  typedef typename _base::size_type  size_type;   ///< value type

  Jacobi() = delete;

  /// \brief constructor with reference to preconditioner
  /// \param[in] M preconditioner, see \ref HILUCSI
  explicit Jacobi(const M_type &M) : _base(M) {}

 protected:
  using _base::_r;
  using _base::_xk;

  /// \brief default post processing for static inheritance
  /// \param[out] x solution of step k
  inline void _update(const size_type, array_type &x) const {
    const size_type n = x.size();
    for (size_type i(0); i < n; ++i) x[i] = _r[i] + _xk[i];
  }
};

/// \class ChebyshevJacobi
/// \brief Jacobi iterations with Chebyshev accelerations
/// \tparam MType "preconditioner" operator type, see \ref HILUCSI
template <class MType>
class ChebyshevJacobi
    : public internal::JacobiBase<ChebyshevJacobi<MType>, MType> {
  using _base = internal::JacobiBase<ChebyshevJacobi<MType>, MType>;  ///< base
  friend _base;

 public:
  typedef typename _base::M_type     M_type;      ///< preconditioner type
  typedef typename _base::array_type array_type;  ///< value array type
  typedef typename _base::size_type  size_type;   ///< size type
  typedef typename _base::value_type value_type;  ///< value type

  ChebyshevJacobi() = delete;

  /// \brief constructor with preconditioner and largest/smallest eig est
  /// \param[in] M reference to preconditioner
  /// \param[in] lamb1 largest eigenvalue estimation
  /// \param[in] lamb2 smallest eigenvalue estimation
  ChebyshevJacobi(const M_type &M, const value_type lamb1,
                  const value_type lamb2)
      : _base(M), _rho(2), _xkk(M.ncols()) {
    if (M.ncols() && _xkk.empty()) hilucsi_error("memory allocation failed");
    _compute_coeffs(lamb1, lamb2);
  }

 protected:
  mutable value_type _rho;
  mutable array_type _xkk;
  value_type         _gamma;
  value_type         _sigma2;
  using _base::_r;
  using _base::_xk;

 protected:
  /// \brief compute essential coefficients
  inline void _compute_coeffs(const value_type l1, const value_type l2) {
    _gamma  = 2. / (2. - l1 - l2);
    _sigma2 = 0.5 * _gamma * (l1 - l2);
    _sigma2 = _sigma2 * _sigma2;
  }

  /// \brief default post processing for static inheritance
  /// \param[in] iter current inner iteration counts
  /// \param[out] x solution of step k
  inline void _update(const size_type iter, array_type &x) const {
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

/// \class DummyJacobi
/// \brief dummy "Jacobi" that has the same solve interface as, say, \ref Jacobi
/// \tparam MType "preconditioner" operator type, see \ref HILUCSI
template <class MType>
class DummyJacobi {
 public:
  typedef MType                       M_type;
  typedef typename M_type::array_type array_type;
  typedef typename M_type::size_type  size_type;

  DummyJacobi() = delete;

  explicit DummyJacobi(const M_type &M) : _M(M) {}

  template <class Matrix>
  inline bool solve(const Matrix &, const array_type &b, const size_type,
                    array_type &x0) const {
    _M.solve(b, x0);
    return false;
  }

 protected:
  const M_type &_M;
};

/*!
 * @}
 */

}  // namespace internal

/// \class FGMRES
/// \tparam MType preconditioner type, see \ref HILUCSI
/// \brief flexible GMRES implementation
/// \ingroup gmres
template <class MType>
class FGMRES {
 public:
  typedef MType                           M_type;      ///< preconditioner
  typedef typename M_type::array_type     array_type;  ///< value array
  typedef typename array_type::size_type  size_type;   ///< size type
  typedef typename array_type::value_type value_type;  ///< value type
  typedef typename DefaultSettings<value_type>::scalar_type scalar_type;
  ///< scalar type from value_type

 public:
  static_assert(std::is_floating_point<scalar_type>::value,
                "must be floating point type");

  enum {
    TRADITION = 0,     ///< traditional kernel
    JACOBI,            ///< with inner jacobi iterations
    CHEBYSHEV_JACOBI,  ///< with inner Jacobi plus Chebyshev accelerations
  };

  /// \brief get the solver name
  inline static const char *repr() { return "FGMRES"; }

  scalar_type rtol = DefaultSettings<value_type>::rtol;
  ///< relative convergence tolerance
  int       restart = 30;  ///< restart
  size_type maxit   = DefaultSettings<value_type>::max_iters;
  ///< max numer of iterations
  size_type  max_inners = 4u;   ///< maximum inner iterations
  value_type lamb1      = 0.9;  ///< est of largest eigenvalue
  value_type lamb2      = 0.0;  ///< est of smallest eigenvalue

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
      const size_type max_inner_steps = 4u)
      : _M(M),
        rtol(rel_tol),
        restart(rs),
        maxit(max_iters),
        max_inners(max_inner_steps) {
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
  /// \param[in] trunc using truncating style instead hard restart
  /// \param[in] verbose if \a true (default), enable verbose printing
  template <class Matrix>
  inline std::pair<int, size_type> solve_precond(
      const Matrix &A, const array_type &b, array_type &x,
      const bool with_init_guess = false, const bool trunc = false,
      const bool verbose = true) const {
    const static hilucsi::internal::StdoutStruct  Cout;
    const static hilucsi::internal::StderrStruct  Cerr;
    const static hilucsi::internal::DummyStreamer Dummy_streamer;

    if (_validate(A, b)) return std::make_pair(INVALID_ARGS, size_type(0));
    if (verbose) _show("tradition", with_init_guess, trunc);
    x.resize(b.size());
    if (!with_init_guess) std::fill(x.begin(), x.end(), value_type(0));
    const internal::DummyJacobi<M_type> M(*_M);
    if (verbose) hilucsi_info("Calling traditional GMRES kernel...");
    return verbose
               ? _solve(A, M, b, 1u, trunc, x, Cout, Cerr)
               : _solve(A, M, b, 1u, trunc, x, Dummy_streamer, Dummy_streamer);
  }

  /// \brief solve with \ref _M as Jacobi operator
  /// \tparam Matrix user input type, see \ref CRS and \ref CCS
  /// \param[in] A user input matrix
  /// \param[in] b right-hand side vector
  /// \param[in,out] x solution
  /// \param[in] with_init_guess if \a false (default), then assign zero to
  ///             \a x as starting values
  /// \param[in] trunc using truncating style instead hard restart
  /// \param[in] verbose if \a true (default), enable verbose printing
  template <class Matrix>
  inline std::pair<int, size_type> solve_jacobi(
      const Matrix &A, const array_type &b, array_type &x,
      const bool with_init_guess = false, const bool trunc = false,
      const bool verbose = true) const {
    const static hilucsi::internal::StdoutStruct  Cout;
    const static hilucsi::internal::StderrStruct  Cerr;
    const static hilucsi::internal::DummyStreamer Dummy_streamer;

    if (_validate(A, b)) return std::make_pair(INVALID_ARGS, size_type(0));
    if (verbose) _show("Jacobi", with_init_guess, trunc);
    const internal::Jacobi<M_type> M(*_M);
    x.resize(b.size());
    if (!with_init_guess) std::fill(x.begin(), x.end(), value_type(0));
    return verbose ? _solve(A, M, b, max_inners, trunc, x, Cout, Cerr)
                   : _solve(A, M, b, max_inners, trunc, x, Dummy_streamer,
                            Dummy_streamer);
  }

  /// \brief solve with \ref _M as Jacobi operator plus Chebyshev acceleration
  /// \tparam Matrix user input type, see \ref CRS and \ref CCS
  /// \param[in] A user input matrix
  /// \param[in] b right-hand side vector
  /// \param[in,out] x solution
  /// \param[in] with_init_guess if \a false (default), then assign zero to
  ///             \a x as starting values
  /// \param[in] trunc using truncating style instead hard restart
  /// \param[in] verbose if \a true (default), enable verbose printing
  template <class Matrix>
  inline std::pair<int, size_type> solve_chebyshev(
      const Matrix &A, const array_type &b, array_type &x,
      const bool with_init_guess = false, const bool trunc = false,
      const bool verbose = true) const {
    const static hilucsi::internal::StdoutStruct  Cout;
    const static hilucsi::internal::StderrStruct  Cerr;
    const static hilucsi::internal::DummyStreamer Dummy_streamer;

    if (_validate(A, b)) return std::make_pair(INVALID_ARGS, size_type(0));
    if (verbose) {
      _show("Chebyshev", with_init_guess, trunc);
      hilucsi_warning("Chebyshev Jacobi with GMRES is experiemental...");
    }
    const internal::ChebyshevJacobi<M_type> M(*_M, lamb1, lamb2);
    x.resize(b.size());
    if (!with_init_guess) std::fill(x.begin(), x.end(), value_type(0));
    return verbose ? _solve(A, M, b, max_inners, trunc, x, Cout, Cerr)
                   : _solve(A, M, b, max_inners, trunc, x, Dummy_streamer,
                            Dummy_streamer);
  }

  /// \brief solve for solution
  /// \tparam Matrix user input type, see \ref CRS and \ref CCS
  /// \param[in] A user input matrix
  /// \param[in] b right-hand side vector
  /// \param[in,out] x solution
  /// \param[in] kernel default is TRADITION, i.e. \ref solve_precond
  /// \param[in] with_init_guess if \a false (default), then assign zero to
  ///             \a x as starting values
  /// \param[in] trunc using truncating style instead hard restart
  /// \param[in] verbose if \a true (default), enable verbose printing
  template <class Matrix>
  inline std::pair<int, size_type> solve(const Matrix &A, const array_type &b,
                                         array_type &x,
                                         const int   kernel = TRADITION,
                                         const bool  with_init_guess = false,
                                         const bool  trunc           = false,
                                         const bool  verbose = true) const {
    switch (kernel) {
      case TRADITION:
        return solve_precond(A, b, x, with_init_guess, trunc, verbose);
      case JACOBI:
        return solve_jacobi(A, b, x, with_init_guess, trunc, verbose);
      case CHEBYSHEV_JACOBI:
        return solve_chebyshev(A, b, x, with_init_guess, trunc, verbose);
      default:
        hilucsi_warning("Unknown choice of FGMRES kernel %d", kernel);
        return std::make_pair(-99, size_type(0));
    }
  }

 protected:
  std::shared_ptr<M_type> _M;  ///< preconditioner operator
  mutable array_type      _y;
  mutable array_type      _R;
  mutable array_type      _Q;  ///< Q space
  mutable array_type      _Z;
  mutable array_type      _J;
  mutable array_type      _v;
  mutable array_type      _resids;  ///< residual history
  mutable array_type      _w;

 protected:
  /// \brief check and assign any illegal parameters to default setting
  inline void _check_pars() {
    if (rtol <= 0) rtol = DefaultSettings<value_type>::rtol;
    if (restart <= 0) restart = 30;
    if (maxit == 0u) maxit = DefaultSettings<value_type>::max_iters;
    if (max_inners == 0u) max_inners = 4u;
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
    _resids.reserve(maxit);
    _resids.resize(0);
    _w.resize(n);
  }

  /// \brief validation checkings
  template <class Matrix>
  inline bool _validate(const Matrix &A, const array_type &b) const {
    if (!_M || _M->empty()) return true;
    if (_M->nrows() != A.nrows()) return true;
    if (b.size() != A.nrows()) return true;
    if (rtol <= 0.0) return true;
    if (restart <= 0) return true;
    if (maxit == 0u) return true;
    if (max_inners == 0u) return true;
    return false;
  }

  /// \brief show information
  /// \param[in] kernel kernel name
  /// \param[in] with_init_guess solve with initial guess flag
  /// \param[in] trunc using truncation flag
  inline void _show(const char *kernel, const bool with_init_guess,
                    const bool trunc) const {
    hilucsi_info(
        "- FGMRES -\nrtol=%g\nrestart=%d\nmaxiter=%zd\nkernel: %s\ninit-guess: "
        "%s\ntrunc: %s\n",
        rtol, restart, maxit, kernel, (with_init_guess ? "yes" : "no"),
        (trunc ? "yes" : "no"));
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
  /// \param[in] trunc trucation flag
  /// \param[in,out] x0 initial guess and solution on output
  /// \param[in] Cout "stdout" streamer
  /// \param[in] Cerr "stderr" streamer
  template <class Matrix, class Operator, class StreamerCout,
            class StreamerCerr>
  std::pair<int, size_type> _solve(const Matrix &A, const Operator &M,
                                   const array_type &b,
                                   const size_type   max_inner_steps,
                                   const bool trunc, array_type &x0,
                                   const StreamerCout &Cout,
                                   const StreamerCerr &Cerr) const {
    constexpr static int _D =
        std::numeric_limits<scalar_type>::digits10 / 2 + 1;
    const static scalar_type _stag_eps =
        std::pow(scalar_type(10), -(scalar_type)_D);

    size_type       iter(0);
    const size_type n     = b.size();
    const auto      beta0 = std::sqrt(
        std::inner_product(b.cbegin(), b.cend(), b.cbegin(), value_type()));
    // record  time after preconditioner
    _ensure_data_capacities(n);
    const size_type max_outer_iters =
        (size_type)std::ceil((scalar_type)maxit / restart);
    auto &          x    = x0;
    int             flag = SUCCESS;
    scalar_type     resid(1);
    const size_type last_Q_pos = n * (restart - 1);
    for (size_type it_outer = 0u; it_outer < max_outer_iters; ++it_outer) {
      Cout("Enter outer iteration %zd...", it_outer + 1);
      if (it_outer) Cerr("Couldn\'t solve with %d restarts.", restart);
      // initial residual
      A.mv(x, _v);
      for (size_type i = 0u; i < n; ++i) _v[i] = b[i] - _v[i];
      const auto beta = std::sqrt(std::inner_product(
          _v.cbegin(), _v.cend(), _v.cbegin(), value_type(0)));
      _y[0]           = beta;
      if (!trunc || it_outer == 0u) {
        const auto inv_beta = 1. / beta;
        for (size_type i = 0u; i < n; ++i) _Q[i] = _v[i] * inv_beta;
      } else
        std::copy_n(_Q.cbegin() + last_Q_pos, n, _Q.begin());
      size_type       j(0);
      auto            R_itr     = _R.begin();
      const size_type exp_steps = std::min(it_outer + 1, max_inner_steps);
      for (;;) {
        const auto jn = j * n;
        std::copy(_Q.cbegin() + jn, _Q.cbegin() + jn + n, _v.begin());
        if (M.solve(A, _v, exp_steps, _w)) {
          Cerr("Failed to call M operator at iteration %zd.", iter);
          flag = M_SOLVE_ERROR;
          break;
        }
        std::copy(_w.cbegin(), _w.cend(), _Z.begin() + jn);
        A.mv(_w, _v);
        for (size_type k = 0u; k <= j; ++k) {
          auto itr = _Q.cbegin() + k * n;
          _w[k] =
              std::inner_product(_v.cbegin(), _v.cend(), itr, value_type(0));
          const auto tmp = _w[k];
          for (size_type i = 0u; i < n; ++i) _v[i] -= tmp * itr[i];
        }
        const auto v_norm2 = std::inner_product(_v.cbegin(), _v.cend(),
                                                _v.cbegin(), value_type(0));
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
        resid                 = std::abs(_y[j + 1]) / beta0;
        _resids.push_back(resid);
        if (resid >= resid_prev * (1 - _stag_eps)) {
          Cerr("Stagnated detected at iteration %zd.", iter);
          flag = STAGNATED;
          break;
        } else if (iter >= maxit) {
          Cerr("Reached maxit iteration limit %zd.", maxit);
          flag = DIVERGED;
          break;
        }
        ++iter;
        Cout("  At iteration %zd (inner:%zd), relative residual is %g.", iter,
             exp_steps, resid);
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
      if (resid < rtol || flag != SUCCESS) break;
    }
    return std::make_pair(flag, iter);
  }
};
}  // namespace ksp
}  // namespace hilucsi

#endif  // _HILUCSI_KSP_FGMRES_HPP