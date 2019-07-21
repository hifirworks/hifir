//@HEADER
//----------------------------------------------------------------------------
//                Copyright (C) 2019 The HILUCSI AUTHORS
//----------------------------------------------------------------------------
//@HEADER

/// \file hilucsi/ksp/common.hpp
/// \brief common interface (helpers) for KSP solvers
/// \authors Qiao,

#ifndef _HILUCSI_KSP_COMMON_HPP
#define _HILUCSI_KSP_COMMON_HPP

#include <complex>
#include <cstddef>
#include <memory>
#include <string>

#include "hilucsi/utils/common.hpp"
#include "hilucsi/utils/log.hpp"

namespace hilucsi {
namespace ksp {

/*!
 * \addtogroup ksp
 * @{
 */

/// \brief flags for returned information
enum {
  INVALID_ARGS  = -2,  ///< invalid function arguments
  M_SOLVE_ERROR = -1,  ///< preconditioner solve error
  SUCCESS       = 0,   ///< successful converged
  DIVERGED      = 1,   ///< iteration diverged
  STAGNATED     = 2,   ///< iteration stagnated
  BREAK_DOWN    = 3,   ///< solver break down
};

/// \brief get flag representation
/// \param[in] solver solver name
/// \param[in] flag solver returned flag
inline std::string flag_repr(const std::string &solver, const int flag) {
  switch (flag) {
    case INVALID_ARGS:
      return solver + "_" + "INVALID_ARGS";
    case M_SOLVE_ERROR:
      return solver + "_" + "M_SOLVE_ERROR";
    case SUCCESS:
      return solver + "_" + "SUCCESS";
    case DIVERGED:
      return solver + "_" + "DIVERGED";
    case STAGNATED:
      return solver + "_" + "STAGNATED";
    case BREAK_DOWN:
      return solver + "_" + "BREAK_DOWN";
    default:
      return solver + "_" + "UNKNOWN";
  }
}

/// \class DefaultSettings
/// \tparam V value type
/// \brief parameters for default setting
template <class V>
class DefaultSettings {
 public:
  using scalar_type = typename ValueTypeTrait<V>::value_type;  ///< scalar
  static constexpr scalar_type rtol = sizeof(scalar_type) == 8ul ? 1e-6 : 1e-4;
  ///< default relative tolerance for residual convergence
  static constexpr std::size_t max_iters = 500u;
  ///< maximum number of iterations
  static constexpr std::size_t inner_steps = 2u;  ///< inner flexible iterations
};

/// \brief flags for flexible kernels
enum {
  TRADITION = 0,     ///< traditional kernel
  JACOBI,            ///< with inner jacobi iterations
  CHEBYSHEV_JACOBI,  ///< with inner Jacobi plus Chebyshev accelerations
};

/*!
 * @}
 */

namespace internal {

/*!
 * \addtogroup ksp
 * @{
 */

/// \class KSP
/// \brief Krylov subspace method base
/// \tparam ChildSolver child solver type
/// \tparam MType preconditioner type, see \ref HILUCSI
template <class ChildSolver, class MType>
class KSP {
  /// \brief cast to child solver
  inline ChildSolver &_this() { return static_cast<ChildSolver &>(*this); }

  /// \brief cast to child solver with constant specifier
  inline const ChildSolver &_this() const {
    return static_cast<const ChildSolver &>(*this);
  }

 public:
  typedef MType                           M_type;      ///< preconditioner
  typedef typename M_type::array_type     array_type;  ///< value array
  typedef typename array_type::size_type  size_type;   ///< size type
  typedef typename array_type::value_type value_type;  ///< value type
  typedef typename DefaultSettings<value_type>::scalar_type scalar_type;
  ///< scalar type from value_type

  static_assert(std::is_floating_point<scalar_type>::value,
                "must be floating point type");

  scalar_type rtol = DefaultSettings<value_type>::rtol;
  ///< relative convergence tolerance
  size_type maxit = DefaultSettings<value_type>::max_iters;
  ///< max numer of iterations
  size_type inner_steps = DefaultSettings<value_type>::inner_steps;
  ///< inner flexible iterations
  value_type lamb1 = 0.9;  ///< est of largest eigenvalue
  value_type lamb2 = 0.0;  ///< est of smallest eigenvalue

  KSP() = default;

  /// \brief constructor with all essential parameters
  /// \param[in] M multilevel ILU preconditioner
  /// \param[in] rel_tol relative tolerance for convergence (1e-6 for \a double)
  /// \param[in] max_iters maximum number of iterations
  /// \param[in] innersteps inner iterations for jacobi kernels
  explicit KSP(
      std::shared_ptr<M_type> M,
      const scalar_type       rel_tol = DefaultSettings<value_type>::rtol,
      const size_type max_iters       = DefaultSettings<value_type>::max_iters,
      const size_type innersteps = DefaultSettings<value_type>::inner_steps)
      : _M(M), rtol(rel_tol), maxit(max_iters), inner_steps(innersteps) {
    _this()._check_pars();
  }

  /// \brief set preconditioner
  /// \param[in] M multilevel ILU preconditioner
  inline void set_M(std::shared_ptr<M_type> M) {
    _M = M;  // increment internal reference counter
    if (_M && _M->nrows()) _this()._ensure_data_capacities(_M->nrows());
  }

  /// \brief get preconditioner
  inline std::shared_ptr<M_type> get_M() const { return _M; }

  /// \brief get residual array
  inline const array_type &resids() const { return _resids; }

 protected:
  std::shared_ptr<M_type> _M;       ///< preconditioner operator
  mutable array_type      _resids;  ///< residual history

 protected:
  /// \brief check and assign any illegal parameters to default setting
  inline void _check_pars() {
    if (rtol <= 0) rtol = DefaultSettings<value_type>::rtol;
    if (maxit == 0u) maxit = DefaultSettings<value_type>::max_iters;
    if (inner_steps == 0u)
      inner_steps = DefaultSettings<value_type>::inner_steps;
  }

  /// \brief ensure history residuals
  inline void _init_resids() const {
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
    if (inner_steps == 0u) return true;
    return false;
  }

  /// \brief show information
  /// \param[in] kernel kernel name
  /// \param[in] with_init_guess solve with initial guess flag
  /// \param[in] rs restart, nonpositive values indicate N/A
  inline void _show(const char *kernel, const bool with_init_guess,
                    const int rs) const {
    hilucsi_info(
        "- %s -\n"
        "rtol=%g\n"
        "restart=%s\n"
        "maxiter=%zd\n"
        "flex-kernel: %s\n"
        "init-guess: %s\n",
        ChildSolver::repr(), rtol,
        (rs > 0 ? std::to_string(rs).c_str() : "N/A"), maxit, kernel,
        (with_init_guess ? "yes" : "no"));
  }
};

/// \class JacobiBase
/// \tparam Child child jacobi iteration object, either \ref Jacobi or
///         \ref ChebyshevJacobi
/// \tparam MType "preconditioner" operator type, see \ref HILUCSI
/// \brief generic base class for Jacobi iterations
///
/// It's worth noting that instead of a real Jacobi processing, i.e. taking
/// into account of (block) diagonals, we abstract the processing by thinking
/// the multilevel preconditioner as a fashion of "diagonal" of the input A.
/// This is indeed true due to the fact that multilevel preconditioning
/// always processes on the leading block of the Schur complements thus
/// resulting a structure of hierarchical diagonal blocks.
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
  ///
  /// This function implements the abstracted Jacobi processes as follows
  ///
  /// \f{eqnarray*}{
  ///   \mathbf{Ax}&=&\mathbf{b} \\
  ///   (\mathbf{D+A-D})\mathbf{x}&=&\mathbf{b} \\
  ///   \mathbf{Dx}&=&\mathbf{b}-(\mathbf{A-D})\mathbf{x} \\
  ///   \mathbf{x}&=&\mathbf{D}^{-1}(\mathbf{r}+\mathbf{Dx}) \\
  ///   \mathbf{x}_{k+1}&=&\mathbf{D}^{-1}\mathbf{r}+\mathbf{x}_k
  /// \f}
  ///
  /// We then replace \f$\mathbf{D}\f$ with the multilevel preconditioner
  /// \f$\mathbf{M}\f$. Therefore, for each of the iteration, we have one
  /// matrix-vector multiplication plus an M-solve.
  template <class Matrix>
  inline bool solve(const Matrix &A, const array_type &b, const size_type N,
                    array_type &x0) const {
    if (_M.nrows() != A.nrows() || _M.ncols() != A.ncols()) return true;
    if (b.size() != A.nrows() || A.ncols() != x0.size()) return true;
    size_type       iters(0);
    const size_type n(b.size());
    auto &          x = x0;
    std::fill(x.begin(), x.end(), value_type(0));
    for (; iters < N; ++iters) {
      // copy rhs to x
      std::copy(x.cbegin(), x.cend(), _xk.begin());
      if (iters) {
        // compute A*xk=x
        A.mv(_xk, x);
        // compute residual x=b-x
        for (size_type i(0); i < n; ++i) x[i] = b[i] - x[i];
      } else
        std::copy_n(b.cbegin(), n, x.begin());
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

}  // namespace ksp
}  // namespace hilucsi

#endif  // _HILUCSI_KSP_COMMON_HPP