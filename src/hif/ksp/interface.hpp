///////////////////////////////////////////////////////////////////////////////
//                  This file is part of the HIFIR library                   //
///////////////////////////////////////////////////////////////////////////////

/*!
 * \file hif/ksp/interface.hpp
 * \brief Interface solver for KSP methods
 * \author Qiao Chen

\verbatim
Copyright (C) 2021 NumGeom Group at Stony Brook University

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

#ifndef _HIF_KSP_INTERFACE_HPP
#define _HIF_KSP_INTERFACE_HPP

#include "hif/ds/CompressedStorage.hpp"
#include "hif/ksp/FBICGSTAB.hpp"
#include "hif/ksp/FGMRES.hpp"
#include "hif/ksp/FQMRCGSTAB.hpp"
#include "hif/ksp/GMRES.hpp"
#include "hif/ksp/TGMRESR.hpp"

namespace hif {
namespace ksp {

/*!
 * \addtogroup ksp
 * @{
 */

/// \class KSPSolver
/// \tparam MType preconditioner type, see \ref HIF
/// \tparam ValueType interface value type, default is that used in \a MType
/// \brief semi-abstract base class of all ksp solvers
///
/// This ABC works for all KSP solvers but with \a MType and \a ValueType
/// explicitly given. This uses runtime override, but the overhead is negligible
/// due to all virtual functions are top level interfaces.
template <class MType, class ValueType = void>
class KSPSolver {
 public:
  typedef MType M_type;  ///< preconditioner type
  typedef typename std::conditional<std::is_void<ValueType>::value,
                                    typename M_type::array_type,
                                    Array<ValueType>>::type array_type;
  ///< array type
  typedef typename array_type::value_type value_type;  ///< value type
  typedef typename array_type::size_type  size_type;   ///< size type
  typedef CRS<value_type, typename M_type::index_type> crs_type;  ///< crs type
  typedef CCS<value_type, typename M_type::index_type> ccs_type;  ///< crs type
  typedef typename DefaultSettings<value_type>::scalar_type scalar_type;
  ///< scalar type
  typedef std::function<void(const void *, const size_type, const char, void *,
                             const char, const bool)>
      func_type;
  ///< user callback for computing A*x

  /// \brief virtual destructor
  virtual ~KSPSolver() {}

  /// \name parameters
  /// @{
  virtual void        set_rtol(const scalar_type)            = 0;
  virtual scalar_type get_rtol() const                       = 0;
  virtual void        set_maxit(const size_type)             = 0;
  virtual size_type   get_maxit() const                      = 0;
  virtual void        set_inner_steps(const size_type)       = 0;
  virtual size_type   get_inner_steps(const size_type) const = 0;
  virtual void        set_lamb1(const value_type)            = 0;
  virtual value_type  get_lamb1() const                      = 0;
  virtual void        set_lamb2(const value_type)            = 0;
  virtual value_type  get_lamb2() const                      = 0;
  virtual void        set_restart_or_cycle(const int) {
    hif_warning("this solver does not support restart or truncated cycle!");
  }
  virtual int get_restart_or_cycle() const {
    hif_warning("this solver does not support restart or truncated cycle!");
    return 0;
  }
  virtual const Array<scalar_type> &get_resids() const = 0;
  /// @}

  // handy utilities

  /// \brief check name representation
  virtual const char *repr() const = 0;

  /// \brief check if the solver uses Arnoldi iteration
  virtual bool is_arnoldi() const = 0;

  /// \brief check if mixed solver
  inline constexpr bool is_mixed() const {
    return !std::is_same<value_type, typename M_type::value_type>::value;
  }

  /// \brief set preconditioner
  virtual void set_M(std::shared_ptr<M_type>) = 0;

  /// \brief get preconditioner
  virtual std::shared_ptr<M_type> get_M() const = 0;

  /// \brief solve interface for CRS
  virtual std::pair<int, size_type> solve(
      const crs_type &, const array_type &, array_type &, const int = TRADITION,
      const bool /* with_init_guess */ = false,
      const bool /* verbose */         = true) const = 0;

  /// \brief solve interface for CCS
  virtual std::pair<int, size_type> solve(
      const ccs_type &, const array_type &, array_type &, const int = TRADITION,
      const bool /* with_init_guess */ = false,
      const bool /* verbose */         = true) const = 0;

  /// \brief solve interface for user callback
  virtual std::pair<int, size_type> solve(
      const func_type &, const array_type &, array_type &,
      const int = TRADITION, const bool /* with_init_guess */ = false,
      const bool /* verbose */ = true) const = 0;
};

/// \class KSPAdaptor
/// \tparam KspType Ksp solver type, e.g. see \ref FGMRES
/// \brief Adaptor for ksp solvers
/// \note All member functions here are final, thus further overriding is
///       strictly prevented
template <class KspType>
class KSPAdaptor
    : public KspType,
      public KSPSolver<typename KspType::M_type, typename KspType::value_type> {
 protected:
  using _base = KspType;  ///< base
  using _abc_base =
      KSPSolver<typename KspType::M_type, typename KspType::value_type>;
  ///< abc base

 public:
  using M_type          = typename KspType::M_type;     ///< preconditioner type
  using scalar_type     = typename _base::scalar_type;  ///< scalar
  using value_type      = typename _base::value_type;   ///< value
  using size_type       = typename _base::size_type;    ///< size
  using array_type      = typename _base::array_type;   ///< array
  using crs_type        = typename _abc_base::crs_type;  ///< \ref CRS type
  using ccs_type        = typename _abc_base::ccs_type;  ///< \ref CCS type
  using abc_solver_type = _abc_base;                     ///< abc type
  using func_type = typename _abc_base::func_type;  ///< functor type for A*x

  KSPAdaptor() = default;

  /// \brief constructor with preconditioner
  /// \param[in] M preconditioner
  explicit KSPAdaptor(std::shared_ptr<M_type> M) : _base(M), _abc_base() {}

  /// \brief virtual destructor
  virtual ~KSPAdaptor() {}

  /// \name parameters
  /// @{
  virtual void set_rtol(const scalar_type tol) override final {
    _base::rtol = tol;
  }
  virtual scalar_type get_rtol() const override final { return _base::rtol; }
  virtual void        set_maxit(const size_type max_iters) override final {
    _base::maxit = max_iters;
  }
  virtual size_type get_maxit() const override final { return _base::maxit; }
  virtual void      set_inner_steps(const size_type innersteps) override final {
    _base::inner_steps = innersteps;
  }
  virtual size_type get_inner_steps(const size_type) const override final {
    return _base::inner_steps;
  }
  virtual void set_lamb1(const value_type lamb) override final {
    _base::lamb1 = lamb;
  }
  virtual value_type get_lamb1() const override final { return _base::lamb1; }
  virtual void       set_lamb2(const value_type lamb) override final {
    _base::lamb2 = lamb;
  }
  virtual value_type get_lamb2() const override final { return _base::lamb2; }

  virtual void set_restart_or_cycle(const int rs) override final {
    static const std::string name = _base::repr();
    if (name.find("GMRES") != std::string::npos)
      _base::restart = rs;
    else
      _abc_base::set_restart_or_cycle(rs);
  }
  virtual int get_restart_or_cycle() const override final {
    static const std::string name = _base::repr();
    if (name.find("GMRES") != std::string::npos) return _base::restart;
    return _abc_base::get_restart_or_cycle();
  }
  virtual const Array<scalar_type> &get_resids() const override final {
    return _base::resids();
  }
  /// @}

  /// \brief check name representation
  virtual const char *repr() const override final { return _base::repr(); }

  /// \brief check if the solver uses Arnoldi iteration
  virtual bool is_arnoldi() const override final {
    static const std::string name = _base::repr();
    return name.find("GMRES") != std::string::npos;
  }

  /// \brief set preconditioner
  /// \param[in] M multilevel ILU preconditioner
  virtual void set_M(std::shared_ptr<M_type> M) override final {
    _base::set_M(M);
  }

  /// \brief get preconditioner
  virtual std::shared_ptr<M_type> get_M() const override final {
    return _base::get_M();
  }

  /// \brief solve for solution with \ref CRS type
  /// \param[in] A user input matrix
  /// \param[in] b right-hand side vector
  /// \param[in,out] x solution
  /// \param[in] kernel default is TRADITION
  /// \param[in] with_init_guess if \a false (default), then assign zero to
  ///             \a x as starting values
  /// \param[in] verbose if \a true (default), enable verbose printing
  virtual std::pair<int, size_type> solve(
      const crs_type &A, const array_type &b, array_type &x,
      const int kernel = TRADITION, const bool with_init_guess = false,
      const bool verbose = true) const override final {
    return _base::solve(A, b, x, kernel, with_init_guess, verbose);
  }

  /// \brief solve for solution with \ref CCS type
  /// \param[in] A user input matrix
  /// \param[in] b right-hand side vector
  /// \param[in,out] x solution
  /// \param[in] kernel default is TRADITION
  /// \param[in] with_init_guess if \a false (default), then assign zero to
  ///             \a x as starting values
  /// \param[in] verbose if \a true (default), enable verbose printing
  virtual std::pair<int, size_type> solve(
      const ccs_type &A, const array_type &b, array_type &x,
      const int kernel = TRADITION, const bool with_init_guess = false,
      const bool verbose = true) const override final {
    return _base::solve(A, b, x, kernel, with_init_guess, verbose);
  }

  /// \brief solve for solution with user callback for A*x
  /// \param[in] A user callback for computing A*x
  /// \param[in] b right-hand side vector
  /// \param[in,out] x solution
  /// \param[in] kernel default is TRADITION
  /// \param[in] with_init_guess if \a false (default), then assign zero to
  ///             \a x as starting values
  /// \param[in] verbose if \a true (default), enable verbose printing
  virtual std::pair<int, size_type> solve(
      const func_type &A, const array_type &b, array_type &x,
      const int kernel = TRADITION, const bool with_init_guess = false,
      const bool verbose = true) const override final {
    return _base::solve(A, b, x, kernel, with_init_guess, verbose);
  }

 private:
  using _base::solve;
};

/// \class KSPFactory
/// \tparam MType preconditioner type, see \ref HIF
/// \tparam ValueType interface value type, default is that used in \a MType
/// \brief type factory for all ksp methods
template <class MType, class ValueType = void>
class KSPFactory {
 public:
  /// \name static
  /// @{
  using fgmres_type     = FGMRES<MType, ValueType>;
  using tgmresr_type    = TGMRESR<MType, ValueType>;
  using fqmrcgstab_type = FQMRCGSTAB<MType, ValueType>;
  using fbicgstab_type  = FBICGSTAB<MType, ValueType>;
  using gmres_type      = GMRES<MType, ValueType>;
  /// @}

  /// \name runtime
  /// @{
  using fgmres     = KSPAdaptor<fgmres_type>;
  using tgmresr    = KSPAdaptor<tgmresr_type>;
  using fqmrcgstab = KSPAdaptor<fqmrcgstab_type>;
  using fbicgstab  = KSPAdaptor<fbicgstab_type>;
  using gmres      = KSPAdaptor<gmres_type>;
  using abc_solver = typename fgmres::abc_solver_type;
  /// @}
};

/*!
 *@}
 */

}  // namespace ksp
}  // namespace hif

#endif  // _HIF_KSP_INTERFACE_HPP
