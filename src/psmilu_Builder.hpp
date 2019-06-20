//@HEADER
//----------------------------------------------------------------------------
//                Copyright (C) 2019 The PSMILU AUTHORS
//----------------------------------------------------------------------------
//@HEADER

/// \file psmilu_Builder.hpp
/// \brief Top level user class for building MILU preconditioner
/// \authors Qiao,

#ifndef _PSMILU_BUILDER_HPP
#define _PSMILU_BUILDER_HPP

#include <algorithm>
#include <iterator>
#include <numeric>

#include "psmilu_CompressedStorage.hpp"
#include "psmilu_Options.h"
#include "psmilu_Prec.hpp"
#include "psmilu_Timer.hpp"
#include "psmilu_fac.hpp"
#include "psmilu_fac_defer.hpp"
#include "psmilu_fac_defer_thin.hpp"
#include "psmilu_fac_pvt.hpp"
#include "psmilu_prec_solve.hpp"
#include "psmilu_utils.hpp"
#include "psmilu_version.h"

namespace psmilu {

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace internal {
const static char *intro =
    "\n"
    "=======================================================================\n"
    "|    Pre-dominantly Symmetric Multi-level Incomplete LU (PS-MILU)     |\n"
    "|                                                                     |\n"
    "| PSMILU is a package for computing multi-level incomplete LU factor- |\n"
    "| ization with optimal time complexity. PSMILU is the first software  |\n"
    "| package to utilize the pre-dominantly symmetric systems, which occ- |\n"
    "| ur quite often but were not precisely defined and appreciated.      |\n"
    "|                                                                     |\n"
    "-----------------------------------------------------------------------\n"
    "\n"
    " Package information:\n"
    "\n"
    "\t\tCopyright (C) The PSMILU AUTHORS\n"
    "\t\tVersion: %d.%d.%d\n"
    "\t\tBuilt on: %s, %s\n"
    "\n"
    "=======================================================================\n";
static bool introduced = false;
}  // namespace internal
#endif  // DOXYGEN_SHOULD_SKIP_THIS

/*!
 * \addtogroup cpp
 * @{
 */

/// \class PSMILU
/// \tparam ValueType numerical value type, e.g. \a double
/// \tparam IndexType index type, e.g. \a int
/// \tparam OneBased if \a false (default), then assume C index system
/// \tparam SSSType default is LU with partial pivoting
///
/// This is top user interface (C++); it is designed as a preconditioner that
/// can be easily plugin other codes. There are two core member functions, 1)
/// \ref compute a multilevel ILU preconditioner and 2) \ref solve the
/// preconditioner system. For computing preconditioner, the input can be either
/// \ref CCS or \ref CRS, for solving, the input must be \ref Array. However,
/// be aware that both CCS/CRS and Array can be used as external data wrappers,
/// thus one should not worry about input data duplications.
///
/// The following is a simple workflow for Builder.
///
/// \code{.cpp}
///   #include <PSMILU.hpp>
///   using namespace psmilu;
///   using builder_t = Builder<double, int>; // C index
///   using crs_t = builder_t::crs_type;
///   int main() {
///     const auto A = wrap_crs<crs_t>(...); // make a data wrapper
///     builder_t builder;
///     builder.compute(A);
///     builder.solve(...);
///   }
/// \endcode
template <class ValueType, class IndexType, bool OneBased = false,
          SmallScaleType SSSType = SMALLSCALE_LUP>
class PSMILU {
 public:
  typedef ValueType         value_type;                     ///< value type
  typedef Array<value_type> array_type;                     ///< array type
  typedef IndexType         index_type;                     ///< index type
  typedef Array<index_type> iarray_type;                    ///< index array
  constexpr static bool     ONE_BASED = OneBased;           ///< index flag
  typedef CRS<value_type, index_type, ONE_BASED> crs_type;  ///< crs type
  typedef typename crs_type::other_type          ccs_type;  ///< ccs type
  constexpr static SmallScaleType sss_type = SSSType;  ///< small scale type
  typedef Precs<value_type, index_type, ONE_BASED, sss_type> precs_type;
  ///< multilevel preconditioner type
  typedef typename precs_type::value_type prec_type;  ///< single level prec
  typedef typename prec_type::size_type   size_type;  ///< size type

  /// \brief check empty or not
  inline bool empty() const { return _precs.empty(); }

  /// \brief check number of levels
  /// \note This function takes \f$\mathcal{O}(1)\f$ since C++11
  inline size_type levels() const {
    const size_type lvls = _precs.size();
    if (lvls)
      return lvls + !_precs.back().dense_solver.empty() +
             !_precs.back().sparse_solver.empty();
    return 0;
  }

  // utilities

  /// \brief get constant reference to preconditioners
  inline const precs_type &precs() const { return _precs; }

  /// \brief compute the overall nnz of the multilevel preconditioners
  inline size_type nnz() const {
    if (empty()) return 0u;
    size_type n(0);
    for (auto itr = _precs.cbegin(); itr != _precs.cend(); ++itr)
      n += itr->nnz();
    return n;
  }

  /// \brief get constant reference to a specific level
  /// \note This function takes linear time complexity
  inline const prec_type &prec(const size_type level) const {
    psmilu_error_if(level >= levels(), "%zd exceeds the total level number %zd",
                    level, levels());
    auto itr = _precs.cbegin();
    std::advance(itr, level);
    return *itr;
  }

  /// \brief factorize the MILU preconditioner
  /// \tparam CsType compressed storage input, either \ref CRS or \ref CCS
  /// \param[in] A input matrix
  /// \param[in] m0 leading block size, if it's zero (default), then the routine
  ///               will assume an asymmetric leading block.
  /// \param[in] opts control parameters, using the default values in the paper.
  /// \param[in] check if \a true (default), will perform validity checking
  /// \param[in] use_thin if \a true (default), then call single-list version
  /// \sa solve
  template <class CsType>
  inline void factorize(const CsType &A, const size_type m0 = 0u,
                        const Options &opts = get_default_options(),
                        const bool check = true, const bool use_thin = true) {
    use_thin ? deferred_factorize_thin(A, m0, opts, check)
             : deferred_factorize(A, m0, opts, check);
  }

  /// \brief deferred factorization
  /// \tparam CsType compressed storage input, either \ref CRS or \ref CCS
  /// \param[in] A input matrix
  /// \param[in] m0 leading block size, if it's zero (default), then the routine
  ///               will assume an asymmetric leading block.
  /// \param[in] opts control parameters, using the default values in the paper.
  /// \param[in] check if \a true (default), will perform validity checking
  /// \sa _deferred_factorize_kernel
  template <class CsType>
  inline void deferred_factorize(const CsType &A, const size_type m0 = 0u,
                                 const Options &opts  = get_default_options(),
                                 const bool     check = true) {
    static_assert(!(CsType::ONE_BASED ^ ONE_BASED), "inconsistent index base");

    const static internal::StdoutStruct  Crout_cout;
    const static internal::DummyStreamer Crout_cout_dummy;

    // print introduction
    if (psmilu_verbose(INFO, opts)) {
      if (!internal::introduced) {
        psmilu_info(internal::intro, PSMILU_GLOBAL_VERSION,
                    PSMILU_MAJOR_VERSION, PSMILU_MINOR_VERSION, __TIME__,
                    __DATE__);
        internal::introduced = true;
      }
      psmilu_info("Options (control parameters) are:\n");
      psmilu_info(opt_repr(opts).c_str());
    }
    const bool revert_warn = warn_flag();
    if (psmilu_verbose(NONE, opts)) (void)warn_flag(0);

    // check validity of the input system
    if (check) {
      if (psmilu_verbose(INFO, opts))
        psmilu_info("perform input matrix validity checking");
      A.check_validity();
    }

    DefaultTimer t;  // record overall time
    t.start();
    if (!empty()) {
      psmilu_warning(
          "multilevel precs are not empty, wipe previous results first");
      _precs.clear();
      // also clear the previous buffer
      _prec_work.resize(0);
    }
    // create size references for dropping
    iarray_type row_sizes, col_sizes;
    if (psmilu_verbose(FAC, opts))
      _deferred_factorize_kernel(A, m0, opts, row_sizes, col_sizes, Crout_cout);
    else
      _deferred_factorize_kernel(A, m0, opts, row_sizes, col_sizes,
                                 Crout_cout_dummy);
    t.finish();
    if (psmilu_verbose(INFO, opts)) {
      const size_type Nnz = nnz();
      psmilu_info("\ninput nnz(A)=%zd, nnz(precs)=%zd, ratio=%g", A.nnz(), Nnz,
                  (double)Nnz / A.nnz());
      psmilu_info("\nmultilevel precs building time (overall) is %gs",
                  t.time());
    }
    if (revert_warn) (void)warn_flag(1);
  }

  /// \brief deferred factorization using less memory
  /// \tparam CsType compressed storage input, either \ref CRS or \ref CCS
  /// \param[in] A input matrix
  /// \param[in] m0 leading block size, if it's zero (default), then the routine
  ///               will assume an asymmetric leading block.
  /// \param[in] opts control parameters, using the default values in the paper.
  /// \param[in] check if \a true (default), will perform validity checking
  /// \sa _thin_deferred_factorize_kernel
  ///
  /// This version, differing from deferred_factorize, is that it only builds
  /// a single linked list for each of L and U factors for the current crout
  /// step.
  template <class CsType>
  inline void deferred_factorize_thin(
      const CsType &A, const size_type m0 = 0u,
      const Options &opts = get_default_options(), const bool check = true) {
    static_assert(!(CsType::ONE_BASED ^ ONE_BASED), "inconsistent index base");

    const static internal::StdoutStruct  Crout_cout;
    const static internal::DummyStreamer Crout_cout_dummy;

    // print introduction
    if (psmilu_verbose(INFO, opts)) {
      if (!internal::introduced) {
        psmilu_info(internal::intro, PSMILU_GLOBAL_VERSION,
                    PSMILU_MAJOR_VERSION, PSMILU_MINOR_VERSION, __TIME__,
                    __DATE__);
        internal::introduced = true;
      }
      psmilu_info("Options (control parameters) are:\n");
      psmilu_info(opt_repr(opts).c_str());
    }
    const bool revert_warn = warn_flag();
    if (psmilu_verbose(NONE, opts)) (void)warn_flag(0);

    // check validity of the input system
    if (check) {
      if (psmilu_verbose(INFO, opts))
        psmilu_info("perform input matrix validity checking");
      A.check_validity();
    }

    DefaultTimer t;  // record overall time
    t.start();
    if (!empty()) {
      psmilu_warning(
          "multilevel precs are not empty, wipe previous results first");
      _precs.clear();
      // also clear the previous buffer
      _prec_work.resize(0);
    }
    // create size references for dropping
    iarray_type row_sizes, col_sizes;
    if (psmilu_verbose(FAC, opts))
      _thin_deferred_factorize_kernel(A, m0, opts, row_sizes, col_sizes,
                                      Crout_cout);
    else
      _thin_deferred_factorize_kernel(A, m0, opts, row_sizes, col_sizes,
                                      Crout_cout_dummy);
    t.finish();
    if (psmilu_verbose(INFO, opts)) {
      const size_type Nnz = nnz();
      psmilu_info("\ninput nnz(A)=%zd, nnz(precs)=%zd, ratio=%g", A.nnz(), Nnz,
                  (double)Nnz / A.nnz());
      psmilu_info("\nmultilevel precs building time (overall) is %gs",
                  t.time());
    }
    if (revert_warn) (void)warn_flag(1);
  }

  template <class CsType>
  inline void pivot_factorize(const CsType &A, const size_type m0 = 0u,
                              const Options &opts  = get_default_options(),
                              const bool     check = true) {
    psmilu_error_if(m0 != 0u,
                    "pivot factorization only supports general systems!");
    static_assert(!(CsType::ONE_BASED ^ ONE_BASED), "inconsistent index base");

    const static internal::StdoutStruct  Crout_cout;
    const static internal::DummyStreamer Crout_cout_dummy;

    // print introduction
    if (psmilu_verbose(INFO, opts)) {
      if (!internal::introduced) {
        psmilu_info(internal::intro, PSMILU_GLOBAL_VERSION,
                    PSMILU_MAJOR_VERSION, PSMILU_MINOR_VERSION, __TIME__,
                    __DATE__);
        internal::introduced = true;
      }
      psmilu_info("Options (control parameters) are:\n");
      psmilu_info(opt_repr(opts).c_str());
    }
    const bool revert_warn = warn_flag();
    if (psmilu_verbose(NONE, opts)) (void)warn_flag(0);

    // check validity of the input system
    if (check) {
      if (psmilu_verbose(INFO, opts))
        psmilu_info("perform input matrix validity checking");
      A.check_validity();
    }

    DefaultTimer t;  // record overall time
    t.start();
    if (!empty()) {
      psmilu_warning(
          "multilevel precs are not empty, wipe previous results first");
      _precs.clear();
      // also clear the previous buffer
      _prec_work.resize(0);
    }
    // create size references for dropping
    iarray_type row_sizes, col_sizes;
    if (psmilu_verbose(FAC, opts))
      _pvt_factorize_kernel(A, m0, opts, row_sizes, col_sizes, Crout_cout);
    else
      _pvt_factorize_kernel(A, m0, opts, row_sizes, col_sizes,
                            Crout_cout_dummy);
    t.finish();
    if (psmilu_verbose(INFO, opts)) {
      const size_type Nnz = nnz();
      psmilu_info("\ninput nnz(A)=%zd, nnz(precs)=%zd, ratio=%g", A.nnz(), Nnz,
                  (double)Nnz / A.nnz());
      psmilu_info("\nmultilevel precs building time (overall) is %gs",
                  t.time());
    }
    if (revert_warn) (void)warn_flag(1);
  }

  /// \brief solve \f$\boldsymbol{x}=\boldsymbol{M}^{-1}\boldsymbol{b}\f$
  /// \param[in] b right-hand side vector
  /// \param[out] x solution vector
  /// \sa factorize
  inline void solve(const array_type &b, array_type &x) const {
    psmilu_error_if(empty(), "MILU-Prec is empty!");
    psmilu_error_if(b.size() != x.size(), "unmatched sizes");
    if (_prec_work.empty())
      _prec_work.resize(
          compute_prec_work_space(_precs.cbegin(), _precs.cend()));
    prec_solve(_precs.cbegin(), b, x, _prec_work);
  }

 protected:
  template <class CsType, class CroutStreamer>
  inline void _deferred_factorize_kernel(const CsType &A, const size_type m0,
                                         const Options &      opts,
                                         iarray_type &        row_sizes,
                                         iarray_type &        col_sizes,
                                         const CroutStreamer &Crout_info) {
    psmilu_error_if(A.nrows() != A.ncols(),
                    "Currently only squared systems are supported");
    size_type       m(m0);  // init m
    size_type       N;      // reference size
    const size_type cur_level = levels() + 1;

    // determine the reference size
    if (opts.N >= 0)
      N = opts.N;
    else
      N = cur_level == 1u ? A.nrows() : _precs.front().n;

    // check symmetry
    const bool sym = cur_level == 1u && m > 0u;
    if (!sym) m = A.nrows();  // IMPORTANT! If asymmetric, set m = n

    // instantiate IsSymm here
    const CsType S =
        sym ? iludp_factor_defer<true>(A, m, N, opts, Crout_info, _precs,
                                       row_sizes, col_sizes)
            : iludp_factor_defer<false>(A, m, N, opts, Crout_info, _precs,
                                        row_sizes, col_sizes);

    // check last level
    if (!_precs.back().is_last_level())
      this->_deferred_factorize_kernel(S, 0u, opts, row_sizes, col_sizes,
                                       Crout_info);
  }

  template <class CsType, class CroutStreamer>
  inline void _thin_deferred_factorize_kernel(const CsType &       A,
                                              const size_type      m0,
                                              const Options &      opts,
                                              iarray_type &        row_sizes,
                                              iarray_type &        col_sizes,
                                              const CroutStreamer &Crout_info) {
    psmilu_error_if(A.nrows() != A.ncols(),
                    "Currently only squared systems are supported");
    size_type       m(m0);  // init m
    size_type       N;      // reference size
    const size_type cur_level = levels() + 1;

    // determine the reference size
    if (opts.N >= 0)
      N = opts.N;
    else
      N = cur_level == 1u ? A.nrows() : _precs.front().n;

    // check symmetry
    const bool sym = cur_level == 1u && m > 0u;
    if (!sym) m = A.nrows();  // IMPORTANT! If asymmetric, set m = n

    // instantiate IsSymm here
    const CsType S =
        sym ? iludp_factor_defer_thin<true>(A, m, N, opts, Crout_info, _precs,
                                            row_sizes, col_sizes)
            : iludp_factor_defer_thin<false>(A, m, N, opts, Crout_info, _precs,
                                             row_sizes, col_sizes);

    // check last level
    if (!_precs.back().is_last_level())
      this->_thin_deferred_factorize_kernel(S, 0u, opts, row_sizes, col_sizes,
                                            Crout_info);
  }

  template <class CsType, class CroutStreamer>
  inline void _pvt_factorize_kernel(const CsType & A, const size_type /* m0 */,
                                    const Options &opts, iarray_type &row_sizes,
                                    iarray_type &        col_sizes,
                                    const CroutStreamer &Crout_info) {
    psmilu_error_if(A.nrows() != A.ncols(),
                    "Currently only squared systems are supported");
    size_type       N;  // reference size
    const size_type cur_level = levels() + 1;

    // determine the reference size
    if (opts.N >= 0)
      N = opts.N;
    else
      N = cur_level == 1u ? A.nrows() : _precs.front().n;

    // instantiate IsSymm here
    const CsType S = iludp_factor_pvt<false>(A, A.nrows(), N, opts, Crout_info,
                                             _precs, row_sizes, col_sizes);

    // check last level
    if (!_precs.back().is_last_level())
      this->_pvt_factorize_kernel(S, 0u, opts, row_sizes, col_sizes,
                                  Crout_info);
  }

 protected:
  precs_type         _precs;      ///< multilevel preconditioners
  mutable array_type _prec_work;  ///< preconditioner work space for solving
};

/// \typedef C_PSMILU
/// \tparam ValueType numerical value type, e.g. \a double
/// \tparam IndexType index type, e.g. \a int
/// \sa F_PSMILU
///
/// This is the type wrapper for C index inputs
template <class ValueType, class IndexType>
using C_PSMILU = PSMILU<ValueType, IndexType>;

/// \typedef F_PSMILU
/// \tparam ValueType numerical value type, e.g. \a double
/// \tparam IndexType index type, e.g. \a int
/// \sa C_PSMILU
///
/// This is the type wrapper for Fortran index inputs
template <class ValueType, class IndexType>
using F_PSMILU = PSMILU<ValueType, IndexType, true>;

/// \typedef C_Default_PSMILU
/// \sa F_Default_PSMILU
///
/// This is the type wrapper for default builder for C index, using \a int as
/// index type and \a double as value type.
typedef C_PSMILU<double, int> C_Default_PSMILU;

/// \typedef F_DefaultBuilder
/// \sa C_DefaultBuilder
///
/// This is the type wrapper for default builder for Fortran index, using \a int
/// as index type and \a double as value type.
typedef F_PSMILU<double, int> F_Default_PSMILU;

/*!
 * @}
 */ // group cpp

}  // namespace psmilu

#endif  // _PSMILU_BUILDER_HPP
