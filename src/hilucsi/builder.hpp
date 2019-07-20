//@HEADER
//----------------------------------------------------------------------------
//                Copyright (C) 2019 The HILUCSI AUTHORS
//----------------------------------------------------------------------------
//@HEADER

/// \file hilucsi/builder.hpp
/// \brief Top level user class for building MILU preconditioner
/// \authors Qiao,

#ifndef _HILUCSI_BUILDER_HPP
#define _HILUCSI_BUILDER_HPP

#include <algorithm>
#include <iterator>
#include <numeric>

#include "hilucsi/Options.h"
#include "hilucsi/alg/Prec.hpp"
#include "hilucsi/alg/factor.hpp"
#include "hilucsi/alg/prec_solve.hpp"
#include "hilucsi/utils/mt.hpp"

#include "hilucsi/version.h"

namespace hilucsi {

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace internal {
const static char *intro =
    "\n"
    "=======================================================================\n"
    "|    Hierarchical ILU Crout with Scalability and Inverse Thresholds   |\n"
    "|                                                                     |\n"
    "| HILUCSI is a package for computing multilevel incomplete LU factor- |\n"
    "| ization with nearly linear time complexity. In addition, HILUCSI    |\n"
    "| can also be very robust.                                            |\n"
    "-----------------------------------------------------------------------\n"
    "\n"
    " Package information:\n"
    "\n"
    "\t\tCopyright (C) The HILUCSI AUTHORS\n"
    "\t\tVersion: %d.%d.%d\n"
    "\t\tBuilt on: %s, %s\n"
    "\n"
    "=======================================================================\n";
static bool introduced = false;
}  // namespace internal
#endif  // DOXYGEN_SHOULD_SKIP_THIS

/*!
 * \addtogroup itr
 * @{
 */

/// \class HILUCSI
/// \tparam ValueType numerical value type, e.g. \a double
/// \tparam IndexType index type, e.g. \a int
/// \tparam SSSType default is LU with partial pivoting
///
/// This is top user interface (C++); it is designed as a preconditioner that
/// can be easily plugin other codes. There are two core member functions, 1)
/// \ref factorize a multilevel ILU preconditioner and 2) \ref solve the
/// preconditioner system. For computing preconditioner, the input can be either
/// \ref CCS or \ref CRS, for solving, the input must be \ref Array. However,
/// be aware that both CCS/CRS and Array can be used as external data wrappers,
/// thus one should not worry about input data duplications.
///
/// The following is a simple workflow for Builder.
///
/// \code{.cpp}
///   #include <HILUCSI.hpp>
///   using namespace hilucsi;
///   using builder_t = HILUCSI<double, int>;
///   using crs_t = builder_t::crs_type;
///   int main() {
///     const auto A = wrap_crs<crs_t>(...);
///     builder_t builder;
///     builder.factorize(A);
///     builder.solve(...);
///   }
/// \endcode
template <class ValueType, class IndexType,
          SmallScaleType SSSType = SMALLSCALE_LUP>
class HILUCSI {
 public:
  typedef ValueType                     value_type;    ///< value type
  typedef Array<value_type>             array_type;    ///< array type
  typedef IndexType                     index_type;    ///< index type
  typedef Array<index_type>             iarray_type;   ///< index array
  typedef CRS<value_type, index_type>   crs_type;      ///< crs type
  typedef typename crs_type::other_type ccs_type;      ///< ccs type
  constexpr static SmallScaleType sss_type = SSSType;  ///< small scale type
  typedef Precs<value_type, index_type, sss_type> precs_type;
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

  /// \brief compute the nnz in \a E and \a F components
  inline size_type nnz_EF() const {
    if (empty()) return 0u;
    size_type n(0);
    for (auto itr = _precs.cbegin(); itr != _precs.cend(); ++itr)
      n += itr->nnz_EF();
    return n;
  }

  /// \brief compute the nnz in \a L, \a D, and \a U
  inline size_type nnz_LDU() const { return nnz() - nnz_EF(); }

  /// \brief get number of rows
  inline size_type nrows() const { return _nrows; }

  /// \brief get number of columns
  inline size_type ncols() const { return _ncols; }

  /// \brief get the stats
  /// \param[in] entry statistic entry
  ///
  /// Currently, the following stats are available:
  ///   - 0: total deferrals
  ///   - 1: dynamic deferrals
  ///   - 2: dynamic deferrals due to bad diagonals
  ///   - 3: dynamic deferrals due to bad conditioning
  ///   - 4: total droppings
  ///   - 5: droppings due to space-controling
  /// Notice that entry \a 0 subtract entry \a 1 is the static deferrals, which
  /// are due to zero diagonals that occurs when applying symmetric
  /// preprocessing.
  inline size_type stats(const size_type entry) const {
    static const char *help =
        "\t0: total deferrals\n"
        "\t1: dynamic deferrals\n"
        "\t2: dynamic deferrals due to bad diagonals\n"
        "\t3: dynamic deferrals due to bad conditioning\n"
        "\t4: total droppings\n"
        "\t5: droppings due to space-controling\n";
    hilucsi_error_if(empty(), "no stats available for empty structure");
    if (entry > 5u) {
      // NOTE, we cannot pass the help information as variadic arguments due
      // to the internal buffer may overflow thus causing segfault.
      std::stringstream ss;
      ss << entry << " exceeds maximum statistics entry (5)\nhelp:\n" << help;
      hilucsi_error(ss.str().c_str());
    }
    return _stats[entry];
  }

  /// \brief get constant reference to a specific level
  /// \note This function takes linear time complexity
  inline const prec_type &prec(const size_type level) const {
    hilucsi_error_if(level >= levels(),
                     "%zd exceeds the total level number %zd", level, levels());
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
  /// \sa solve
  template <class CsType>
  inline void factorize(const CsType &A, const size_type m0 = 0u,
                        const Options &opts = get_default_options()) {
    const static internal::StdoutStruct  Crout_cout;
    const static internal::DummyStreamer Crout_cout_dummy;

    // print introduction
    if (hilucsi_verbose(INFO, opts)) {
      if (!internal::introduced) {
        hilucsi_info(internal::intro, HILUCSI_GLOBAL_VERSION,
                     HILUCSI_MAJOR_VERSION, HILUCSI_MINOR_VERSION, __TIME__,
                     __DATE__);
        internal::introduced = true;
      }
      hilucsi_info("Options (control parameters) are:\n");
      hilucsi_info(opt_repr(opts).c_str());
    }
    const bool revert_warn = warn_flag();
    if (hilucsi_verbose(NONE, opts)) (void)warn_flag(0);

    // check validity of the input system
    if (opts.check) {
      if (hilucsi_verbose(INFO, opts))
        hilucsi_info("perform input matrix validity checking");
      A.check_validity();
    }

    _nrows = A.nrows();
    _ncols = A.ncols();

    DefaultTimer t;  // record overall time
    t.start();
    if (!empty()) {
      hilucsi_warning(
          "multilevel precs are not empty, wipe previous results first");
      _precs.clear();
      // also clear the previous buffer
      _prec_work.resize(0);
    }
    const int schur_threads = mt::get_nthreads(opts.threads);
    // initialize statistics
    for (size_type i(0); i < sizeof(_stats) / sizeof(size_type); ++i)
      _stats[i] = 0;
    // create size references for dropping
    iarray_type row_sizes, col_sizes;
    if (hilucsi_verbose(FAC, opts))
      _factorize_kernel(A, m0, opts, row_sizes, col_sizes, Crout_cout,
                        schur_threads);
    else
      _factorize_kernel(A, m0, opts, row_sizes, col_sizes, Crout_cout_dummy,
                        schur_threads);
    t.finish();
    if (hilucsi_verbose(INFO, opts)) {
      const size_type Nnz = nnz();
      hilucsi_info("\ninput nnz(A)=%zd, nnz(precs)=%zd, ratio=%g", A.nnz(), Nnz,
                   (double)Nnz / A.nnz());
      hilucsi_info("\nmultilevel precs building time (overall) is %gs",
                   t.time());
    }
    if (revert_warn) (void)warn_flag(1);
  }

  /// \brief solve \f$\boldsymbol{x}=\boldsymbol{M}^{-1}\boldsymbol{b}\f$
  /// \param[in] b right-hand side vector
  /// \param[out] x solution vector
  /// \sa factorize
  inline void solve(const array_type &b, array_type &x) const {
    hilucsi_error_if(empty(), "MILU-Prec is empty!");
    hilucsi_error_if(b.size() != x.size(), "unmatched sizes");
    if (_prec_work.empty())
      _prec_work.resize(
          compute_prec_work_space(_precs.cbegin(), _precs.cend()));
    prec_solve(_precs.cbegin(), b, x, _prec_work);
  }

 protected:
  template <class CsType, class CroutStreamer>
  inline void _factorize_kernel(const CsType &A, const size_type m0,
                                const Options &opts, iarray_type &row_sizes,
                                iarray_type &        col_sizes,
                                const CroutStreamer &Crout_info,
                                const int            schur_threads = 1) {
    hilucsi_error_if(A.nrows() != A.ncols(),
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
    CsType S =
        sym ? level_factorize<true>(A, m, N, opts, Crout_info, _precs,
                                    row_sizes, col_sizes, _stats, schur_threads)
            : level_factorize<false>(A, m, N, opts, Crout_info, _precs,
                                     row_sizes, col_sizes, _stats,
                                     schur_threads);

    // check last level
    if (!_precs.back().is_last_level())
      this->_factorize_kernel(S, 0u, opts, row_sizes, col_sizes, Crout_info,
                              schur_threads);
  }

 protected:
  precs_type         _precs;      ///< multilevel preconditioners
  mutable array_type _prec_work;  ///< preconditioner work space for solving
  size_type          _stats[6];   ///< statistics
  size_type          _nrows;      ///< number of rows from user input
  size_type          _ncols;      ///< number of columns from user input
};

/// \typedef DefaultHILUCSI
/// \brief default HILUCSI with \a double as value type and \a int as index
typedef HILUCSI<double, int> DefaultHILUCSI;

/*!
 * @}
 */ // group itr

}  // namespace hilucsi

#endif  // _HILUCSI_BUILDER_HPP