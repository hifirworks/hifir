///////////////////////////////////////////////////////////////////////////////
//                  This file is part of the HIFIR library                   //
///////////////////////////////////////////////////////////////////////////////

/*!
 * \file hif/builder.hpp
 * \brief Top level user class for building MILU preconditioner
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

#ifndef _HIF_BUILDER_HPP
#define _HIF_BUILDER_HPP

#include <algorithm>
#include <iterator>
#include <numeric>
#include <type_traits>

#include "hif/NspFilter.hpp"
#include "hif/Options.h"
#include "hif/alg/IterRefine.hpp"
#include "hif/alg/Prec.hpp"
#include "hif/alg/factor.hpp"
#include "hif/alg/pivot_factor.hpp"
#include "hif/alg/prec_prod.hpp"
#include "hif/alg/prec_solve.hpp"
#include "hif/alg/symm_factor.hpp"
#include "hif/utils/common.hpp"
#include "hif/utils/mt.hpp"

#include "hif/version.h"

namespace hif {

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace internal {
const static char *intro =
    "\n"
    "=======================================================================\n"
    "|           Hybrid (Hierarchical) Incomplete Factorizations           |\n"
    "|                                                                     |\n"
    "| HIF is a package for computing hybrid (hierarchical) incomplete fa- |\n"
    "| ctorizations with nearly linear time complexity.                    |\n"
    "-----------------------------------------------------------------------\n"
    "\n"
    " > Package information:\n"
    "\n"
    "\t* Copyright (C) NumGeom Group at Stony Brook University\n"
    "\t* Version: %d.%d.%d\n"
    "\n"
    "=======================================================================\n";
static bool introduced = false;
}  // namespace internal
#endif  // DOXYGEN_SHOULD_SKIP_THIS

/*!
 * \addtogroup itr
 * @{
 */

/// \class HIF
/// \tparam ValueType numerical value type, e.g. \a double
/// \tparam IndexType index type, e.g. \a int
/// \tparam UserDenseFactor Potential user customized dense factor
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
///   #include <HIF.hpp>
///   using namespace hif;
///   using builder_t = HIF<double, int>;
///   using crs_t = builder_t::crs_type;
///   int main() {
///     const auto A = wrap_crs<crs_t>(...);
///     builder_t builder;
///     builder.factorize(A);
///     builder.solve(...);
///   }
/// \endcode
template <class ValueType, class IndexType,
          template <class> class UserDenseFactor = DefaultDenseSolver>
class HIF {
 public:
  typedef ValueType                     value_type;   ///< value type
  typedef Array<value_type>             array_type;   ///< array type
  typedef IndexType                     index_type;   ///< index type
  typedef Array<index_type>             iarray_type;  ///< index array
  typedef CRS<value_type, index_type>   crs_type;     ///< crs type
  typedef typename crs_type::other_type ccs_type;     ///< ccs type
  // constexpr static SmallScaleType sss_type = SSSType;  ///< small scale type
  typedef Precs<value_type, index_type, UserDenseFactor> precs_type;
  ///< multilevel preconditioner type
  typedef typename precs_type::value_type prec_type;  ///< single level prec
  typedef typename prec_type::size_type   size_type;  ///< size type
#ifdef HIF_HIGH_PRECISION_SOLVE
  typedef typename ValueTypeMixedTrait<value_type>::boost_type boost_value_type;
#else
  typedef value_type boost_value_type;
#endif
  ///< high-precision value type
  typedef Array<boost_value_type> work_array_type;  ///< work array type

  /// \brief check if or not we can export data
  constexpr bool can_export() const {
#ifdef HIF_ENABLE_MUMPS
    return false;
#else
    return true;
#endif  // HIF_ENABLE_MUMPS
  }

  /// \brief check empty or not
  inline bool empty() const { return _precs.empty(); }

  /// \brief check number of levels
  /// \note This function takes \f$\mathcal{O}(1)\f$ since C++11
  inline size_type levels() const {
    const size_type lvls = _precs.size();
    if (lvls)
      return lvls + !_precs.back().dense_solver.empty() +
             !_precs.back().symm_dense_solver.empty() +
             !_precs.back().sparse_solver.empty();
    return 0;
  }

  // utilities

  /// \brief get constant reference to preconditioners
  inline const precs_type &precs() const { return _precs; }

  /// \brief get normal reference to preconditioners
  inline precs_type &precs() { return _precs; }

  /// \brief compute the overall nnz of the multilevel preconditioners
  inline size_type nnz() const {
    if (empty()) return 0u;
    size_type n(0);
    for (auto itr = _precs.cbegin(); itr != _precs.cend(); ++itr)
      n += itr->nnz();
    return n;
  }

  /// \brief get the numerical rank
  inline size_type rank() const {
    if (empty()) return 0u;
    const auto &prec_last = precs().back();
    return std::min(nrows(), ncols()) -
           ((prec_last.n - prec_last.m) - prec_last.last_rank());
  }

  /// \brief get the numerical rank for last level (dense Schur complement)
  inline size_type schur_rank() const {
    if (empty()) return 0u;
    return precs().back().last_rank();
  }

  /// \brief get the Schur complement size
  inline size_type schur_size() const {
    if (empty()) return 0u;
    return precs().back().n - precs().back().m;
  }

  /// \brief compute the nnz in \a E and \a F components
  inline size_type nnz_ef() const {
    if (empty()) return 0u;
    size_type n(0);
    for (auto itr = _precs.cbegin(); itr != _precs.cend(); ++itr)
      n += itr->nnz_ef();
    return n;
  }

  /// \brief compute the nnz in \a L, \a D, and \a U
  inline size_type nnz_ldu() const { return nnz() - nnz_ef(); }

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
    hif_error_if(empty(), "no stats available for empty structure");
    if (entry > 5u) {
      // NOTE, we cannot pass the help information as variadic arguments due
      // to the internal buffer may overflow thus causing segfault.
      std::stringstream ss;
      ss << entry << " exceeds maximum statistics entry (5)\nhelp:\n" << help;
      hif_error(ss.str().c_str());
    }
    return _stats[entry];
  }

  /// \brief get constant reference to a specific level
  /// \note This function takes linear time complexity
  inline const prec_type &prec(const size_type level) const {
    hif_error_if(level >= levels(), "%zd exceeds the total level number %zd",
                 level, levels());
    auto itr = _precs.cbegin();
    std::advance(itr, level);
    return *itr;
  }

  /// \brief clear internal storage
  inline void clear() {
    _precs.clear();
    // _prec_work.resize(0);
    work_array_type().swap(_prec_work);
    nsp.reset();
    nsp_tran.reset();
    _ir.clear();
  }

  /// \brief factorize the MILU preconditioner
  /// \tparam CsType compressed storage input, either \ref CRS or \ref CCS
  /// \param[in] A input matrix
  /// \param[in] params control parameters, using the default values
  /// \param[in] m0 leading block size, if it's zero (default), then the routine
  ///               will assume an asymmetric leading block.
  /// \sa solve
  template <class CsType>
  inline void factorize(const CsType &A, const Params &params = DEFAULT_PARAMS,
                        const size_type m0 = 0u) {
    const static internal::StdoutStruct  Crout_cout;
    const static internal::DummyStreamer Crout_cout_dummy;
    using cs_type =
        typename std::conditional<CsType::ROW_MAJOR, crs_type, ccs_type>::type;

    static_assert(std::is_same<index_type, typename CsType::index_type>::value,
                  "inconsistent index types");

    // print introduction
    if (hif_verbose(INFO, params)) {
      if (!internal::introduced) {
        hif_info(internal::intro, HIF_GLOBAL_VERSION, HIF_MAJOR_VERSION,
                 HIF_MINOR_VERSION);
        internal::introduced = true;
      }
      hif_info("Params (control parameters) are:\n");
      hif_info(opt_repr(params).c_str());
    }
    const bool revert_warn = warn_flag();
    if (hif_verbose(NONE, params))
      (void)warn_flag(0);
    else
      warn_flag(1);

    _nrows = A.nrows();
    _ncols = A.ncols();

    DefaultTimer t;  // record overall time
    t.start();
    if (!empty()) {
      if (hif_verbose(INFO, params))
        hif_info("multilevel precs are not empty, wipe previous results first");
      _precs.clear();
      // also clear the previous buffer
      _prec_work.resize(0);
    }
    const int schur_threads = mt::get_nthreads(params.threads);
    // initialize statistics
    for (size_type i(0); i < sizeof(_stats) / sizeof(size_type); ++i)
      _stats[i] = 0;

    cs_type AA;
    AA.resize(A.nrows(), A.ncols());
    // check index base
    if (A.nrows()) {
      if (A.ind_start()[0] == index_type(0)) {
        // 0-based, shallow copy
        AA.ind_start() = A.ind_start();
        AA.inds()      = A.inds();
      } else if (A.ind_start()[0] == index_type(1)) {
        // 1-based
        if (hif_verbose(INFO, params))
          hif_info("converting to 0-based compressed matrix...");
        AA.ind_start().resize(A.ind_start().size());
        for (size_type i(0); i < A.ind_start().size(); ++i)
          AA.ind_start()[i] = A.ind_start()[i] - 1;
        AA.inds().resize(A.inds().size());
        for (size_type i(0); i < A.inds().size(); ++i)
          AA.inds()[i] = A.inds()[i] - 1;
      } else
        hif_error("only {0,1}-based compressed sparse matrices are supported!");
    }
    // check value type consistency
    if (std::is_same<value_type, typename CsType::value_type>::value)
      AA.vals() = array_type(A.nnz(), (value_type *)A.vals().data(), true);
    else {
      if (hif_verbose(INFO, params))
        hif_info("converting value type precision...");
      // shallow integer arrays
      AA.resize(A.nrows(), A.ncols());
      AA.vals().resize(A.nnz());
      hif_error_if(AA.vals().status() == DATA_UNDEF,
                   "memory allocation failed");
      std::copy(A.vals().cbegin(), A.vals().cend(), AA.vals().begin());
    }
    // check validity of the input system
    if (params.check) {
      if (hif_verbose(INFO, params))
        hif_info("perform input matrix validity checking");
      AA.check_validity();
    }
    // create size references for dropping
    iarray_type row_sizes, col_sizes;
    if (hif_verbose(FAC, params))
      _factorize_kernel(AA, m0, params, row_sizes, col_sizes, Crout_cout,
                        schur_threads);
    else
      _factorize_kernel(AA, m0, params, row_sizes, col_sizes, Crout_cout_dummy,
                        schur_threads);
    t.finish();
    if (hif_verbose(INFO, params)) {
      const size_type Nnz = nnz();
      hif_info("\ninput nnz(A)=%zd, nnz(precs)=%zd, ratio=%g", A.nnz(), Nnz,
               (double)Nnz / A.nnz());
      hif_info("\nmultilevel precs building time (overall) is %gs", t.time());
    }
    if (revert_warn) (void)warn_flag(1);
  }

  /// \brief factorize the MILU preconditioner with generic interface
  /// \tparam IsCrs if \a true, then the input compressed structure will be
  ///               assumed to be \ref CRS format
  /// \tparam IndexType_ integer data type, e.g., \a int
  /// \tparam ValueType_ value data type, e.g., \a double
  /// \param[in] n size of system
  /// \param[in] indptr index starting position array, must be length of \a n+1
  /// \param[in] indices index value array, must be length of \a indptr[n]
  /// \param[in] vals value array, same length as that of \a indices
  /// \param[in] params control parameters, using the default values.
  /// \param[in] m0 leading block size, if it's zero (default), then the routine
  ///               will assume an asymmetric leading block.
  ///
  /// This interface differs from the above one is that it takes plain-old-data
  /// types as input thus flexible. Notice that the integer and floating data
  /// types don't need to align with \ref index_type and \ref value_type, which
  /// aims for mixed-precision computation.
  template <bool IsCrs, class IndexType_, class ValueType_>
  inline void factorize(const size_type n, const IndexType_ *indptr,
                        const IndexType_ *indices, const ValueType_ *vals,
                        const Params &  params = DEFAULT_PARAMS,
                        const size_type m0     = 0u) {
    using foreign_crs_type = CRS<ValueType_, IndexType_>;
    using foreign_ccs_type = typename foreign_crs_type::other_type;
    using foreign_cs_type  = typename std::conditional<IsCrs, foreign_crs_type,
                                                      foreign_ccs_type>::type;
    const foreign_cs_type A(n, (IndexType_ *)indptr, (IndexType_ *)indices,
                            (ValueType_ *)vals, true);
    factorize(A, params, m0);
  }

  /// \brief optimization a priori
  /// \param[in] tag optimization tag
  inline void optimize(const int tag = 0) {
    for (auto &prec : _precs) prec.optimize(tag);
  }

  /// \brief solve \f$\mathbf{x}=\mathbf{M}^{-1}\mathbf{b}\f$
  /// \tparam RhsType right-hand side type
  /// \tparam SolType solution type
  /// \param[in] b right-hand side vector
  /// \param[out] x solution vector
  /// \param[in] trans (optional) transpose/Hermitian flag (default is false)
  /// \param[in] r (optional) dimension for back solve for last level
  ///              efault is its numerical rank
  template <class RhsType, class SolType>
  inline void solve(const RhsType &b, SolType &x, const bool trans = false,
                    const size_type r = 0u) const {
    hif_error_if(empty(), "MILU-Prec is empty!");
    hif_error_if(b.size() != x.size(), "unmatched sizes");
    if (_prec_work.empty())
      _prec_work.resize(
          compute_prec_work_space(_precs.cbegin(), _precs.cend()));
    !trans ? prec_solve(_precs.cbegin(), b, r, x, _prec_work)
           : prec_solve_tran(_precs.cbegin(), b, r, x, _prec_work);
    if (!trans && nsp)
      nsp->filter(x.data(), x.size());  // filter null space
    else if (trans && nsp_tran)
      nsp_tran->filter(x.data(), x.size());  // filter (left) null space
  }

  /// \brief solve \f$\mathbf{X}=\mathbf{M}^{-1}\mathbf{B}\f$ (multiple RHS)
  /// \tparam RhsValueType right-hand-side (RHS) value type
  /// \tparam SolValueType solution value type
  /// \tparam Nrhs number of RHS
  /// \param[in] b right-hand side vector
  /// \param[out] x solution vector
  /// \param[in] r (optional) dimension for back solve for last level
  ///              default is its numerical rank
  template <class RhsValueType, class SolValueType, size_type Nrhs>
  inline void solve_mrhs(const Array<std::array<RhsValueType, Nrhs>> &b,
                         Array<std::array<SolValueType, Nrhs>> &      x,
                         const size_type r = 0u) const {
    hif_error_if(empty(), "MILU-Prec is empty!");
    hif_error_if(b.size() != x.size(), "unmatched sizes");
    hif_error_if(!_precs.back().sparse_solver.empty(),
                 "multiple RHS solve does not support sparse last level");
    const auto nw = compute_prec_work_space(_precs.cbegin(), _precs.cend());
    if (_prec_work.empty()) _prec_work.resize(nw * Nrhs);
    Array<std::array<boost_value_type, Nrhs>> w(
        nw, (std::array<boost_value_type, Nrhs> *)_prec_work.data(), true);
    prec_solve_mrhs(_precs.cbegin(), b, r, x, w);
    hif_warning_if(nsp, "multiple RHS does not support null space filter.");
  }

  /// \brief solve with iterative refinement
  /// \tparam Matrix matrix type
  /// \tparam RhsType right-hand side type
  /// \tparam SolType solution type
  /// \param[in] A matrix operator
  /// \param[in] b right-hand side vector
  /// \param[in] N iteration count
  /// \param[out] x solution vector
  /// \param[in] trans (optional) transpose/Hermitian flag, default is false
  /// \param[in] r (optional) dimension for back solve for last level
  ///            default is its numerical rank
  template <class Matrix, class RhsType, class SolType>
  void hifir(const Matrix &A, const RhsType &b, const size_type N, SolType &x,
             const bool      trans = false,
             const size_type r     = static_cast<size_type>(-1)) const {
    // NOTE, do not assume A shares interface of CRS, as it can be
    // user callback
    _ir.iter_refine(*this, A, b, N, x, r, trans);
  }

  NspFilterPtr nsp;       ///< null space filter
  NspFilterPtr nsp_tran;  ///< transpose null space filter (left null space)

  /// \brief compute \f$\mathbf{y}=\mathbf{Mx}\f$
  /// \tparam RhsType right-hand side type
  /// \tparam SolType solution type
  /// \param[in] x right-hand side vector
  /// \param[out] y solution vector
  /// \param[in] trans (optional) transpose/Hermitian flag (default is false)
  /// \param[in] r (optional) dimension for back solve for last level
  ///              default is its numerical rank
  template <class RhsType, class SolType>
  inline void mmultiply(const RhsType &x, SolType &y, const bool trans = false,
                        const size_type r = 0u) const {
    hif_error_if(empty(), "MILU-Prec is empty!");
    hif_error_if(y.size() != x.size(), "unmatched sizes");
    if (_prec_work.empty())
      _prec_work.resize(
          compute_prec_work_space(_precs.cbegin(), _precs.cend()));
    !trans ? prec_prod(_precs.cbegin(), x, r, y, _prec_work)
           : prec_prod_tran(_precs.cbegin(), x, r, y, _prec_work);
  }

 protected:
  template <class CsType, class CroutStreamer>
  inline void _factorize_kernel(const CsType &A, const size_type m0,
                                const Params &opts, iarray_type &row_sizes,
                                iarray_type &        col_sizes,
                                const CroutStreamer &Crout_info,
                                const int            schur_threads = 1,
                                const bool           must_pivot    = false) {
    hif_error_if(A.nrows() != A.ncols(),
                 "Currently only squared systems are supported");
    size_type       m(m0);  // init m
    size_type       N;      // reference size
    const size_type cur_level = levels() + 1;

    // determine the reference size
    if (opts.N >= 0)
      N = opts.N;
    else
      N = cur_level == 1u ? A.nrows() : _precs.front().n;

    // check partial symmetry
    const bool sym = cur_level == 1u && m > 0u;
    if (!sym) m = A.nrows();  // IMPORTANT! If asymmetric, set m = n

    CsType S;
    int    pivot_flag = 0;
    if (opts.is_symm) {
      S = symm_level_factorize(A, m, N, opts, Crout_info, _precs, row_sizes,
                               _stats, schur_threads);
    } else {
      if (opts.pivot == PIVOTING_OFF)
        // instantiate IsSymm here
        S = sym ? level_factorize<true>(A, m, N, opts, Crout_info, _precs,
                                        row_sizes, col_sizes, _stats,
                                        schur_threads)
                : level_factorize<false>(A, m, N, opts, Crout_info, _precs,
                                         row_sizes, col_sizes, _stats,
                                         schur_threads);
      else if (opts.pivot == PIVOTING_ON)
        S = pivot_level_factorize(A, m, N, opts, Crout_info, _precs, row_sizes,
                                  col_sizes, _stats, schur_threads);
      else {
        if (must_pivot) {
          S          = pivot_level_factorize(A, m, N, opts, Crout_info, _precs,
                                    row_sizes, col_sizes, _stats,
                                    schur_threads);
          pivot_flag = 1;  // Always recursively do pivoting
        } else
          S = sym ? level_factorize<true>(A, m, N, opts, Crout_info, _precs,
                                          row_sizes, col_sizes, _stats,
                                          schur_threads, &pivot_flag)
                  : level_factorize<false>(A, m, N, opts, Crout_info, _precs,
                                           row_sizes, col_sizes, _stats,
                                           schur_threads, &pivot_flag);
      }
    }

    // check last level
    if (!_precs.back().is_last_level())
      this->_factorize_kernel(S, 0u, opts, row_sizes, col_sizes, Crout_info,
                              schur_threads, pivot_flag != 0);
  }

 protected:
  precs_type              _precs;  ///< multilevel preconditioners
  mutable work_array_type _prec_work;
  ///< preconditioner work space for solving
  size_type                    _stats[6];  ///< statistics
  size_type                    _nrows;     ///< number of rows from user input
  size_type                    _ncols;  ///< number of columns from user input
  IterRefine<boost_value_type> _ir;     ///< high-precision iterative refinement
};

/// \typedef DefaultHIF
/// \brief default HIF with \a double as value type and \a int as index
typedef HIF<double, int> DefaultHIF;

/*!
 * @}
 */ // group itr

}  // namespace hif

#endif  // _HIF_BUILDER_HPP
