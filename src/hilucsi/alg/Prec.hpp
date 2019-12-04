///////////////////////////////////////////////////////////////////////////////
//                This file is part of HILUCSI project                       //
///////////////////////////////////////////////////////////////////////////////

/*!
 * \file hilucsi/alg/Prec.hpp
 * \brief Multilevel preconditioner interface
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

#ifndef _HILUCSI_ALG_PREC_HPP
#define _HILUCSI_ALG_PREC_HPP

#include <list>
#include <utility>

#include "hilucsi/ds/CompressedStorage.hpp"
#include "hilucsi/ds/IntervalCompressedStorage.hpp"
#include "hilucsi/small_scale/solver.hpp"
#ifdef HILUCSI_ENABLE_MUMPS
#  include "hilucsi/sparse_direct/mumps.hpp"
#endif  // HILUCSI_ENABLE_MUMPS
#if HILUCSI_HAS_SPARSE_MKL
#  include "hilucsi/arch/mkl_trsv.hpp"
#endif

namespace hilucsi {

#ifndef DOXYGEN_SHOULD_SKIP_THIS

namespace internal {
struct DummySparseSolver {
  inline constexpr bool        empty() const { return true; }
  inline constexpr std::size_t nnz() const { return 0; }
  inline constexpr int         info() const { return 0; }
  inline static const char *   backend() { return "DummySparse"; }
  template <class ArrayType>
  inline void solve(ArrayType &) {}
};
}  // namespace internal

#endif  // DOXYGEN_SHOULD_SKIP_THIS

/// \class Prec
/// \brief A single level preconditioner
/// \tparam ValueType value data type, e.g. \a double
/// \tparam IndexType index data type, e.g. \a int
/// \tparam IntervalBased default is true, using interval based
/// \ingroup slv
template <class ValueType, class IndexType, bool IntervalBased = true>
struct Prec {
  typedef ValueType                     value_type;  ///< value type
  typedef IndexType                     index_type;  ///< index type
  typedef CRS<value_type, index_type>   crs_type;    ///< crs type
  typedef CCS<value_type, index_type>   ccs_type;    ///< ccs type
  typedef Array<index_type>             perm_type;   ///< permutation
  typedef typename ccs_type::size_type  size_type;   ///< size
  typedef typename ccs_type::array_type array_type;  ///< array
  typedef ccs_type                      mat_type;    ///< interface type
#ifdef HILUCSI_ENABLE_MUMPS
  using sparse_direct_type = MUMPS<ValueType>;
#else
  using sparse_direct_type     = internal::DummySparseSolver;
#endif                                                   // HILUCSI_ENABLE_MUMPS
  static constexpr char EMPTY_PREC     = '\0';           ///< empty prec
  static constexpr bool INTERVAL_BASED = IntervalBased;  ///< interval
  using data_mat_type                  = typename std::conditional<
      INTERVAL_BASED, typename using_interval_from_classical<mat_type>::type,
      mat_type>::type;  ///< data matrix type
#if HILUCSI_HAS_SPARSE_MKL
  using tri_mat_type                  = crs_type;
  using tri_mat_interface_type        = tri_mat_type;
  constexpr static bool OPTIMIZE_FLAG = true;
#else
  using tri_mat_type           = data_mat_type;  ///< triangular matrix type
  using tri_mat_interface_type = mat_type;       ///< interface type
  constexpr static bool OPTIMIZE_FLAG = false;   ///< optimization flag
#endif

 private:
  typedef SmallScaleSolverTrait<SMALLSCALE_LUP> _sss_trait;
  ///< small scale trait
 public:
  typedef typename _sss_trait::template solver_type<value_type> sss_solver_type;
  ///< small scaled solver type

  /// \brief default constructor
  Prec() = default;

  /// \brief empty copy constructor
  Prec(const Prec &) {}

  /// \brief rvalue reference constructor
  /// \param[in] mm leading size
  /// \param[in] nn system size
  /// \param[in] L_b lower part
  /// \param[in] d_b diagonal
  /// \param[in] U_b upper part
  /// \param[in] e E part
  /// \param[in] f F part
  /// \param[in] S row scaling
  /// \param[in] T column scaling
  /// \param[in] P row permutation
  /// \param[in] Q_inv inverse column permutation
  /// \note This allows us to use emplace back in STL efficiently
  Prec(size_type mm, size_type nn, tri_mat_interface_type &&L_b,
       array_type &&d_b, tri_mat_interface_type &&U_b, mat_type &&e,
       mat_type &&f, array_type &&S, array_type &&T, perm_type &&P,
       perm_type &&Q_inv)
      : m(mm),
        n(nn),
        L_B(std::move(L_b)),
        d_B(std::move(d_b)),
        U_B(std::move(U_b)),
        E(std::move(e)),
        F(std::move(f)),
        s(std::move(S)),
        t(std::move(T)),
        p(std::move(P)),
        q_inv(std::move(Q_inv)) {}

  /// \brief get number of nonzeros
  inline size_type nnz() const {
    size_type nz = m ? L_B.nnz() + U_B.nnz() + m : 0;
    if (n - m) nz += E.nnz() + F.nnz();
    if (!dense_solver.empty()) return nz + (n - m) * (n - m);
    if (!sparse_solver.empty()) return nz + sparse_solver.nnz();
    return nz;
  }

  /// \brief get the number of nonzeros in \a E and \a F components
  inline size_type nnz_EF() const {
    if (n - m) return E.nnz() + F.nnz();
    return 0;
  }

  /// \brief Using SFINAE to report interval-based information
  template <typename T = void>
  inline typename std::enable_if<INTERVAL_BASED, T>::type report_status_ef()
      const {
    static const auto report_kernel = [](const data_mat_type &mat,
                                         const char *         name) {
      if (!mat.converted()) {
        hilucsi_info(
            "%s was not able to be converted to interval, too small average "
            "interval (<2)",
            name);
      } else {
        hilucsi_info(
            "%s was converted to interval!\n"
            "\tstorage ratio (interval:classical) = %.3f%%\n"
            "\taverage interval length = %.4g",
            name, 100.0 * mat.storage_cost_ratio(),
            (double)mat.nnz() / mat.nitrvs());
      }
    };

    report_kernel(E, "E");
    report_kernel(F, "F");
  }

  template <typename T = void>
  inline typename std::enable_if<!INTERVAL_BASED, T>::type report_status_ef()
      const {
    // do nothing
  }

  /// \brief report LU status with SFINAE
  template <typename T = void>
  inline typename std::enable_if<is_interval_cs<tri_mat_type>::value, T>::type
  report_status_lu() const {
    static const auto report_kernel = [](const data_mat_type &mat,
                                         const char *         name) {
      if (!mat.converted()) {
        hilucsi_info(
            "%s was not able to be converted to interval, too small average "
            "interval (<2)",
            name);
      } else {
        hilucsi_info(
            "%s was converted to interval!\n"
            "\tstorage ratio (interval:classical) = %.3f%%\n"
            "\taverage interval length = %.4g",
            name, 100.0 * mat.storage_cost_ratio(),
            (double)mat.nnz() / mat.nitrvs());
      }
    };

    report_kernel(L_B, "L_B");
    report_kernel(U_B, "U_B");
  }

  template <typename T = void>
  inline typename std::enable_if<!is_interval_cs<tri_mat_type>::value, T>::type
  report_status_lu() const {}

  /// \brief enable explicitly calling move
  /// \param[in,out] L_b lower part
  /// \param[in,out] d_b diagonal
  /// \param[in,out] U_b upper part
  /// \param[in,out] e E part
  /// \param[in,out] f F part
  /// \param[in,out] S row scaling
  /// \param[in,out] T column scaling
  /// \param[in,out] P row permutation
  /// \param[in,out] Q_inv inverse column permutation
  /// \note Sizes are not included, assign them explicitly.
  ///
  /// we pass lvalue reference, but will explicitly destroy all input arguments
  /// on output. This allows one to avoid writing a ton of \a std::move while
  /// calling this routine.
  ///
  /// \warning Everything on output is destroyed, as the routine name says.
  inline void move_destroy(tri_mat_interface_type &L_b, array_type &d_b,
                           tri_mat_interface_type &U_b, mat_type &e,
                           mat_type &f, array_type &S, array_type &T,
                           perm_type &P, perm_type &Q_inv) {
    L_B   = std::move(L_b);
    d_B   = std::move(d_b);
    U_B   = std::move(U_b);
    E     = std::move(e);
    F     = std::move(f);
    s     = std::move(S);
    t     = std::move(T);
    p     = std::move(P);
    q_inv = std::move(Q_inv);
  }

  /// \brief a priori optimization
  /// \param[in] expected_calls number of calls
  inline void optimize(const size_type expected_calls = -1) {
#if HILUCSI_HAS_SPARSE_MKL
    mkl_L.setup(L_B.row_start(), L_B.col_ind(), L_B.vals());
    mkl_U.setup(U_B.row_start(), U_B.col_ind(), U_B.vals());
    const MKL_INT exp_calls = expected_calls == (size_type)-1
                                  ? std::numeric_limits<MKL_INT>::max()
                                  : expected_calls;
    mkl_L.template optimize<false>(exp_calls);
    mkl_U.template optimize<true>(exp_calls);
#else
    (void)expected_calls;
#endif
  }

  /// \brief check if this a last level preconditioner
  ///
  /// The idea is first check if the \ref dense_solver is empty or not; be aware
  /// that there is another situation that we should treat the preconditioner
  /// is last level---if \ref m is equal to \ref n.
  ///
  /// \note Currently, we test m == n, which is fine for squared systems.
  inline bool is_last_level() const {
    return !sparse_solver.empty() || !dense_solver.empty() || m == n;
  }

  size_type                  m;      ///< leading block size
  size_type                  n;      ///< system size
  tri_mat_type               L_B;    ///< lower part of leading block
  array_type                 d_B;    ///< diagonal block of leading block
  tri_mat_type               U_B;    ///< upper part of leading block
  data_mat_type              E;      ///< scaled and permutated E part
  data_mat_type              F;      ///< scaled and permutated F part
  array_type                 s;      ///< row scaling vector
  array_type                 t;      ///< column scaling vector
  perm_type                  p;      ///< row permutation matrix
  perm_type                  q_inv;  ///< column inverse permutation matrix
  sss_solver_type            dense_solver;   ///< dense decomposition
  mutable sparse_direct_type sparse_solver;  ///< sparse solver
#if HILUCSI_HAS_SPARSE_MKL
  MKL_SpTrSolver<value_type, index_type> mkl_L;
  MKL_SpTrSolver<value_type, index_type> mkl_U;
#endif

 private:
  /// \brief allow casting to \a char, this is needed to add an empty node
  Prec(char) {}

  friend class std::list<Prec>;  // give list friendship
};

/// \typedef Precs
/// \brief multilevel preconditioners
/// \tparam ValueType value data type, e.g. \a double
/// \tparam IndexType index data type, e.g. \a int
/// \tparam IntervalBased default is true, using interval based
/// \ingroup slv
///
/// We choose to use STL list because adding new node is constant time without
/// bothering copying. It's worth noting that querying the size of \a list has
/// become constant since C++11. Also, with the constructors in \ref Prec, the
/// following two ways can be used to construct list:
///
/// \code{.cpp}
/// using precs_t = Precs<double, int>;
/// using prec_t = precs_t::value_type;
/// precs_t precs;
/// precs.emplace_back(m, n, /* rvalue references */);
/// \endcode
///
/// and
///
/// \code{.cpp}
/// using precs_t = Precs<double, int>;
/// using prec_t = precs_t::value_type;
/// precs_t precs;
/// precs.push_back(prec_t::EMPTY_PREC);
/// auto &prec = precs.back();
/// prec.move_destroy(/* lvalue references */);
/// prec.m = m;
/// prec.n = n;
/// \endcode
///
/// These fit into the usage for Precs. Notice that for the second way, after
/// calling \ref Prec::move_destroy, all input arguments will be destroyed.
template <class ValueType, class IndexType, bool IntervalBased = true>
using Precs = std::list<Prec<ValueType, IndexType, IntervalBased>>;

}  // namespace hilucsi

#endif  // _HILUCSI_ALG_PREC_HPP
