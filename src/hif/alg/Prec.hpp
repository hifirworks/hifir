///////////////////////////////////////////////////////////////////////////////
//                  This file is part of the HIFIR library                   //
///////////////////////////////////////////////////////////////////////////////

/*!
 * \file hif/alg/Prec.hpp
 * \brief Multilevel preconditioner interface
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

#ifndef _HIF_ALG_PREC_HPP
#define _HIF_ALG_PREC_HPP

#include <list>
#include <type_traits>
#include <utility>

#include "hif/ds/CompressedStorage.hpp"
#include "hif/ds/IntervalCompressedStorage.hpp"
#include "hif/small_scale/solver.hpp"
#include "hif/utils/common.hpp"
#ifdef HIF_ENABLE_MUMPS
#  include "hif/sparse_direct/mumps.hpp"
#endif  // HIF_ENABLE_MUMPS

namespace hif {

#ifndef DOXYGEN_SHOULD_SKIP_THIS

// use QRCP by default
#  ifndef HIF_DENSE_MODE
#    define HIF_DENSE_MODE 1
#  endif  // HIF_DENSE_MODE

namespace internal {
struct DummySparseSolver {
  inline constexpr bool        empty() const { return true; }
  inline constexpr std::size_t nnz() const { return 0; }
  inline constexpr int         info() const { return 0; }
  inline static const char *   backend() { return "DummySparse"; }
  template <class ArrayType>
  inline void solve(ArrayType &, const bool = false) {}
};
}  // namespace internal

template <class T>
struct DefaultDenseSolver {
  using unsymm_solver_type = void;
  using symm_solver_type   = void;
};

#endif  // DOXYGEN_SHOULD_SKIP_THIS

/// \class Prec
/// \brief A single level preconditioner
/// \tparam ValueType value data type, e.g. \a double
/// \tparam IndexType index data type, e.g. \a int
/// \tparam UserDenseFactor Potential user customized dense factory for the
///                         final Schur complement, if it is
///                         \a DefaultDenseSolver (default), the code then uses
///                         the built-in dense factors.
///
/// The following demonstrates how to customize a user-defined dense factory
///
/// \code{.cpp}
///   #include <HIF.hpp>
///   template <class T>
///   using MyQRCP = hif::QRCP<T>; // simply wrap built-in QRCP solver
///   template <class T>
///   struct MyDenseSolver : public hif::DefaultDenseSolver<T> {
///     using unsymm_solver_type = MyQRCP<T>;
///   };
///   hif::HIF<double, int, MyDenseSolver> my_fac;
/// \endcode
/// \ingroup slv
template <class ValueType, class IndexType,
          template <class> class UserDenseFactor = DefaultDenseSolver>
struct Prec {
  typedef ValueType                     value_type;  ///< value type
  typedef IndexType                     index_type;  ///< index type
  typedef CRS<value_type, index_type>   crs_type;    ///< crs type
  typedef CCS<value_type, index_type>   ccs_type;    ///< ccs type
  typedef Array<index_type>             perm_type;   ///< permutation
  typedef typename ccs_type::size_type  size_type;   ///< size
  typedef typename ccs_type::array_type array_type;  ///< array
  typedef ccs_type                      mat_type;    ///< interface type
  typedef typename std::conditional<
      std::is_same<typename ValueTypeTrait<value_type>::value_type,
                   long double>::value,
      typename ValueTypeMixedTrait<value_type>::reduce_type, value_type>::type
                                                          last_level_type;
  typedef typename ValueTypeTrait<value_type>::value_type scalar_type;
  ///< scalar type
  typedef Array<scalar_type> sarray_type;  ///< scalar array type
#ifdef HIF_ENABLE_MUMPS
  using sparse_direct_type = MUMPS<last_level_type>;
#else
  using sparse_direct_type = internal::DummySparseSolver;
#endif                                           // HIF_ENABLE_MUMPS
  static constexpr char EMPTY_PREC     = '\0';   ///< empty prec
  static constexpr bool INTERVAL_BASED = false;  ///< interval
  using data_mat_type                  = typename std::conditional<
      INTERVAL_BASED, typename using_interval_from_classical<mat_type>::type,
      mat_type>::type;  ///< data matrix type

 private:
  typedef SmallScaleSolverTrait<HIF_DENSE_MODE> _sss_trait;
  ///< small scale trait
  static constexpr bool _SAME_DENSE =
      std::is_same<UserDenseFactor<value_type>,
                   DefaultDenseSolver<value_type>>::value;

 public:
  typedef typename std::conditional<
      _SAME_DENSE, _sss_trait::template solver_type<last_level_type>,
      typename std::conditional<
          std::is_void<
              typename UserDenseFactor<value_type>::unsymm_solver_type>::value,
          _sss_trait::template solver_type<last_level_type>,
          typename UserDenseFactor<value_type>::unsymm_solver_type>::type>::type
      sss_solver_type;
  ///< small scaled solver type
  typedef typename std::conditional<
      _SAME_DENSE, SYEIG<last_level_type>,
      typename std::conditional<
          std::is_void<
              typename UserDenseFactor<value_type>::symm_solver_type>::value,
          SYEIG<last_level_type>,
          typename UserDenseFactor<value_type>::symm_solver_type>::type>::type
      sss_symm_solver_type;
  ///< small scaled solver for symmetric systems

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
  /// \param[in] P_inv inverse row permutation
  /// \param[in] Q column permutation
  /// \param[in] Q_inv inverse column permutation
  /// \note This allows us to use emplace back in STL efficiently
  Prec(size_type mm, size_type nn, mat_type &&L_b, array_type &&d_b,
       mat_type &&U_b, mat_type &&e, mat_type &&f, sarray_type &&S,
       sarray_type &&T, perm_type &&P, perm_type &&P_inv, perm_type &&Q,
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
        p_inv(std::move(P_inv)),
        q(std::move(Q)),
        q_inv(std::move(Q_inv)) {}

  /// \brief get number of nonzeros
  inline size_type nnz() const {
    size_type nz = m ? L_B.nnz() + U_B.nnz() + m : 0;
    if (n - m) nz += E.nnz() + F.nnz();
    if (!dense_solver.empty() || !symm_dense_solver.empty())
      return nz + (n - m) * (n - m);
    if (!sparse_solver.empty()) return nz + sparse_solver.nnz();
    return nz;
  }

  /// \brief get the number of nonzeros in \a E and \a F components
  inline size_type nnz_ef() const {
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
        hif_info(
            "%s was not able to be converted to interval, too small average "
            "interval (<2)",
            name);
      } else {
        hif_info(
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
  inline typename std::enable_if<is_interval_cs<mat_type>::value, T>::type
  report_status_lu() const {
    static const auto report_kernel = [](const data_mat_type &mat,
                                         const char *         name) {
      if (!mat.converted()) {
        hif_info(
            "%s was not able to be converted to interval, too small average "
            "interval (<2)",
            name);
      } else {
        hif_info(
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
  inline typename std::enable_if<!is_interval_cs<mat_type>::value, T>::type
  report_status_lu() const {}

#if 0
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
  /// \param[in] tag optimization tag
  inline void optimize(const int tag = 0) {
    _L_crs = crs_type(L_B);
    _U_crs = crs_type(U_B);
    if (!tag) {
      ls_L.setup(_L_crs);
      ls_L.template optimize<false>();
      ls_U.setup(_U_crs);
      ls_U.template optimize<true>();
    }
#  if HIF_HAS_SPARSE_MKL
    else if (tag == 1) {
      mkl_L.setup(_L_crs.row_start(), _L_crs.col_ind(), _L_crs.vals());
      mkl_U.setup(_U_crs.row_start(), _U_crs.col_ind(), _U_crs.vals());
      mkl_L.template optimize<false>(1);
      mkl_U.template optimize<true>(1);
    }
#  else
    else
      hif_error("unknown optimization tag %d", tag);
#  endif
  }
#endif

  /// \brief check if this a last level preconditioner
  ///
  /// The idea is first check if the \ref dense_solver is empty or not; be aware
  /// that there is another situation that we should treat the preconditioner
  /// is last level---if \ref m is equal to \ref n.
  ///
  /// \note Currently, we test m == n, which is fine for squared systems.
  inline bool is_last_level() const {
    return !sparse_solver.empty() || !dense_solver.empty() ||
           !symm_dense_solver.empty() || m == n;
  }

  /// \brief check the (numerical) rank for last level (if applicable)
  inline size_type last_rank() const {
    if (!is_last_level() || m == n) return 0u;
    // NOTE: For sparse last level, we currently don't have QR, thus full rank
    if (!sparse_solver.empty()) return n - m;
    if (!dense_solver.empty()) return dense_solver.rank();
    return symm_dense_solver.rank();
  }

  /// \brief query size information
  inline void inquire_sizes(size_type &m_, size_type &n_, size_type &nnz_L,
                            size_type &nnz_U, size_type &nnz_E,
                            size_type &nnz_F) const {
    m_    = m;
    n_    = n;
    nnz_L = L_B.nnz();
    nnz_U = U_B.nnz();
    nnz_E = E.nnz();
    nnz_F = F.nnz();
  }

  /// \brief export all numerical data
  template <class ExportType, typename T = void>
  inline typename std::enable_if<!INTERVAL_BASED, T>::type export_sparse_data(
      typename ExportType::ipointer L_indptr,
      typename ExportType::ipointer L_indices,
      typename ExportType::pointer L_vals, typename ExportType::pointer D_vals,
      typename ExportType::ipointer U_indptr,
      typename ExportType::ipointer U_indices,
      typename ExportType::pointer  U_vals,
      typename ExportType::ipointer E_indptr,
      typename ExportType::ipointer E_indices,
      typename ExportType::pointer  E_vals,
      typename ExportType::ipointer F_indptr,
      typename ExportType::ipointer F_indices,
      typename ExportType::pointer  F_vals,
      typename ValueTypeTrait<typename ExportType::value_type>::value_type
          *s_vals,
      typename ValueTypeTrait<typename ExportType::value_type>::value_type
          *                         t_vals,
      typename ExportType::ipointer p_vals,
      typename ExportType::ipointer pinv_vals,
      typename ExportType::ipointer q_vals,
      typename ExportType::ipointer qinv_vals, const bool jit_destroy = false) {
    // leading block
    export_compressed_data<ExportType>(L_B, L_indptr, L_indices, L_vals);  // L
    std::copy_n(d_B.cbegin(), m, D_vals);                                  // D
    if (jit_destroy) array_type().swap(d_B);
    export_compressed_data<ExportType>(U_B, U_indptr, U_indices, U_vals);  // U
    if (jit_destroy) U_B.destroy();
    // off-diagonal blocks
    export_compressed_data<ExportType>(E, E_indptr, E_indices, E_vals);  // E
    if (jit_destroy) E.destroy();
    export_compressed_data<ExportType>(F, F_indptr, F_indices, F_vals);  // F
    if (jit_destroy) F.destroy();
    // row/column scaling
    std::copy_n(s.cbegin(), n, s_vals);  // s
    std::copy_n(t.cbegin(), n, t_vals);  // t
    // row/column permutations
    std::copy_n(p.cbegin(), n, p_vals);         // P
    std::copy_n(p_inv.cbegin(), n, pinv_vals);  // P^T
    std::copy_n(q.cbegin(), n, q_vals);         // Q
    std::copy_n(q_inv.cbegin(), n, qinv_vals);  // Q^T
    if (jit_destroy) {
      // destroy all 1D arrays
      array_type().swap(d_B);
      sarray_type().swap(s);
      sarray_type().swap(t);
      perm_type().swap(p);
      perm_type().swap(p_inv);
      perm_type().swap(q);
      perm_type().swap(q_inv);
      m = n = 0;
    }
  }

  template <class ExportType, typename T = void>
  inline typename std::enable_if<INTERVAL_BASED, T>::type export_sparse_data(
      typename ExportType::ipointer, typename ExportType::ipointer,
      typename ExportType::pointer, typename ExportType::pointer,
      typename ExportType::ipointer, typename ExportType::ipointer,
      typename ExportType::pointer, typename ExportType::ipointer,
      typename ExportType::ipointer, typename ExportType::pointer,
      typename ExportType::ipointer, typename ExportType::ipointer,
      typename ExportType::pointer,
      typename ValueTypeTrait<typename ExportType::value_type>::value_type *,
      typename ValueTypeTrait<typename ExportType::value_type>::value_type *,
      typename ExportType::ipointer, typename ExportType::ipointer,
      typename ExportType::ipointer, typename ExportType::ipointer,
      const bool = false) const {
    hif_error("cannot export data for interval-based compressed formats!");
  }

  template <class T>
  inline void inquire_or_export_dense(T *mat, size_type &nrows,
                                      size_type &ncols,
                                      const bool destroy = false) {
#ifdef HIF_ENABLE_MUMPS
    hif_error("cannot export data for direct sparse last level!");
#endif
    if (!dense_solver.empty()) {
      if (!mat) {
        nrows = dense_solver.mat_backup().nrows();
        ncols = dense_solver.mat_backup().ncols();
        return;
      }
      // assume mat is properly allocated
      std::copy(dense_solver.mat_backup().array().cbegin(),
                dense_solver.mat_backup().array().cend(), mat);
      if (destroy) {
        array_type().swap(dense_solver.mat_backup().array());
        array_type().swap(dense_solver.mat().array());
      }
      return;
    }
    hif_assert(!symm_dense_solver.empty(), "fatal");
    if (!mat) {
      nrows = symm_dense_solver.mat_backup().nrows();
      ncols = symm_dense_solver.mat_backup().ncols();
      return;
    }
    // assume mat is properly allocated
    std::copy(symm_dense_solver.mat_backup().array().cbegin(),
              symm_dense_solver.mat_backup().array().cend(), mat);
    if (destroy) {
      array_type().swap(symm_dense_solver.mat_backup().array());
      array_type().swap(symm_dense_solver.mat().array());
    }
  }

  size_type            m;             ///< leading block size
  size_type            n;             ///< system size
  mat_type             L_B;           ///< lower part of leading block
  array_type           d_B;           ///< diagonal block of leading block
  mat_type             U_B;           ///< upper part of leading block
  data_mat_type        E;             ///< scaled and permutated E part
  data_mat_type        F;             ///< scaled and permutated F part
  sarray_type          s;             ///< row scaling vector
  sarray_type          t;             ///< column scaling vector
  perm_type            p;             ///< row permutation matrix
  perm_type            p_inv;         ///< row inverse permutation matrix
  perm_type            q;             ///< column permutation matrix
  perm_type            q_inv;         ///< column inverse permutation matrix
  sss_solver_type      dense_solver;  ///< dense decomposition
  sss_symm_solver_type symm_dense_solver;    ///< symmetric dense decomposition
  mutable sparse_direct_type sparse_solver;  ///< sparse solver

 protected:
  crs_type _L_crs;  ///< crs type for parallel trsv
  crs_type _U_crs;  ///< crs type for parallel trsv

 private:
  /// \brief allow casting to \a char, this is needed to add an empty node
  Prec(char) {}

  friend class std::list<Prec>;  // give list friendship
};

/// \typedef Precs
/// \brief multilevel preconditioners
/// \tparam ValueType value data type, e.g. \a double
/// \tparam IndexType index data type, e.g. \a int
/// \tparam UserDenseFactor Potential user customized dense factor
/// \ingroup slv
///
/// We choose to use STL list because adding new node is constant time without
/// bothering copying. It's worth noting that querying the size of \a list has
/// become constant since C++11. Also, with the constructors in \ref Prec, the
/// following way can be used to construct list:
///
/// \code{.cpp}
/// using precs_t = Precs<double, int>;
/// using prec_t = precs_t::value_type;
/// precs_t precs;
/// precs.emplace_back(m, n, /* rvalue references */);
/// \endcode
template <class ValueType, class IndexType,
          template <class> class UserDenseFactor = DefaultDenseSolver>
using Precs = std::list<Prec<ValueType, IndexType, UserDenseFactor>>;

}  // namespace hif

#endif  // _HIF_ALG_PREC_HPP
