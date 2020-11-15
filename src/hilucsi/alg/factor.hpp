///////////////////////////////////////////////////////////////////////////////
//                This file is part of HILUCSI project                       //
///////////////////////////////////////////////////////////////////////////////

/*!
 * \file hilucsi/alg/factor.hpp
 * \brief Kernels for deferred factorization
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

#ifndef _HILUCSI_ALG_FACTOR_HPP
#define _HILUCSI_ALG_FACTOR_HPP

#include <algorithm>
#include <cmath>
#include <type_traits>
#include <utility>

#ifdef HILUCSI_SAVE_FIRST_LEVEL_PERM_A
#  include <ctime>
#  include <random>
#endif  // HILUCSI_SAVE_FIRST_LEVEL_PERM_A

#include "hilucsi/ds/Array.hpp"
#include "hilucsi/ds/CompressedStorage.hpp"
#include "hilucsi/ds/DenseMatrix.hpp"
#include "hilucsi/ds/PermMatrix.hpp"
#include "hilucsi/ds/SparseVec.hpp"
#include "hilucsi/pre/driver.hpp"
#include "hilucsi/utils/Timer.hpp"
#include "hilucsi/utils/common.hpp"

#include "hilucsi/alg/Crout.hpp"
#include "hilucsi/alg/Schur.hpp"
#include "hilucsi/alg/thresholds.hpp"

#ifndef DOXYGEN_SHOULD_SKIP_THIS

#  ifndef HILUCSI_RESERVE_FAC
#    define HILUCSI_RESERVE_FAC 5
#  endif  // HILUCSI_RESERVE_FAC

#  ifndef HILUCSI_LASTLEVEL_DENSE_SIZE
#    define HILUCSI_LASTLEVEL_DENSE_SIZE 1500
#  endif  // HILUCSI_RESERVE_FAC

#  ifndef HILUCSI_LASTLEVEL_SPARSE_SIZE
#    define HILUCSI_LASTLEVEL_SPARSE_SIZE 15000
#  endif  // HILUCSI_LASTLEVEL_SPARSE_SIZE

#  ifndef HILUCSI_MIN_LOCAL_SIZE_PERCTG
#    define HILUCSI_MIN_LOCAL_SIZE_PERCTG 85
#  endif  // HILUCSI_RESERVE_FAC

#endif  // DOXYGEN_SHOULD_SKIP_THIS

namespace hilucsi {
namespace internal {

/*!
 * \addtogroup fac
 * @{
 */

/// \brief adjust parameters based on levels
/// \param[in] opts control parameters, i.e. Options
/// \param[in] lvl levels
/// \return refined parameters
inline std::tuple<double, double, double, double, double, double>
determine_fac_pars(const Options &opts, const int lvl) {
  double kappa_d, kappa, tau_U, tau_L, alpha_L, alpha_U;
  if (opts.rf_par) {
    const int    fac  = std::min(lvl, 2);
    const double fac2 = 1. / std::min(10.0, std::pow(10.0, lvl - 1));
    kappa_d           = std::max(2.0, std::pow(opts.kappa_d, 1. / fac));
    kappa             = std::max(2.0, std::pow(opts.kappa, 1. / fac));
    tau_U             = opts.tau_U * fac2;
    tau_L             = opts.tau_L * fac2;
    if (lvl > 2) {
      alpha_L = opts.alpha_L;
      alpha_U = opts.alpha_U;
    } else {
      alpha_L = opts.alpha_L * fac;
      alpha_U = opts.alpha_U * fac;
    }
  } else {
    kappa_d = opts.kappa_d;
    kappa   = opts.kappa;
    tau_U   = opts.tau_U;
    tau_L   = opts.tau_L;
    alpha_L = opts.alpha_L;
    alpha_U = opts.alpha_U;
  }
  return std::make_tuple(kappa_d, kappa, tau_U, tau_L, alpha_L, alpha_U);
}

/// \brief extract permutated diagonal
/// \tparam CcsType input ccs matrix, see \ref CCS
/// \tparam ScalingType scaling vector type, see \ref Array
/// \tparam PermType permutation matrix type, see \ref BiPermMatrix
/// \param[in] s row scaling vector
/// \param[in] A input matrix in CCS format
/// \param[in] t column scaling vector
/// \param[in] m leading block size
/// \param[in] p row permutation vector
/// \param[in] q column permutation vector
/// \param[in] m0 actual extrated size, default is \a m
/// \return permutated diagonal of \a A
///
/// This routine, essentially, is to compute:
///
/// \f[
///   \mathbf{D}=\left(\mathbf{SAT}\right)_{\mathbf{p}_{1:m},
///     \mathbf{q}_{1:m}}
/// \f]
///
/// This routine is used before \ref Crout update to extract the initial
/// diagonal entries.
template <class CcsType, class ScalingType, class PermType>
inline Array<typename CcsType::value_type> extract_perm_diag(
    const ScalingType &s, const CcsType A, const ScalingType &t,
    const typename CcsType::size_type m, const PermType &p, const PermType &q,
    const typename CcsType::size_type m0 = 0) {
  using value_type = typename CcsType::value_type;
  using size_type  = typename CcsType::size_type;
  using array_type = Array<value_type>;

  hilucsi_error_if(m > A.nrows() || m > A.ncols(),
                   "invalid leading block size %zd", m);

  array_type diag(m);
  hilucsi_error_if(diag.status() == DATA_UNDEF, "memory allocation failed");

  auto            v_begin = A.vals().cbegin();
  auto            i_begin = A.row_ind().cbegin();
  const size_type M       = m0 ? m0 : m;
  for (size_type i = 0u; i < M; ++i) {
    hilucsi_assert((size_type)q[i] < A.ncols(),
                   "permutated index %zd exceeds the col bound",
                   (size_type)q[i]);
    hilucsi_assert((size_type)p[i] < A.nrows(),
                   "permutated index %zd exceeds the row bound",
                   (size_type)p[i]);
    // using binary search
    auto info = find_sorted(A.row_ind_cbegin(q[i]), A.row_ind_cend(q[i]), p[i]);
    if (info.first)
      diag[i] = s[p[i]] * *(v_begin + (info.second - i_begin)) * t[q[i]];
    else {
      // hilucsi_warning("zero diagonal entry %zd detected!", i);
      diag[i] = 0;
    }
  }
  return diag;
}

/// \brief extract the \a E part
/// \tparam CrsType input crs matrix, see \ref CRS
/// \tparam ScalingType scaling vector type, see \ref Array
/// \tparam PermType permutation matrix type, see \ref BiPermMatrix
/// \param[in] s row scaling vector
/// \param[in] A input matrix in crs format
/// \param[in] t column scaling vector
/// \param[in] m leading block size (final)
/// \param[in] p row permutation vector
/// \param[in] q column permutation vector
/// \return The \a E part in \ref CCS format
/// \sa extract_F
///
/// This routine is to extract the \a E part \b after \ref Crout update.
/// Essentially, this routine is to perform:
///
/// \f[
///   \mathbf{E}=\left(\mathbf{SAT}\right)_{\mathbf{p}_{m+1:n},
///     \mathbf{q}_{1:m}}
/// \f]
template <class CrsType, class ScalingType, class PermType>
inline typename CrsType::other_type extract_E(
    const ScalingType &s, const CrsType &A, const ScalingType &t,
    const typename CrsType::size_type m, const PermType &p, const PermType &q) {
  // it's efficient to extract E from CRS
  static_assert(CrsType::ROW_MAJOR, "input A must be CRS!");
  using ccs_type   = typename CrsType::other_type;
  using size_type  = typename CrsType::size_type;
  using index_type = typename CrsType::index_type;

  const size_type n = A.nrows();

  hilucsi_error_if(m > n || m > A.ncols(),
                   "leading block size %zd should not exceed matrix size", m);
  const size_type N = n - m;
  ccs_type        E(N, m);
  if (!N) {
    // hilucsi_warning("empty E matrix detected!");
    return E;
  }

  auto &col_start = E.col_start();
  col_start.resize(m + 1);
  hilucsi_error_if(col_start.status() == DATA_UNDEF,
                   "memory allocation failed");
  std::fill(col_start.begin(), col_start.end(), 0);

  for (size_type i = m; i < n; ++i) {
    for (auto itr = A.col_ind_cbegin(p[i]), last = A.col_ind_cend(p[i]);
         itr != last; ++itr) {
      const size_type qi = q.inv(*itr);
      if (qi < m) ++col_start[qi + 1];
    }
  }

  // accumulate for nnz
  for (size_type i = 0u; i < m; ++i) col_start[i + 1] += col_start[i];

  if (!col_start[m]) {
    // hilucsi_warning(
    //     "exactly zero E, this most likely is a bug! Continue anyway...");
    return E;
  }

  E.reserve(col_start[m]);
  auto &row_ind = E.row_ind();
  auto &vals    = E.vals();

  hilucsi_error_if(
      row_ind.status() == DATA_UNDEF || vals.status() == DATA_UNDEF,
      "memory allocation failed");

  row_ind.resize(col_start[m]);
  vals.resize(col_start[m]);

  for (size_type i = m; i < n; ++i) {
    const size_type pi  = p[i];
    auto            itr = A.col_ind_cbegin(pi), last = A.col_ind_cend(pi);
    auto            v_itr = A.val_cbegin(pi);
    const auto      si    = s[pi];
    for (; itr != last; ++itr, ++v_itr) {
      const size_type j  = *itr;
      const size_type qi = q.inv(j);
      if (qi < m) {
        row_ind[col_start[qi]] = i - m;
        vals[col_start[qi]]    = si * *v_itr * t[j];
        ++col_start[qi];
      }
    }
  }

  // revert
  index_type tmp(0);
  for (size_type i = 0u; i < m; ++i) std::swap(col_start[i], tmp);

  return E;
}

/// \brief extract the \a F part
/// \tparam CcsType input ccs matrix, see \ref CCS
/// \tparam ScalingType scaling vector type, see \ref Array
/// \tparam PermType permutation matrix type, see \ref BiPermMatrix
/// \tparam BufferType work space array, can be \ref Array or \a std::vector
/// \param[in] s row scaling vector
/// \param[in] A input matrix in CCS format
/// \param[in] t column scaling vector
/// \param[in] m leading block size
/// \param[in] p row permutation vector
/// \param[in] q column permutation vector
/// \param buf work space
/// \return The \a F part in ccs storage
/// \sa extract_E
///
/// Note that unlike extracting the \a E part, this routine takes \ref CCS as
/// input, and with the permutation vectors, we need a value buffer space as
/// an intermidiate storage to store the values so that is will make sorting
/// much easier. The buffer space is a dense array, and can be passed in from
/// that of \a l or \a ut (if squared systems).
///
/// This routine, essentially, is to compute:
///
/// \f[
///   \mathbf{F}=\left(\mathbf{SAT}\right)_{\mathbf{p}_{1:m},
///     \mathbf{q}_{m+1:n}}
/// \f]
template <class CcsType, class ScalingType, class PermType, class BufferType>
inline CcsType extract_F(const ScalingType &s, const CcsType &A,
                         const ScalingType &               t,
                         const typename CcsType::size_type m, const PermType &p,
                         const PermType &q, BufferType &buf) {
  static_assert(!CcsType::ROW_MAJOR, "input A must be CCS!");
  using size_type  = typename CcsType::size_type;
  using index_type = typename CcsType::index_type;

  const size_type n = A.ncols();

  hilucsi_error_if(m > n || m > A.nrows(),
                   "leading block size %zd should not exceed matrix size", m);

  const size_type N = n - m;
  CcsType         F(m, N);
  if (!N) {
    // hilucsi_warning("empty F matrix detected!");
    return F;
  }

  auto &col_start = F.col_start();
  col_start.resize(N + 1);
  hilucsi_error_if(col_start.status() == DATA_UNDEF,
                   "memory allocation failed");
  col_start.front() = 0;

  for (size_type i = 0u; i < N; ++i) {
    const size_type qi = q[i + m];
    size_type       nnz(0);
    std::for_each(A.row_ind_cbegin(qi), A.row_ind_cend(qi),
                  [&](const index_type i) {
                    if (static_cast<size_type>(p.inv(i)) < m) ++nnz;
                  });
    col_start[i + 1] = col_start[i] + nnz;
  }

  if (!(col_start[N])) {
    // hilucsi_warning(
    //     "exactly zero F, this most likely is a bug! Continue anyway...");
    return F;
  }

  F.reserve(col_start[N]);
  auto &row_ind = F.row_ind();
  auto &vals    = F.vals();
  hilucsi_error_if(
      row_ind.status() == DATA_UNDEF || vals.status() == DATA_UNDEF,
      "memory allocation failed");

  row_ind.resize(col_start[N]);
  vals.resize(col_start[N]);

  auto v_itr = vals.begin();

  for (size_type i = 0u; i < N; ++i) {
    const size_type qi      = q[i + m];
    auto            itr     = F.row_ind_begin(i);
    auto            A_itr   = A.row_ind_cbegin(qi);
    auto            A_v_itr = A.val_cbegin(qi);
    const auto      ti      = t[qi];
    for (auto A_last = A.row_ind_cend(qi); A_itr != A_last;
         ++A_itr, ++A_v_itr) {
      const size_type j  = *A_itr;
      const size_type pi = p.inv(j);
      if (pi < m) {
        *itr = pi;
        ++itr;
        buf[pi] = s[j] * *A_v_itr * ti;
      }
    }
    std::sort(F.row_ind_begin(i), itr);
    for (auto itr = F.row_ind_cbegin(i), last = F.row_ind_cend(i); itr != last;
         ++itr, ++v_itr)
      *v_itr = buf[*itr];
  }
  return F;
}

/// \class CompressedTypeTrait
/// \brief Core component to filter CCS and CRS types
/// \tparam CsType1 compressed type I
/// \tparam CsType2 compressed type II
///
/// Since we allow user to arbitrarily use either \ref CCS or \ref CRS as input,
/// we need to build the counterpart and, most importantly, determine the
/// \ref CCS and \ref CRS from the input and its counterpart. This helper trait
/// is to accomplish this task.
template <class CsType1, class CsType2>
class CompressedTypeTrait {
  static_assert(CsType1::ROW_MAJOR ^ CsType2::ROW_MAJOR,
                "cannot have same type");
  constexpr static bool _1_IS_CRS = CsType1::ROW_MAJOR;
  ///< flag of type I is \ref CRS

 public:
  typedef typename std::conditional<_1_IS_CRS, CsType1, CsType2>::type crs_type;
  ///< \ref CRS type
  typedef typename crs_type::other_type ccs_type;  ///< \ref CCS type

  /// \brief choose crs type from two inputs
  /// \param[in] A crs type
  /// \return reference to \a A
  /// \sa crs_type, select_ccs
  template <class T = void>
  inline static const crs_type &select_crs(
      const CsType1 &A, const CsType2 & /* B */,
      const typename std::enable_if<_1_IS_CRS, T>::type * = nullptr) {
    return A;
  }

  // dual version
  template <class T = void>
  inline static const crs_type &select_crs(
      const CsType1 & /* A */, const CsType2 &B,
      const typename std::enable_if<!_1_IS_CRS, T>::type * = nullptr) {
    return B;
  }

  /// \brief choose ccs type from two inputs
  /// \param[in] B ccs type
  /// \return reference to \a B
  /// \sa ccs_type, select_crs
  template <class T = void>
  inline static const ccs_type &select_ccs(
      const CsType1 & /* A */, const CsType2 &B,
      const typename std::enable_if<_1_IS_CRS, T>::type * = nullptr) {
    return B;
  }

  // dual version
  template <class T = void>
  inline static const ccs_type &select_ccs(
      const CsType1 &A, const CsType2 & /* B */,
      const typename std::enable_if<!_1_IS_CRS, T>::type * = nullptr) {
    return A;
  }
};

/// \brief compress offsets to have a compact L and U
/// \tparam L_Type storage for \a L, see \ref CCS
/// \tparam U_Type storage for \a U, see \ref CRS
/// \tparam PosArray array for storing starting positions, see \ref Array
/// \param[in,out] U uncompressed \a U part
/// \param[in,out] L uncompressed \a L part
/// \param[in] U_start starting positions of the offset of \a U
/// \param[in] L_start starting positions of the offset of \a L
/// \param[in] m leading block size
/// \param[in] dfrs total number of deferrals
template <class L_Type, class U_Type, class PosArray>
inline void compress_tails(U_Type &U, L_Type &L, const PosArray &U_start,
                           const PosArray &                   L_start,
                           const typename PosArray::size_type m,
                           const typename PosArray::size_type dfrs) {
  using size_type  = typename PosArray::size_type;
  using index_type = typename L_Type::index_type;

  if (dfrs) {
    const auto comp_index = [=](index_type &j) { j -= dfrs; };
    auto       U_first = U.col_ind().begin(), L_first = L.row_ind().begin();
    for (size_type i(0); i < m; ++i) {
      std::for_each(U_first + U_start[i], U.col_ind_end(i), comp_index);
      std::for_each(L_first + L_start[i], L.row_ind_end(i), comp_index);
    }
  }

  // reshape the secondary axis of the matrices
  L.resize_nrows(L.nrows() / 2);
  U.resize_ncols(U.ncols() / 2);

#ifndef NDEBUG
  L.check_validity();
  U.check_validity();
#endif
}

/// \brief print out information regarding handling Schur complement
inline void print_post_flag(const int flag) {
  hilucsi_info("\t=================================");
  switch (flag) {
    case 0:
      hilucsi_info("\tthe Schur compl. has good size");
      break;
    case 1:
      hilucsi_info(
          "\tresort to complete factorization\n"
          "\ton the input due to too many\n"
          "\tstatic deferrals");
      break;
    case 2:
      hilucsi_info(
          "\tresort to complete factorization\n"
          "\ton the input due to too many\n"
          "\tstatic+dynamic deferrals");
      break;
    default:
      hilucsi_info(
          "\tuse complete factorization on\n"
          "\tthe Schur compl. due to its size\n"
          "\tis relatively large compared to\n"
          "\tthe input");
      break;
  }
  hilucsi_info("\t=================================");
}

/*!
 * @}
 */ // fac group

}  // namespace internal

/// \brief perform partial incomplete LU for a level
/// \tparam IsSymm if \a true, then assume a symmetric leading block
/// \tparam CsType input compressed storage, either \ref CRS or \ref CCS
/// \tparam CroutStreamer information streamer for \ref Crout update
/// \tparam PrecsType multilevel preconditioner type, \ref Precs and \ref Prec
/// \tparam IntArray integer array for storing sizes
/// \param[in] A input for this level
/// \param[in] m0 initial leading block size
/// \param[in] N reference \b global size for determining Schur sparsity
/// \param[in] opts control parameters
/// \param[in] Crout_info information streamer, API same as \ref hilucsi_info
/// \param[in,out] precs list of preconditioner, newly computed components will
///                      be pushed back to the list.
/// \param[in,out] row_sizes local row nnz of user input
/// \param[in,out] col_sizes local column nnz of user input
/// \param[in,out] stats hierarchical stats
/// \param[in] schur_threads threads usedin Schur-related computations
/// \return Schur complement for next level (if needed), in the same type as
///         that of the input, i.e. \a CsType
/// \ingroup fac
template <bool IsSymm, class CsType, class CroutStreamer, class PrecsType,
          class IntArray>
inline CsType level_factorize(const CsType &                   A,
                              const typename CsType::size_type m0,
                              const typename CsType::size_type N,
                              const Options &                  opts,
                              const CroutStreamer &Crout_info, PrecsType &precs,
                              IntArray &row_sizes, IntArray &col_sizes,
                              typename CsType::size_type *stats,
                              const int                   schur_threads = 1) {
  typedef CsType                      input_type;
  typedef typename CsType::other_type other_type;
  using cs_trait = internal::CompressedTypeTrait<input_type, other_type>;
  typedef typename cs_trait::crs_type    crs_type;
  typedef typename cs_trait::ccs_type    ccs_type;
  typedef typename CsType::index_type    index_type;
  typedef typename CsType::size_type     size_type;
  typedef typename CsType::value_type    value_type;
  typedef DenseMatrix<value_type>        dense_type;
  typedef typename PrecsType::value_type prec_type;  // precs is std::list

  hilucsi_error_if(A.nrows() != A.ncols(),
                   "only squared systems are supported");

  hilucsi_assert(m0 <= std::min(A.nrows(), A.ncols()),
                 "leading size should be smaller than size of A");
  const size_type cur_level = precs.size() + 1;

  if (hilucsi_verbose(INFO, opts))
    hilucsi_info("\nenter level %zd (%s).\n", cur_level,
                 (IsSymm ? "symmetric" : "asymmetric"));

  DefaultTimer timer;

  // build counterpart type
  other_type A_counterpart(A);

  // now use our trait and its static methods to precisely determine the ccs
  // and crs components.
  const crs_type &A_crs = cs_trait::select_crs(A, A_counterpart);
  const ccs_type &A_ccs = cs_trait::select_ccs(A, A_counterpart);

  // handle row and column sizes
  if (cur_level == 1u) {
    row_sizes.resize(A.nrows());
    col_sizes.resize(A.ncols());
    constexpr static double min_local_size_ratio =
        HILUCSI_MIN_LOCAL_SIZE_PERCTG / 100.0;
    for (size_type i(0); i < A.nrows(); ++i) row_sizes[i] = A_crs.nnz_in_row(i);
    for (size_type i(0); i < A.ncols(); ++i) col_sizes[i] = A_ccs.nnz_in_col(i);
    // filter out too small terms
    const size_type lower_row =
        std::ceil(min_local_size_ratio * A.nnz() / A.nrows());
    const size_type lower_col =
        std::ceil(min_local_size_ratio * A.nnz() / A.ncols());
    std::replace_if(row_sizes.begin(), row_sizes.begin() + A.nrows(),
                    [=](const index_type i) { return i < lower_row; },
                    lower_row);
    std::replace_if(col_sizes.begin(), col_sizes.begin() + A.ncols(),
                    [=](const index_type i) { return i < lower_col; },
                    lower_col);
  }

  const size_type must_symm_pre_lvls =
      opts.symm_pre_lvls <= 0 ? 0 : opts.symm_pre_lvls;

  // preprocessing
  timer.start();
  Array<value_type>        s, t;
  BiPermMatrix<index_type> p, q;
  size_type                m;
  if (!IsSymm && cur_level > must_symm_pre_lvls) {
    if (hilucsi_verbose(INFO, opts))
      hilucsi_info(
          "performing asymm preprocessing with leading block size %zd...", m0);
    m = do_preprocessing<false>(A_ccs, A_crs, m0, cur_level, opts, s, t, p, q);
  } else {
    if (hilucsi_verbose(INFO, opts))
      hilucsi_info(
          "performing symm preprocessing with leading block size %zd...", m0);
    m = do_preprocessing<true>(A_ccs, A_crs, m0, cur_level, opts, s, t, p, q);
  }

  timer.finish();  // prefile pre-processing

  // if post flag is 0, means we do standard Schur treatment after ILU
  // if post flag is 2, means we have too many static and dynamic deferrals,
  //    thus we set S=A
  // if post flag is -1, means we have moderate many deferrals, thus we
  //    directly factorize S with complete factorization.
  int post_flag = 0;

  if (hilucsi_verbose(INFO, opts)) {
    hilucsi_info("preprocessing done with leading block size %zd...", m);
    hilucsi_info("time: %gs", timer.time());
  }

#ifdef HILUCSI_SAVE_FIRST_LEVEL_PERM_A
  if (cur_level == 1u) {
    std::mt19937                       eng(std::time(0));
    std::uniform_int_distribution<int> d(1000, 1000000);
    const std::string fname  = "Perm_A_" + std::to_string(d(eng)) + ".hilucsi";
    auto              A_perm = A_crs.compute_perm(p(), q.inv(), m);
    Array<value_type> s2(m), t2(m);
    for (size_type i = 0; i < m; ++i) {
      s2[i] = s[p[i]];
      t2[i] = t[q[i]];
    }
    A_perm.scale_diag_left(s2);
    A_perm.scale_diag_right(t2);
    hilucsi_info("\nsaving first level permutated matrix to file %s\n",
                 fname.c_str());
    A_perm.write_bin(fname.c_str(), IsSymm ? m : size_type(0));
  }
#endif

  if (hilucsi_verbose(INFO, opts)) hilucsi_info("preparing data variables...");

  timer.start();

  // extract diagonal
  auto d = internal::extract_perm_diag(s, A_ccs, t, m, p, q);

  // create U storage with deferred
  crs_type U(m, A.ncols() * 2);
  hilucsi_error_if(U.row_start().status() == DATA_UNDEF,
                   "memory allocation failed for U:row_start at level %zd.",
                   cur_level);
  do {
    const size_type rsv_fac = HILUCSI_RESERVE_FAC <= 0 ? std::ceil(opts.alpha_U)
                                                       : HILUCSI_RESERVE_FAC;
    U.reserve(A.nnz() * rsv_fac);
    hilucsi_error_if(
        U.col_ind().status() == DATA_UNDEF || U.vals().status() == DATA_UNDEF,
        "memory allocation failed for U-nnz arrays at level %zd.", cur_level);
  } while (false);

  // create L storage with deferred
  ccs_type L(A.nrows() * 2, m);
  hilucsi_error_if(L.col_start().status() == DATA_UNDEF,
                   "memory allocation failed for L:col_start at level %zd.",
                   cur_level);
  do {
    const size_type rsv_fac = HILUCSI_RESERVE_FAC <= 0 ? std::ceil(opts.alpha_L)
                                                       : HILUCSI_RESERVE_FAC;
    L.reserve(A.nnz() * rsv_fac);
    hilucsi_error_if(
        L.row_ind().status() == DATA_UNDEF || L.vals().status() == DATA_UNDEF,
        "memory allocation failed for L-nnz arrays at level %zd.", cur_level);
  } while (false);

  // create l and ut buffer
  SparseVector<value_type, index_type> l(A.nrows() * 2), ut(A.ncols() * 2);

  // create buffer for L and U starts
  Array<index_type> L_start(m), U_start(m);
  hilucsi_error_if(
      L_start.status() == DATA_UNDEF || U_start.status() == DATA_UNDEF,
      "memory allocation failed for L_start and/or U_start at level %zd.",
      cur_level);

  Array<index_type> L_offsets;
  if (IsSymm) {
    L_offsets.resize(m);
    hilucsi_error_if(L_offsets.status() == DATA_UNDEF,
                     "memory allocation failed for L_offsets at level %zd.",
                     cur_level);
  }

  const Array<index_type> &Crout_L_start = !IsSymm ? L_start : L_offsets;

  // create buffer for L and U lists
  Array<index_type> L_list(A.nrows() * 2), U_list(A.ncols() * 2);
  hilucsi_error_if(
      L_list.status() == DATA_UNDEF || U_list.status() == DATA_UNDEF,
      "memory allocation failed for L_list and/or U_list at level %zd.",
      cur_level);

  // set default value
  std::fill(L_list.begin(), L_list.end(), static_cast<index_type>(-1));
  std::fill(U_list.begin(), U_list.end(), static_cast<index_type>(-1));

  // create storage for kappa's
  Array<value_type> kappa_l(m), kappa_ut(m);
  hilucsi_error_if(
      kappa_l.status() == DATA_UNDEF || kappa_ut.status() == DATA_UNDEF,
      "memory allocation failed for kappa_l and/or kappa_ut at level %zd.",
      cur_level);

  U.begin_assemble_rows();
  L.begin_assemble_cols();

  // localize parameters
  double kappa_d, kappa, tau_L, tau_U, alpha_L, alpha_U;
  std::tie(kappa_d, kappa, tau_L, tau_U, alpha_L, alpha_U) =
      internal::determine_fac_pars(opts, cur_level);

  // Removing bounding the large diagonal values
  const auto is_bad_diag = [=](const value_type a) -> bool {
    return std::abs(1. / a) > kappa_d;  // || std::abs(a) > tau_d;
  };

  const size_type m2(m), n(A.nrows());

  // deferred permutations
  Array<index_type> P(n * 2), Q(n * 2);
  hilucsi_error_if(P.status() == DATA_UNDEF || Q.status() == DATA_UNDEF,
                   "memory allocation failed for P and/or Q at level %zd",
                   cur_level);
  std::copy_n(p().cbegin(), n, P.begin());
  std::copy_n(q().cbegin(), n, Q.begin());
  auto &P_inv = p.inv(), &Q_inv = q.inv();

  // 0 for defer due to diagonal, 1 for defer due to bad inverse conditioning
  index_type info_counter[] = {0, 0, 0, 0, 0, 0, 0};

  if (hilucsi_verbose(INFO, opts)) hilucsi_info("start Crout update...");
  Crout step;
  for (; step < m; ++step) {
    // first check diagonal
    bool            pvt         = is_bad_diag(d[step.deferred_step()]);
    const size_type m_prev      = m;
    const size_type defers_prev = step.defers();
    info_counter[0] += pvt;

    Crout_info(" Crout step %zd, leading block size %zd", step, m);

    // compute kappa for u wrp deferred index
    step.update_kappa(U, U_list, U_start, kappa_ut);
    // then compute kappa for l
    if (!IsSymm)
      step.update_kappa(L, L_list, L_start, kappa_l);
    else
      kappa_l[step] = kappa_ut[step];

    // check condition number if diagonal doesn't satisfy
    if (!pvt) {
      pvt = std::abs(kappa_ut[step]) > kappa || std::abs(kappa_l[step]) > kappa;
      info_counter[1] += pvt;
    }

    // handle defer
    if (pvt) {
      while (m > step) {
        --m;
        const auto tail_pos = n + step.defers();
        step.defer_entry(tail_pos, U_start, U, U_list);
        if (!IsSymm)
          step.defer_entry(tail_pos, L_start, L, L_list);
        else
          step.symm_defer_l(tail_pos, L_start, L, L_list, L_offsets);
        P[tail_pos]        = p[step.deferred_step()];
        Q[tail_pos]        = q[step.deferred_step()];
        P_inv[P[tail_pos]] = tail_pos;
        Q_inv[Q[tail_pos]] = tail_pos;
        // mark as empty entries
        P[step.deferred_step()] = Q[step.deferred_step()] = -1;

        step.increment_defer_counter();  // increment defers here
        // handle the last step
        if (step.deferred_step() >= m2) {
          m = step;
          break;
        }
        pvt = is_bad_diag(d[step.deferred_step()]);
        if (pvt) {
          ++info_counter[0];
          continue;
        }
        // compute kappa for u wrp deferred index
        step.update_kappa(U, U_list, U_start, kappa_ut);
        // then compute kappa for l
        if (!IsSymm)
          step.update_kappa(L, L_list, L_start, kappa_l);
        else
          kappa_l[step] = kappa_ut[step];
        pvt =
            std::abs(kappa_ut[step]) > kappa || std::abs(kappa_l[step]) > kappa;
        if (pvt) {
          ++info_counter[1];
          continue;
        }
        break;
      }                      // while
      if (m == step) break;  // break for
    }

    //----------------
    // inverse thres
    //----------------

    const auto k_ut = kappa_ut[step], k_l = kappa_l[step];

    // check pivoting
    hilucsi_assert(!(std::abs(k_ut) > kappa || std::abs(k_l) > kappa),
                   "should not happen!");

    Crout_info("  kappa_ut=%g, kappa_l=%g", (double)std::abs(k_ut),
               (double)std::abs(k_l));

    Crout_info(
        "  previous/current leading block sizes %zd/%zd, local/total "
        "defers=%zd/%zd",
        m_prev, m, step.defers() - defers_prev, step.defers());

    //------------------------
    // update start positions
    //------------------------

    Crout_info("  updating L_start/U_start and performing Crout update");

    // compress diagonal
    step.compress_array(d);

    // compress permutation vectors
    step.compress_array(p);
    step.compress_array(q);

    // compute ut
    step.compute_ut(s, A_crs, t, p[step], Q_inv, L, L_start, L_list, d, U,
                    U_start, ut);
    // compute l
    step.compute_l<IsSymm>(s, A_ccs, t, P_inv, q[step], m2, L, Crout_L_start, d,
                           U, U_start, U_list, l);

    // update diagonal entries for u first
#ifndef NDEBUG
    const bool u_is_nonsingular =
#else
    (void)
#endif
        step.scale_inv_diag(d, ut);
    hilucsi_assert(!u_is_nonsingular, "u is singular at level %zd step %zd",
                   cur_level, step);

    // update diagonals b4 dropping
    step.update_diag<IsSymm>(l, ut, m2, d);

#ifndef NDEBUG
    const bool l_is_nonsingular =
#else
    (void)
#endif
        step.scale_inv_diag(d, l);
    hilucsi_assert(!l_is_nonsingular, "l is singular at level %zd step %zd",
                   cur_level, step);

    //---------------
    // drop and sort
    //---------------

    const size_type ori_ut_size = ut.size(), ori_l_size = l.size();

    // apply drop for U
    apply_num_dropping(tau_U, std::abs(k_ut) * kappa_d, ut);
#ifndef HILUCSI_DISABLE_SPACE_DROP
    const size_type n_ut = ut.size();
    apply_space_dropping(row_sizes[p[step]], alpha_U, ut);
    info_counter[3] += n_ut - ut.size();
#endif  // HILUCSI_DISABLE_SPACE_DROP
    info_counter[5] += ori_ut_size - ut.size();
    ut.sort_indices();

    // push back rows to U
    U.push_back_row(step, ut.inds().cbegin(), ut.inds().cbegin() + ut.size(),
                    ut.vals());

    Crout_info("  ut sizes before/after dropping %zd/%zd, drops=%zd",
               ori_ut_size, ut.size(), ori_ut_size - ut.size());

    // apply numerical dropping on L
    apply_num_dropping(tau_L, std::abs(k_l) * kappa_d, l);

    if (IsSymm) {
#ifndef HILUCSI_DISABLE_SPACE_DROP
      // for symmetric cases, we need first find the leading block size
      auto info =
          find_sorted(ut.inds().cbegin(), ut.inds().cbegin() + ut.size(), m2);
      apply_space_dropping(col_sizes[q[step]], alpha_L, l,
                           info.second - ut.inds().cbegin());

      auto u_last = info.second;
#else   // !HILUCSI_DISABLE_SPACE_DROP
      auto u_last = ut.inds().cbegin() + ut.size();
#endif  // HILUCSI_DISABLE_SPACE_DROP

      l.sort_indices();
      Crout_info(
          "  l sizes (asymm parts) before/after dropping %zd/%zd, drops=%zd",
          ori_l_size, l.size(), ori_l_size - l.size());

      // push back symmetric entries and offsets
      L.push_back_col(step, ut.inds().cbegin(), u_last, ut.vals(),
                      l.inds().cbegin(), l.inds().cbegin() + l.size(),
                      l.vals());
    } else {
#ifndef HILUCSI_DISABLE_SPACE_DROP
      const size_type n_l = l.size();
      apply_space_dropping(col_sizes[q[step]], alpha_L, l);
      info_counter[4] += n_l - l.size();
#endif  // HILUCSI_DISABLE_SPACE_DROP
      info_counter[6] += ori_l_size - l.size();
      l.sort_indices();
      Crout_info("  l sizes before/after dropping %zd/%zd, drops=%zd",
                 ori_l_size, l.size(), ori_l_size - l.size());
      L.push_back_col(step, l.inds().cbegin(), l.inds().cbegin() + l.size(),
                      l.vals());
    }

    // update position
    step.update_compress(U, U_list, U_start);
    step.update_compress(L, L_list, L_start);
    if (IsSymm) {
      if (!step.defers() && m2 == n)
        L_offsets[step] = L.nnz_in_col(step);
      else
        step.symm_update_lstart(L, m2, L_offsets);
    }
    Crout_info(" Crout step %zd done!", step);
  }  // for

  // compress permutation vectors
  for (; step < n; ++step) {
    step.assign_gap_array(P, p);
    step.assign_gap_array(Q, q);
  }

  // rebuild the inverse mappings
  p.build_inv();
  q.build_inv();

  U.end_assemble_rows();
  L.end_assemble_cols();

  // Important! Revert the starting positions to global index that is required
  // in older routines
  for (size_type i(0); i < m; ++i) {
    L_start[i] += L.col_start()[i];
    U_start[i] += U.row_start()[i];
  }

  // compress tails
  internal::compress_tails(U, L, U_start, L_start, m, step.defers());

  timer.finish();  // profile Crout update

  // analyze reminder size
  if (!post_flag && (double)m <= 0.25 * m2) {
    post_flag = 2;  // check after factorization
    m         = 0;
    for (size_type i(0); i < sizeof(info_counter) / sizeof(index_type); ++i)
      info_counter[i] = 0;
  } else if ((double)m <= 0.4 * m2)
    post_flag = -1;

  // collecting stats for deferrals
  stats[0] += m0 - m;                            // total deferals
  stats[1] += m ? step.defers() : size_type(0);  // dynamic deferrals
  stats[2] += info_counter[0];                   // diagonal deferrals
  stats[3] += info_counter[1];                   // conditioning deferrals

  // collecting stats for dropping
  stats[4] += info_counter[5] + info_counter[6];  // total droppings
  stats[5] += info_counter[3] + info_counter[4];  // space-based droppings

  // now we are done
  if (hilucsi_verbose(INFO, opts)) {
    hilucsi_info(
        "finish Crout update...\n"
        "\ttotal deferrals=%zd\n"
        "\tleading block size in=%zd\n"
        "\tleading block size out=%zd\n"
        "\tdiff=%zd\n"
        "\tdiag deferrals=%zd\n"
        "\tinv-norm deferrals=%zd\n"
        "\tdrop ut=%zd\n"
        "\tspace drop ut=%zd\n"
        "\tdrop l=%zd\n"
        "\tspace drop l=%zd\n"
        "\tmin |kappa_u|=%g\n"
        "\tmax |kappa_u|=%g\n"
        "\tmin |kappa_l|=%g\n"
        "\tmax |kappa_l|=%g\n"
        "\tmax |d|=%g",
        step.defers(), m0, m, m0 - m, (size_type)info_counter[0],
        (size_type)info_counter[1], (size_type)info_counter[5],
        (size_type)info_counter[3], (size_type)info_counter[6],
        (size_type)info_counter[4],
        (double)std::abs(
            *std::min_element(kappa_ut.cbegin(), kappa_ut.cbegin() + m,
                              [](const value_type l, const value_type r) {
                                return std::abs(l) < std::abs(r);
                              })),
        (double)std::abs(
            *std::max_element(kappa_ut.cbegin(), kappa_ut.cbegin() + m,
                              [](const value_type l, const value_type r) {
                                return std::abs(l) < std::abs(r);
                              })),
        (double)std::abs(
            *std::min_element(kappa_l.cbegin(), kappa_l.cbegin() + m,
                              [](const value_type l, const value_type r) {
                                return std::abs(l) < std::abs(r);
                              })),
        (double)std::abs(
            *std::max_element(kappa_l.cbegin(), kappa_l.cbegin() + m,
                              [](const value_type l, const value_type r) {
                                return std::abs(l) < std::abs(r);
                              })),
        (double)std::abs(
            *std::max_element(d.cbegin(), d.cbegin() + m,
                              [](const value_type l, const value_type r) {
                                return std::abs(l) < std::abs(r);
                              })));
    if (post_flag == 2)
      hilucsi_info(
          "too many dynamic deferrals, resort to complete factorization");
    hilucsi_info("time: %gs", timer.time());
  }

  if (hilucsi_verbose(INFO, opts)) {
    hilucsi_info("computing Schur complement and assembling Prec...");
    internal::print_post_flag(post_flag);
  }

  timer.start();

  crs_type S;

  const auto L_nnz = L.nnz(), U_nnz = U.nnz();

  ccs_type L_B, U_B;
  if (m && post_flag <= 0) {
    // TODO: need to handle the case where m is too small comparing to n, thus
    // we need to resort to direct factorizations.
    do {
      DefaultTimer timer2;
      timer2.start();
      auto L_E = L.template split_crs<true>(m, L_start);
      L_B      = L.template split<false>(m, L_start);
      L.destroy();
      timer2.finish();
      if (hilucsi_verbose(INFO, opts))
        hilucsi_info("splitting LB and freeing L took %gs.", timer2.time());
      crs_type U_F;
      do {
        timer2.start();
        auto U_F2 = U.template split_ccs<true>(m, U_start);
        U_B       = U.template split_ccs<false>(m, U_start);
        U.destroy();
        timer2.finish();
        if (hilucsi_verbose(INFO, opts))
          hilucsi_info("splitting UB and freeing U took %gs.", timer2.time());
        timer2.start();
        const size_type nnz1 = L_E.nnz(), nnz2 = U_F2.nnz();
#ifndef HILUCSI_NO_DROP_LE_UF
        double a_L = opts.alpha_L, a_U = opts.alpha_U;
        if (cur_level == 1u && opts.fat_schur_1st) {
          a_L *= 2;
          a_U *= 2;
        }
        if (hilucsi_verbose(INFO, opts))
          hilucsi_info(
              "applying dropping on L_E and U_F with alpha_{L,U}=%g,%g...", a_L,
              a_U);
        if (m < n) {
          // use P and Q as buffers
          P[0] = Q[0] = 0;
          for (size_type i(m); i < n; ++i) {
            P[i - m + 1] = P[i - m] + row_sizes[p[i]];
            Q[i - m + 1] = Q[i - m] + col_sizes[q[i]];
          }
#  ifndef _OPENMP
          drop_L_E(P, a_L, L_E, l.vals(), l.inds());
          drop_U_F(Q, a_U, U_F2, ut.vals(), ut.inds());
#  else
          mt::drop_L_E_and_U_F(P, a_L, Q, a_U, L_E, U_F2, l.vals(), l.inds(),
                               ut.vals(), ut.inds(), schur_threads);
#  endif
        }
#endif  // HILUCSI_NO_DROP_LE_UF
        timer2.finish();
        U_F = crs_type(U_F2);
        if (hilucsi_verbose(INFO, opts))
          hilucsi_info("nnz(L_E)=%zd/%zd, nnz(U_F)=%zd/%zd, time: %gs...", nnz1,
                       L_E.nnz(), nnz2, U_F.nnz(), timer2.time());
      } while (false);  // U_F2 got freed

      timer2.start();
// compute S version of Schur complement
#ifndef _OPENMP
      (void)schur_threads;
      S = compute_Schur_simple(s, A_crs, t, p, q, m, L_E, d, U_F, l);
#else
      if (hilucsi_verbose(INFO, opts))
        hilucsi_info("using %d for Schur computation...", schur_threads);
      S = mt::compute_Schur_simple(s, A_crs, t, p, q, m, L_E, d, U_F, l,
                                   schur_threads);
#endif
      timer2.finish();
      if (hilucsi_verbose(INFO, opts))
        hilucsi_info("pure Schur computation time: %gs...", timer2.time());
    } while (false);
  } else {
    S = A;
    p.make_eye();
    q.make_eye();
    std::fill(s.begin(), s.end(), 1);
    std::fill(t.begin(), t.end(), 1);
  }

  // compute the nnz(A)-nnz(B) from the first level
  size_type AmB_nnz(0);
  for (size_type i(m); i < n; ++i) AmB_nnz += row_sizes[p[i]] + col_sizes[q[i]];

  // L and U got freed, only L_B and U_B exist

  const size_type dense_thres1 = static_cast<size_type>(
                      std::max(opts.alpha_L, opts.alpha_U) * AmB_nnz),
                  dense_thres2 =
                      std::max(static_cast<size_type>(
                                   std::ceil(opts.c_d * std::cbrt(N))),
                               size_type(HILUCSI_LASTLEVEL_DENSE_SIZE));

  if (hilucsi_verbose(INFO, opts))
    hilucsi_info(
        "nnz(S_C)=%zd, nnz(L/L_B)=%zd/%zd, nnz(U/U_B)=%zd/%zd\n"
        "dense_thres{1,2}=%zd/%zd...",
        S.nnz(), L_nnz, L_B.nnz(), U_nnz, U_B.nnz(), dense_thres1,
        dense_thres2);

  // test H version
  const size_type nm = n - m;
  dense_type      S_D;
  hilucsi_assert(S_D.empty(), "fatal!");

  if (post_flag < 0 ||
      (size_type)std::ceil(nm * nm * opts.rho) <= dense_thres1 ||
      nm <= dense_thres2 || !m) {
    S_D = dense_type::from_sparse(S);
    if (hilucsi_verbose(INFO, opts))
      hilucsi_info("converted Schur complement (S) to dense for last level...");
  }

  if (S_D.empty()) {
    // update the row and column sizes
    // Important! Update this information before destroying p and q in the
    // following emplace_back call
    // NOTE use P and Q as buffers
    for (size_type i(m); i < n; ++i) {
      P[i] = row_sizes[p[i]];
      Q[i] = col_sizes[q[i]];
    }
    for (size_type i(m); i < n; ++i) {
      row_sizes[i - m] = P[i];
      col_sizes[i - m] = Q[i];
    }
  }

  ccs_type E, F;

  if (input_type::ROW_MAJOR) {
    F = internal::extract_F(s, A_ccs, t, m, p, q, ut.vals());
    A_counterpart.destroy();
    E = internal::extract_E(s, A_crs, t, m, p, q);
  } else {
    E = internal::extract_E(s, A_crs, t, m, p, q);
    A_counterpart.destroy();
    F = internal::extract_F(s, A_ccs, t, m, p, q, ut.vals());
  }

  // recursive function free
  if (cur_level > 1u) const_cast<input_type &>(A).destroy();

  precs.emplace_back(m, n, std::move(L_B), std::move(d), std::move(U_B),
                     std::move(E), std::move(F), std::move(s), std::move(t),
                     std::move(p()), std::move(p.inv()), std::move(q()),
                     std::move(q.inv()));

  // report if using interval based data structures
  if (hilucsi_verbose(INFO, opts)) {
    precs.back().report_status_lu();
    precs.back().report_status_ef();
  }

  // if dense is not empty, then push it back
  if (!S_D.empty()) {
    auto &last_level = precs.back().dense_solver;
    last_level.set_matrix(std::move(S_D));
    last_level.factorize(opts);
    if (hilucsi_verbose(INFO, opts))
      hilucsi_info("successfully factorized the dense component...");
  }
#ifdef HILUCSI_ENABLE_MUMPS
  else {
    if (nm <= static_cast<size_type>(HILUCSI_LASTLEVEL_SPARSE_SIZE)) {
      DefaultTimer timer2;
      auto &       last_level = precs.back().sparse_solver;
      last_level.set_info(
          opts.mumps_blr,
          std::sqrt(
              Const<typename ValueTypeTrait<value_type>::value_type>::EPS),
          schur_threads);
      const double nnz_b4 = 0.01 * S.nnz();
      timer2.start();
      last_level.factorize(S);
      timer2.finish();
      hilucsi_error_if(last_level.info(), "%s returned error %d",
                       last_level.backend(), last_level.info());
      if (hilucsi_verbose(INFO, opts))
        hilucsi_info(
            "successfully factorized the sparse component with %s...\n"
            "\tfill-ratio: %.2f%%\n"
            "\ttime: %gs...",
            last_level.backend(), last_level.nnz() / nnz_b4, timer2.time());
    }
  }
#endif  // HILUCSI_ENABLE_MUMPS

  timer.finish();  // profile post-processing

  if (hilucsi_verbose(INFO, opts)) hilucsi_info("time: %gs", timer.time());

  if (hilucsi_verbose(INFO, opts))
    hilucsi_info("\nfinish level %zd.", cur_level);

  if (precs.back().is_last_level()) return input_type();
  return input_type(S);
}

}  // namespace hilucsi

#endif  // _HILUCSI_ALG_FACTOR_HPP
