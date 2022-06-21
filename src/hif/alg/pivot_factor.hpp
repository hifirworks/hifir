///////////////////////////////////////////////////////////////////////////////
//                  This file is part of the HIFIR library                   //
///////////////////////////////////////////////////////////////////////////////

/*!
 * \file hif/alg/pivot_factor.hpp
 * \brief Kernels for deferred factorization with thresholding pivoting
 * \author Qiao Chen

\verbatim
Copyright (C) 2021 NumGeom Group at Stony Brook University

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
\endverbatim

 */

#ifndef _HIF_ALG_PIVOTFACTOR_HPP
#define _HIF_ALG_PIVOTFACTOR_HPP

#include <algorithm>
#include <cmath>
#include <type_traits>
#include <utility>

#include "hif/ds/AugmentedStorage.hpp"

#include "hif/alg/PivotCrout.hpp"
#include "hif/alg/factor.hpp"

namespace hif {

/// \brief perform partial incomplete LU for a level with thresholding pivoting
/// \tparam CsType input compressed storage, either \ref CRS or \ref CCS
/// \tparam CroutStreamer information streamer for \ref Crout update
/// \tparam PrecsType multilevel preconditioner type, \ref Precs and \ref Prec
/// \tparam IntArray integer array for storing sizes
/// \param[in] A input for this level
/// \param[in] m0 initial leading block size
/// \param[in] N reference \b global size for determining Schur sparsity
/// \param[in] opts control parameters
/// \param[in] Crout_info information streamer, API same as \ref hif_info
/// \param[in,out] precs list of preconditioner, newly computed components will
///                      be pushed back to the list.
/// \param[in,out] row_sizes local row nnz of user input
/// \param[in,out] col_sizes local column nnz of user input
/// \param[in,out] stats hierarchical stats
/// \param[in] schur_threads threads usedin Schur-related computations
/// \return Schur complement for next level (if needed), in the same type as
///         that of the input, i.e. \a CsType
/// \sa level_factorize
/// \ingroup fac
template <class CsType, class CroutStreamer, class PrecsType, class IntArray>
inline CsType pivot_level_factorize(
    const CsType &A, const typename CsType::size_type m0,
    const typename CsType::size_type N, const Options &opts,
    const CroutStreamer &Crout_info, PrecsType &precs, IntArray &row_sizes,
    IntArray &col_sizes, typename CsType::size_type *stats,
    const int schur_threads = 1) {
  typedef CsType                      input_type;
  typedef typename CsType::other_type other_type;
  using cs_trait = internal::CompressedTypeTrait<input_type, other_type>;
  typedef typename cs_trait::crs_type                     crs_type;
  typedef typename cs_trait::ccs_type                     ccs_type;
  typedef AugCRS<crs_type>                                aug_crs_type;
  typedef AugCCS<ccs_type>                                aug_ccs_type;
  typedef typename CsType::index_type                     index_type;
  typedef typename CsType::indptr_type                    indptr_type;
  typedef typename CsType::size_type                      size_type;
  typedef typename CsType::value_type                     value_type;
  typedef typename ValueTypeTrait<value_type>::value_type scalar_type;
  typedef DenseMatrix<value_type>                         dense_type;

  hif_error_if(A.nrows() != A.ncols(), "only squared systems are supported");

  hif_assert(m0 <= std::min(A.nrows(), A.ncols()),
             "leading size should be smaller than size of A");
  const size_type cur_level = precs.size() + 1;

  if (hif_verbose(INFO, opts))
    hif_info("\nenter level %zd with pivoting.\n", cur_level);

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
        HIF_MIN_LOCAL_SIZE_PERCTG / 100.0;
    for (size_type i(0); i < A.nrows(); ++i) row_sizes[i] = A_crs.nnz_in_row(i);
    for (size_type i(0); i < A.ncols(); ++i) col_sizes[i] = A_ccs.nnz_in_col(i);
    // filter out too small terms
    const size_type lower_row =
        std::ceil(min_local_size_ratio * A.nnz() / A.nrows());
    const size_type lower_col =
        std::ceil(min_local_size_ratio * A.nnz() / A.ncols());
    std::replace_if(
        row_sizes.begin(), row_sizes.begin() + A.nrows(),
        [=](const index_type i) { return (size_type)i < lower_row; },
        lower_row);
    std::replace_if(
        col_sizes.begin(), col_sizes.begin() + A.ncols(),
        [=](const index_type i) { return (size_type)i < lower_col; },
        lower_col);
  }

  const size_type must_symm_pre_lvls =
      opts.symm_pre_lvls <= 0 ? 0 : opts.symm_pre_lvls;

  // preprocessing
  timer.start();
  Array<scalar_type>       s, t;
  BiPermMatrix<index_type> p, q;
  size_type                m;
  if (!opts.no_pre) {
    if (cur_level > must_symm_pre_lvls) {
      if (hif_verbose(INFO, opts))
        hif_info(
            "performing asymm preprocessing with leading block size %zd...",
            m0);
      m = do_preprocessing<false>(A_ccs, A_crs, m0, cur_level, opts, s, t, p,
                                  q);
    } else {
      if (hif_verbose(INFO, opts))
        hif_info("performing symm preprocessing with leading block size %zd...",
                 m0);
      m = do_preprocessing<true>(A_ccs, A_crs, m0, cur_level, opts, s, t, p, q);
    }
  } else {
    if (hif_verbose(INFO, opts)) hif_info("skipping preprocessing... ");
    p.resize(A.nrows());
    q.resize(A.ncols());
    p.make_eye();
    q.make_eye();
    s.resize(A.nrows());
    std::fill(s.begin(), s.end(), scalar_type(1));
    t.resize(A.ncols());
    std::fill(t.begin(), t.end(), scalar_type(1));
    m = A.nrows();
  }

  timer.finish();  // prefile pre-processing

  // if post flag is 0, means we do standard Schur treatment after ILU
  // if post flag is 2, means we have too many static and dynamic deferrals,
  //    thus we set S=A
  // if post flag is -1, means we have moderate many deferrals, thus we
  //    directly factorize S with complete factorization.
  int post_flag = 0;

  if (hif_verbose(INFO, opts)) {
    hif_info("preprocessing done with leading block size %zd...", m);
    hif_info("time: %gs", timer.time());
  }

  if (hif_verbose(INFO, opts)) hif_info("preparing data variables...");

  timer.start();

  // create array for diagonal entries
  // auto d = internal::extract_perm_diag(s, A_ccs, t, m, p, q);
  Array<value_type> d(m);
  hif_error_if(d.status() == DATA_UNDEF,
               "memory allocation failed for d at level %zd.", cur_level);

  // create U storage with deferred
  aug_crs_type U(m, A.ncols() * 2);
  hif_error_if(U.row_start().status() == DATA_UNDEF,
               "memory allocation failed for U:row_start at level %zd.",
               cur_level);
  do {
    const size_type rsv_fac =
        HIF_RESERVE_FAC <= 0 ? std::ceil(opts.alpha_U) : HIF_RESERVE_FAC;
    U.reserve(A.nnz() * rsv_fac);
    hif_error_if(
        U.col_ind().status() == DATA_UNDEF || U.vals().status() == DATA_UNDEF,
        "memory allocation failed for U-nnz arrays at level %zd.", cur_level);
  } while (false);

  // create L storage with deferred
  aug_ccs_type L(A.nrows() * 2, m);
  hif_error_if(L.col_start().status() == DATA_UNDEF,
               "memory allocation failed for L:col_start at level %zd.",
               cur_level);
  do {
    const size_type rsv_fac =
        HIF_RESERVE_FAC <= 0 ? std::ceil(opts.alpha_L) : HIF_RESERVE_FAC;
    L.reserve(A.nnz() * rsv_fac);
    hif_error_if(
        L.row_ind().status() == DATA_UNDEF || L.vals().status() == DATA_UNDEF,
        "memory allocation failed for L-nnz arrays at level %zd.", cur_level);
  } while (false);

  // create l and ut buffer
  SparseVector<value_type, index_type> l(A.nrows() * 2), ut(A.ncols() * 2);

  // create buffer for L and U starts
  Array<indptr_type> L_start(m), U_start(m);
  hif_error_if(
      L_start.status() == DATA_UNDEF || U_start.status() == DATA_UNDEF,
      "memory allocation failed for L_start and/or U_start at level %zd.",
      cur_level);

  // create storage for kappa's
  Array<value_type> kappa_l(m), kappa_ut(m);
  hif_error_if(
      kappa_l.status() == DATA_UNDEF || kappa_ut.status() == DATA_UNDEF,
      "memory allocation failed for kappa_l and/or kappa_ut at level %zd.",
      cur_level);

  U.begin_assemble_rows();
  L.begin_assemble_cols();

  // localize parameters
  double kappa_d, kappa, tau_L, tau_U, alpha_L, alpha_U;
  std::tie(kappa_d, kappa, tau_L, tau_U, alpha_L, alpha_U) =
      internal::determine_fac_pars(opts, cur_level);
  // for pivoting, we double alpha's
  alpha_L *= 4;
  alpha_U *= 4;

  // Removing bounding the large diagonal values
  const auto is_bad_diag = [=](const value_type a) -> bool {
    return std::abs(1. / a) > kappa_d;  // || std::abs(a) > tau_d;
  };

  const size_type m2(m), n(A.nrows());

  // deferred permutations
  Array<indptr_type> P(n * 2), Q(n * 2);
  hif_error_if(P.status() == DATA_UNDEF || Q.status() == DATA_UNDEF,
               "memory allocation failed for P and/or Q at level %zd",
               cur_level);
  std::copy_n(p().cbegin(), n, P.begin());
  std::copy_n(q().cbegin(), n, Q.begin());
  auto &P_inv = p.inv(), &Q_inv = q.inv();

  // 0 for defer due to diagonal, 1 for defer due to bad inverse conditioning
  index_type info_counter[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  if (hif_verbose(INFO, opts)) hif_info("start Crout update...");
  PivotCrout step;
  // define pivot unary predictors that ensure every pivot is within the leading
  // block range, and their corresponding incremental inverse norm estimations
  // for L and U are bounded bay kappa.
  auto pvt_l_op = [&](const index_type i) {
    if ((size_type)i >= m2)
      return false;  // pivots must be in the leading block
    step.update_kappa(L, kappa_l, i);
    return std::abs(kappa_l[step]) <= kappa;
  };
  auto pvt_u_op = [&](const index_type i) {
    if ((size_type)i >= m2)
      return false;  // pivots must be in the leading block
    step.update_kappa(U, kappa_ut, i);
    return std::abs(kappa_ut[step]) <= kappa;
  };
  for (; step < m; ++step) {
    const size_type m_prev      = m;
    const size_type defers_prev = step.defers();

    Crout_info(" Crout step %zd, leading block size %zd", step, m);

    //----------------------
    // thresholded pivoting
    //----------------------

    const auto n_pivots = step.apply_thres_pivot(
        s, t, A_ccs, A_crs, m2, opts.gamma, 4, Crout_info, L_start, U_start,
        pvt_l_op, pvt_u_op, p, P_inv, q, Q_inv, L, d, U, l, ut);
    const bool do_pivot = n_pivots.first || n_pivots.second;
    info_counter[9] += do_pivot;
    info_counter[7] += n_pivots.first;   // # of L pivoting
    info_counter[8] += n_pivots.second;  // # of U pivoting

    // first check diagonal
    bool do_defer = is_bad_diag(d[step.deferred_step()]);
    info_counter[10] += do_pivot && do_defer;
    info_counter[0] += do_defer;

    // compute kappa for u wrt deferred index
    step.update_kappa(U, kappa_ut);
    // then compute kappa for l
    step.update_kappa(L, kappa_l);

    // check condition number if diagonal doesn't satisfy
    if (!do_defer) {
      do_defer =
          std::abs(kappa_ut[step]) > kappa || std::abs(kappa_l[step]) > kappa;
      if (do_defer) {
        info_counter[11] += do_pivot && std::abs(kappa_l[step]) > kappa;
        info_counter[12] += do_pivot && std::abs(kappa_ut[step]) > kappa;
      }
      info_counter[1] += do_defer;
    }

    Crout_info("  perform pivoting=%s, perform deferring=%s",
               (do_pivot ? "yes" : "no"), (do_defer ? "yes" : "no"));

    // handle defer
    if (do_defer) {
      while (m > step) {
        --m;
        const auto tail_pos = n + step.defers();
        step.defer_entry(tail_pos, U);
        step.defer_entry(tail_pos, L);
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
        // compute kappa for u wrp deferred index
        step.update_kappa(U, kappa_ut);
        // then compute kappa for l
        step.update_kappa(L, kappa_l);
        do_defer =
            std::abs(kappa_ut[step]) > kappa || std::abs(kappa_l[step]) > kappa;
        if (do_defer) {
          ++info_counter[1];
          continue;
        }
        // compute diagonal
        d[step.deferred_step()] = step.compute_dk(
            s, A_ccs, t, P_inv, q[step.deferred_step()], L, L_start, d, U);
        if (is_bad_diag(d[step.deferred_step()])) {
          ++info_counter[0];
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

    // recompute l
    l.restore_cur_state();
    step.compute_l(s, A_ccs, t, P_inv, q[step.deferred_step()], L, L_start, d,
                   U, l);
    // recompute ut
    ut.restore_cur_state();  // must restore to initial state as used in pvt
    step.compute_ut(s, A_crs, t, p[step.deferred_step()], Q_inv, L, d, U,
                    U_start, ut);

    // compress diagonal
    step.compress_array(d);

    // compress permutation vectors
    step.compress_array(p);
    step.compress_array(q);

    // update diagonal entries for u first
#ifdef HIF_DEBUG
    const bool u_is_nonsingular =
#else
    (void)
#endif
        step.scale_inv_diag(d, ut);
    hif_assert(!u_is_nonsingular, "u is singular at level %zd step %zd",
               cur_level, step);

    // Diagonal is maintained in JIT fashion due to pivoting
    // step.update_diag<false>(l, ut, m2, d);

#ifdef HIF_DEBUG
    const bool l_is_nonsingular =
#else
    (void)
#endif
        step.scale_inv_diag(d, l);
    hif_assert(!l_is_nonsingular, "l is singular at level %zd step %zd",
               cur_level, step);

    //---------------
    // drop and sort
    //---------------

    const size_type ori_ut_size = ut.size(), ori_l_size = l.size();

    // apply drop for U
    apply_num_dropping(tau_U, std::abs(k_ut) * kappa_d, ut);
#ifndef HIF_DISABLE_SPACE_DROP
    const size_type n_ut = ut.size();
    apply_space_dropping(row_sizes[p[step]], alpha_U, ut);
    info_counter[3] += n_ut - ut.size();
#endif  // HIF_DISABLE_SPACE_DROP
    info_counter[5] += ori_ut_size - ut.size();
    ut.sort_indices();

    // push back rows to U
    U.push_back_row(step, ut.inds().cbegin(), ut.inds().cbegin() + ut.size(),
                    ut.vals());

    Crout_info("  ut sizes before/after dropping %zd/%zd, drops=%zd",
               ori_ut_size, ut.size(), ori_ut_size - ut.size());

    // apply numerical dropping on L
    apply_num_dropping(tau_L, std::abs(k_l) * kappa_d, l);
#ifndef HIF_DISABLE_SPACE_DROP
    const size_type n_l = l.size();
    apply_space_dropping(col_sizes[q[step]], alpha_L, l);
    info_counter[4] += n_l - l.size();
#endif  // HIF_DISABLE_SPACE_DROP
    info_counter[6] += ori_l_size - l.size();
    l.sort_indices();
    Crout_info("  l sizes before/after dropping %zd/%zd, drops=%zd", ori_l_size,
               l.size(), ori_l_size - l.size());
    L.push_back_col(step, l.inds().cbegin(), l.inds().cbegin() + l.size(),
                    l.vals());

    // update position
    step.update_compress(U, U_start);
    step.update_compress(L, L_start);
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
  if (hif_verbose(INFO2, opts)) {
    hif_info(
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
        "\tpivoting in L=%zd\n"
        "\tpivoting in U=%zd\n"
        "\tsteps with pivoting=%zd\n"
        "\tsteps with both pivoting & inv(D) deferring=%zd\n"
        "\tsteps with both pivoting & inv(L) deferring=%zd\n"
        "\tsteps with both pivoting & inv(U) deferring=%zd\n"
        "\tmin |kappa_u|=%g\n"
        "\tmax |kappa_u|=%g\n"
        "\tmin |kappa_l|=%g\n"
        "\tmax |kappa_l|=%g\n"
        "\tmax |d|=%g",
        step.defers(), m0, m, m0 - m, (size_type)info_counter[0],
        (size_type)info_counter[1], (size_type)info_counter[5],
        (size_type)info_counter[3], (size_type)info_counter[6],
        (size_type)info_counter[4], (size_type)info_counter[7],
        (size_type)info_counter[8], (size_type)info_counter[9],
        (size_type)info_counter[10], (size_type)info_counter[11],
        (size_type)info_counter[12],
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
  }

  if (hif_verbose(INFO, opts)) {
    if (post_flag == 2)
      hif_info("too many dynamic deferrals, resort to complete factorization");
    hif_info("time: %gs", timer.time());
    // finished logging for factorization here
    hif_info("computing Schur complement and assembling Prec...");
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
      if (hif_verbose(INFO2, opts))
        hif_info("splitting LB and freeing L took %gs.", timer2.time());
      crs_type U_F;
      do {
        timer2.start();
        auto U_F2 = U.template split_ccs<true>(m, U_start);
        U_B       = U.template split_ccs<false>(m, U_start);
        U.destroy();
        timer2.finish();
        if (hif_verbose(INFO2, opts))
          hif_info("splitting UB and freeing U took %gs.", timer2.time());
        timer2.start();
        const size_type nnz1 = L_E.nnz(), nnz2 = U_F2.nnz();
#ifndef HIF_NO_DROP_LE_UF
        double a_L = 4 * opts.alpha_L, a_U = 4 * opts.alpha_U;
        if (cur_level == 1u && opts.fat_schur_1st) {
          a_L *= 2;
          a_U *= 2;
        }
        if (hif_verbose(INFO2, opts))
          hif_info("applying dropping on L_E and U_F with alpha_{L,U}=%g,%g...",
                   a_L, a_U);
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
#endif  // HIF_NO_DROP_LE_UF
        timer2.finish();
        U_F = crs_type(U_F2);
        if (hif_verbose(INFO2, opts))
          hif_info("nnz(L_E)=%zd/%zd, nnz(U_F)=%zd/%zd, time: %gs...", nnz1,
                   L_E.nnz(), nnz2, U_F.nnz(), timer2.time());
      } while (false);  // U_F2 got freed

      timer2.start();
// compute S version of Schur complement
#ifndef _OPENMP
      (void)schur_threads;
      S = compute_Schur_simple(s, A_crs, t, p, q, m, L_E, d, U_F, l);
#else
      if (hif_verbose(INFO, opts))
        hif_info("using %d threads for Schur computation...", schur_threads);
      S = mt::compute_Schur_simple(s, A_crs, t, p, q, m, L_E, d, U_F, l,
                                   schur_threads);
#endif
      timer2.finish();
      if (hif_verbose(INFO, opts))
        hif_info("pure Schur computation time: %gs...", timer2.time());
    } while (false);
  } else {
    S = A_crs;
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
                  dense_thres2 = std::max(
                      static_cast<size_type>(
                          std::ceil(opts.c_d * std::cbrt(N))),
                      size_type(opts.dense_thres <= 0 ? 2000
                                                      : opts.dense_thres));

  if (hif_verbose(INFO, opts))
    hif_info(
        "nnz(S_C)=%zd, nnz(L/L_B)=%zd/%zd, nnz(U/U_B)=%zd/%zd\n"
        "dense_thres{1,2}=%zd/%zd...",
        S.nnz(), L_nnz, L_B.nnz(), U_nnz, U_B.nnz(), dense_thres1,
        dense_thres2);

  // test H version
  const size_type nm = n - m;
  dense_type      S_D;
  hif_assert(S_D.empty(), "fatal!");

  if (post_flag < 0 ||
      (size_type)std::ceil(nm * nm * opts.rho) <= dense_thres1 ||
      nm <= dense_thres2 || !m) {
    S_D = dense_type::from_sparse(S);
    if (hif_verbose(INFO, opts))
      hif_info("converted Schur complement (S) to dense for last level...");
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
  if (hif_verbose(INFO, opts)) {
    precs.back().report_status_lu();
    precs.back().report_status_ef();
  }

  // if dense is not empty, then push it back
  if (!S_D.empty()) {
    auto &last_level = precs.back().dense_solver;
    last_level.set_matrix(std::move(S_D));
    last_level.factorize(opts);
    if (hif_verbose(INFO, opts)) {
      hif_info(
          "successfully factorized the dense component of "
          "(size,rank)=(%zd,%zd)...",
          last_level.mat().nrows(), last_level.rank());
      hif_info("is the final Schur complement full-rank? %s",
               (last_level.mat().nrows() == last_level.rank() ? "yes" : "no"));
    }
  }

  timer.finish();  // profile post-processing

  if (hif_verbose(INFO, opts)) hif_info("time: %gs", timer.time());

  if (hif_verbose(INFO, opts)) hif_info("\nfinish level %zd.", cur_level);

  if (precs.back().is_last_level()) return input_type();
  return input_type(S);
}

}  // namespace hif

#endif  // _HIF_ALG_PIVOTFACTOR_HPP
