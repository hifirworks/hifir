//@HEADER
//----------------------------------------------------------------------------
//                Copyright (C) 2019 The PSMILU AUTHORS
//----------------------------------------------------------------------------
//@HEADER

/// \file psmilu_fac_pvt.hpp
/// \brief Implementation of incomplete multilevel deferred factorization
///        with pivoting.
/// \authors Qiao,

#ifndef _PSMILU_FACPVT_HPP
#define _PSMILU_FACPVT_HPP

#include <type_traits>

#include "psmilu_PivotCrout.hpp"
#include "psmilu_fac_defer.hpp"
#include "psmilu_pivot.hpp"

namespace psmilu {
namespace internal {
// for pivoting, since we will compute U and L then pivoting, we need to, then,
// adjust the starting positions
template <class AugType, class PosArray, class T = void>
inline typename std::enable_if<AugType::ROW_MAJOR, T>::type adjust_start_pos(
    const typename PosArray::size_type step, const AugType &U,
    const typename PosArray::size_type pivot, PosArray &start) {
  using index_type = typename AugType::index_type;
  using size_type  = typename PosArray::size_type;

  index_type aug_id_k   = U.start_col_id(step),
             aug_id_pvt = U.start_col_id(pivot);
  while (!U.is_nil(aug_id_k) || !U.is_nil(aug_id_pvt)) {
    if (!U.is_nil(aug_id_k) && !U.is_nil(aug_id_pvt)) {
      const size_type idx_k   = U.row_idx(aug_id_k),
                      idx_pvt = U.row_idx(aug_id_pvt);
      if (idx_k < idx_pvt) {
        ++start[idx_k];
        aug_id_k = U.next_col_id(aug_id_k);
      } else if (idx_k > idx_pvt) {
        --start[idx_pvt];
        aug_id_pvt = U.next_col_id(aug_id_pvt);
      } else {
        aug_id_k   = U.next_col_id(aug_id_k);
        aug_id_pvt = U.next_col_id(aug_id_pvt);
      }
    } else if (!U.is_nil(aug_id_k)) {
      ++start[U.row_idx(aug_id_k)];
      aug_id_k = U.next_col_id(aug_id_k);
    } else {
      --start[U.row_idx(aug_id_pvt)];
      aug_id_pvt = U.next_col_id(aug_id_pvt);
    }
  }
}

template <class AugType, class PosArray, class T = void>
inline typename std::enable_if<!AugType::ROW_MAJOR, T>::type adjust_start_pos(
    const typename PosArray::size_type step, const AugType &L,
    const typename PosArray::size_type pivot, PosArray &start) {
  using index_type = typename AugType::index_type;
  using size_type  = typename PosArray::size_type;

  index_type aug_id_k   = L.start_row_id(step),
             aug_id_pvt = L.start_row_id(pivot);
  while (!L.is_nil(aug_id_k) || !L.is_nil(aug_id_pvt)) {
    if (!L.is_nil(aug_id_k) && !L.is_nil(aug_id_pvt)) {
      const size_type idx_k   = L.col_idx(aug_id_k),
                      idx_pvt = L.col_idx(aug_id_pvt);
      if (idx_k < idx_pvt) {
        ++start[idx_k];
        aug_id_k = L.next_row_id(aug_id_k);
      } else if (idx_k > idx_pvt) {
        --start[idx_pvt];
        aug_id_pvt = L.next_row_id(aug_id_pvt);
      } else {
        aug_id_k   = L.next_row_id(aug_id_k);
        aug_id_pvt = L.next_row_id(aug_id_pvt);
      }
    } else if (!L.is_nil(aug_id_k)) {
      ++start[L.col_idx(aug_id_k)];
      aug_id_k = L.next_row_id(aug_id_k);
    } else {
      --start[L.col_idx(aug_id_pvt)];
      aug_id_pvt = L.next_row_id(aug_id_pvt);
    }
  }
}
}  // namespace internal

template <bool IsSymm, class CsType, class CroutStreamer, class PrecsType,
          class IntArray>
inline CsType iludp_factor_pvt(const CsType &                   A,
                               const typename CsType::size_type m0,
                               const typename CsType::size_type N,
                               const Options &                  opts,
                               const CroutStreamer &            Crout_info,
                               PrecsType &precs, IntArray &row_sizes,
                               IntArray &col_sizes) {
  typedef CsType                      input_type;
  typedef typename CsType::other_type other_type;
  using cs_trait = internal::CompressedTypeTrait<input_type, other_type>;
  typedef typename cs_trait::crs_type crs_type;
  typedef typename cs_trait::ccs_type ccs_type;
  typedef AugCRS<crs_type>            aug_crs_type;
  typedef AugCCS<ccs_type>            aug_ccs_type;
  typedef typename CsType::index_type index_type;
  typedef typename CsType::size_type  size_type;
  typedef typename CsType::value_type value_type;
  typedef DenseMatrix<value_type>     dense_type;
  constexpr static bool               ONE_BASED = CsType::ONE_BASED;

  psmilu_error_if(A.nrows() != A.ncols(), "only squared systems are supported");

  psmilu_assert(m0 <= std::min(A.nrows(), A.ncols()),
                "leading size should be smaller than size of A");
  const size_type cur_level = precs.size() + 1;

  if (psmilu_verbose(INFO, opts))
    psmilu_info("\nenter level %zd (%s).\n", cur_level,
                (IsSymm ? "symmetric" : "asymmetric"));

  DefaultTimer timer;

  // build counterpart type
  const other_type A_counterpart(A);

  // now use our trait and its static methods to precisely determine the ccs
  // and crs components.
  const crs_type &A_crs = cs_trait::select_crs(A, A_counterpart);
  const ccs_type &A_ccs = cs_trait::select_ccs(A, A_counterpart);

  // handle row and column sizes
  if (cur_level == 1u) {
    row_sizes.resize(A.nrows());
    col_sizes.resize(A.ncols());
  }
#ifndef PSMILU_USE_CUR_SIZES
  if (cur_level == 1u)
#endif  // PSMILU_USE_CUR_SIZES
  {
    for (size_type i(0); i < A.nrows(); ++i) row_sizes[i] = A_crs.nnz_in_row(i);
    for (size_type i(0); i < A.ncols(); ++i) col_sizes[i] = A_ccs.nnz_in_col(i);
  }

  if (psmilu_verbose(INFO, opts))
    psmilu_info("performing preprocessing with leading block size %zd...", m0);

  // preprocessing
  timer.start();
  Array<value_type>        s, t;
  BiPermMatrix<index_type> p, q;
#ifndef PSMILU_DISABLE_PRE
  size_type m;
  if (!IsSymm) {
    if (opts.pre_reorder != REORDER_OFF && m0 == A.nrows()) {
      if (!opts.pre_reorder_lvl1 || cur_level == 1u)
        m = do_preprocessing2(A_ccs, A_crs, opts, cur_level, s, t, p, q,
                              opts.saddle);
      else
        m = do_preprocessing<false>(A_ccs, A_crs, m0, opts, cur_level, s, t, p,
                                    q, opts.saddle);
    } else {
      if (opts.pre_reorder != REORDER_OFF)
        psmilu_warning("pre-reordering is not available for PS systems");
      m = do_preprocessing<false>(A_ccs, A_crs, m0, opts, cur_level, s, t, p, q,
                                  opts.saddle);
    }
  } else
    m = do_preprocessing<true>(A_ccs, A_crs, m0, opts, cur_level, s, t, p, q,
                               opts.saddle);
    // m = defer_dense_tail(A_crs, A_ccs, p, q, m);
#else
  s.resize(m0);
  psmilu_error_if(s.status() == DATA_UNDEF, "memory allocation failed");
  t.resize(m0);
  psmilu_error_if(t.status() == DATA_UNDEF, "memory allocation failed");
  p.resize(m0);
  q.resize(m0);
  std::fill(s.begin(), s.end(), value_type(1));
  std::fill(t.begin(), t.end(), value_type(1));
  p.make_eye();
  q.make_eye();
  size_type m(m0);
#endif             // PSMILU_DISABLE_PRE
  timer.finish();  // prefile pre-processing

  if (psmilu_verbose(INFO, opts)) {
    psmilu_info("preprocessing done with leading block size %zd...", m);
    psmilu_info("time: %gs", timer.time());
  }

#ifdef PSMILU_SAVE_FIRST_LEVEL_PERM_A
  if (cur_level == 1u) {
    std::mt19937                       eng(std::time(0));
    std::uniform_int_distribution<int> d(1000, 1000000);
    const std::string fname  = "Perm_A_" + std::to_string(d(eng)) + ".psmilu";
    auto              A_perm = A_crs.compute_perm(p(), q.inv(), m);
    Array<value_type> s2(m), t2(m);
    for (size_type i = 0; i < m; ++i) {
      s2[i] = s[p[i]];
      t2[i] = t[q[i]];
    }
    A_perm.scale_diag_left(s2);
    A_perm.scale_diag_right(t2);
    psmilu_info("\nsaving first level permutated matrix to file %s\n",
                fname.c_str());
    A_perm.write_native_bin(fname.c_str(), IsSymm ? m : size_type(0));
  }
#endif

  if (psmilu_verbose(INFO, opts)) psmilu_info("preparing data variables...");

  timer.start();

  // extract diagonal for the first entry
  auto d = internal::extract_perm_diag(s, A_ccs, t, m, p, q, 1);

#ifndef NDEBUG
#  define _GET_MAX_MIN_MINABS(__v, __m)                                 \
    const auto max_##__v =                                              \
        *std::max_element(__v.cbegin(), __v.cbegin() + __m),            \
               min_##__v =                                              \
                   *std::min_element(__v.cbegin(), __v.cbegin() + __m), \
               min_abs_##__v = std::abs(*std::min_element(              \
                   __v.cbegin(), __v.cbegin() + __m,                    \
                   [](const value_type a, const value_type b) {         \
                     return std::abs(a) < std::abs(b);                  \
                   }))
#  define _SHOW_MAX_MIN_MINABS(__v)                                         \
    psmilu_info("\t" #__v " max=%g, min=%g, min_abs=%g", (double)max_##__v, \
                (double)min_##__v, (double)min_abs_##__v)
  if (psmilu_verbose(INFO, opts)) {
    _GET_MAX_MIN_MINABS(d, m);
    _SHOW_MAX_MIN_MINABS(d);
    _GET_MAX_MIN_MINABS(s, m);
    _SHOW_MAX_MIN_MINABS(s);
    _GET_MAX_MIN_MINABS(t, m);
    _SHOW_MAX_MIN_MINABS(t);
  }
#endif

  // create U storage with deferred
  aug_crs_type U(m, A.ncols() * 2);
  psmilu_error_if(U.row_start().status() == DATA_UNDEF,
                  "memory allocation failed for U:row_start at level %zd.",
                  cur_level);
  do {
    const size_type rsv_fac =
        PSMILU_RESERVE_FAC <= 0 ? opts.alpha_U : PSMILU_RESERVE_FAC;
    U.reserve(A.nnz() * rsv_fac);
    psmilu_error_if(
        U.col_ind().status() == DATA_UNDEF || U.vals().status() == DATA_UNDEF,
        "memory allocation failed for U-nnz arrays at level %zd.", cur_level);
  } while (false);

  // create L storage with deferred
  aug_ccs_type L(A.nrows() * 2, m);
  psmilu_error_if(L.col_start().status() == DATA_UNDEF,
                  "memory allocation failed for L:col_start at level %zd.",
                  cur_level);
  do {
    const size_type rsv_fac =
        PSMILU_RESERVE_FAC <= 0 ? opts.alpha_L : PSMILU_RESERVE_FAC;
    L.reserve(A.nnz() * rsv_fac);
    psmilu_error_if(
        L.row_ind().status() == DATA_UNDEF || L.vals().status() == DATA_UNDEF,
        "memory allocation failed for L-nnz arrays at level %zd.", cur_level);
  } while (false);

  // create l and ut buffer
  SparseVector<value_type, index_type, ONE_BASED> l(A.nrows() * 2),
      ut(A.ncols() * 2);

  // create buffer for L and U start
  Array<index_type> L_start(m), U_start(m), start_bak(m);
  psmilu_error_if(
      L_start.status() == DATA_UNDEF || U_start.status() == DATA_UNDEF ||
          start_bak.status() == DATA_UNDEF,
      "memory allocation failed for L_start and/or U_start at level %zd.",
      cur_level);

  // create storage for kappa's
  Array<value_type> kappa_l(m), kappa_ut(m);
  psmilu_error_if(
      kappa_l.status() == DATA_UNDEF || kappa_ut.status() == DATA_UNDEF,
      "memory allocation failed for kappa_l and/or kappa_ut at level %zd.",
      cur_level);

  U.begin_assemble_rows();
  L.begin_assemble_cols();

  // determine parameters
  double tau_d, tau_kappa, tau_U, tau_L;
  int    alpha_L, alpha_U;
  std::tie(tau_d, tau_kappa, tau_L, tau_U, alpha_L, alpha_U) =
      determine_fac_pars(opts, cur_level);

  // Removing bounding the large diagonal values
  const auto is_bad_diag = [=](const value_type a) -> bool {
    return std::abs(1. / a) > tau_d;  // || std::abs(a) > tau_d;
  };

  const size_type m2(m), n(A.nrows());

  // deferred permutations
  Array<index_type> P(n * 2), Q(n * 2);
  psmilu_error_if(P.status() == DATA_UNDEF || Q.status() == DATA_UNDEF,
                  "memory allocation failed for P and/or Q at level %zd",
                  cur_level);
  std::copy_n(p().cbegin(), n, P.begin());
  std::copy_n(q().cbegin(), n, Q.begin());
  auto &P_inv = p.inv(), &Q_inv = q.inv();

  // information counter
  index_type info_counter[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};

  if (psmilu_verbose(INFO, opts)) psmilu_info("start Crout update...");
  PivotCrout step;
  for (; step < m; ++step) {
    Crout_info(" Crout step %zd, leading block size %zd", step, m);

    const size_type m_prev(m), defers_prev(step.defers());
    value_type      k_ut, k_l;

    for (;;) {
      // compute kappa for u wrp deferred index
      update_kappa_ut(step, U, kappa_ut, step.deferred_step());
      // then compute kappa for l
      update_kappa_l<false>(step, L, kappa_ut, kappa_l, step.deferred_step());
      // flag to indicate whether or not we shall do deferring
      bool do_defer = false;
      do {
        k_ut = std::abs(kappa_ut[step]);
        k_l  = std::abs(kappa_l[step]);
        if ((k_ut > tau_kappa && k_l > tau_kappa) ||
            (is_bad_diag(d[step]) && IsSymm)) {
          do_defer = true;
          ++info_counter[0];
          break;
        }
        if (k_l > tau_kappa ||
            U.nnz_in_col(step.deferred_step()) <
                L.nnz_in_row(step.deferred_step()) ||
            (U.nnz_in_col(step.deferred_step()) ==
                 L.nnz_in_row(step.deferred_step()) &&
             k_l >= k_ut)) {
          // if (k_l >= k_ut) {
          // we can safely compute the L
          // backup start/end/counts
          const size_type str_bak    = L.row_start()[step],
                          end_bak    = L.row_end()[step],
                          counts_bak = L.row_counts()[step];
          // then update L
          size_type nbaks;
          step.update_L_start_and_compress_L_wbak(L, L_start, start_bak, nbaks);
          // compute Lk
          step.compute_l_diag(s, A_ccs, t, P_inv, Q[step.deferred_step()], L,
                              L_start, d, U, l, d[step]);
          // check if "bad thing" happens
          if (k_l > tau_kappa || is_bad_diag(d[step])) {
            // size_type pivot = static_cast<size_type>(-1), sz(n);
            // for (index_type i(0); i < l.size(); ++i) {
            //   const size_type pvt = l.c_idx(i);
            //   // we want to ignore the deferals
            //   if (pvt < n)
            //     if (!is_bad_diag(l.val(i)) && L.nnz_in_row(pvt) < sz) {
            //       // check inverse conditioning
            //       update_kappa_l<false>(step, L, kappa_ut, kappa_l, pvt);
            //       if (std::abs(kappa_l[step]) <= tau_kappa) {
            //         sz    = L.nnz_in_row(pvt);
            //         pivot = pvt;
            //         k_l   = std::abs(kappa_l[step]);
            //       }
            //     }
            value_type tmp(std::abs(d[step]));
            size_type  pivot = static_cast<size_type>(-1);
            for (index_type i(0); i < l.size(); ++i) {
              const size_type pvt = l.c_idx(i);
              // we want to ignore the deferals
              if (pvt < n)
                if (tmp < std::abs(l.val(i))) {
                  // check inverse conditioning
                  update_kappa_l<false>(step, L, kappa_ut, kappa_l, pvt);
                  if (std::abs(kappa_l[step]) <= tau_kappa) {
                    tmp   = std::abs(l.val(i));
                    pivot = pvt;
                    k_l   = std::abs(kappa_l[step]);
                  }
                }
            }  // for
            if (pivot == static_cast<size_type>(-1) && k_l <= tau_kappa &&
                std::abs(1. / d[step]) <= 100.0)
              pivot = step;
            if (pivot != static_cast<size_type>(-1)) {
              if (pivot != step) {
                // swap p
                std::swap(P[step.deferred_step()], P[pivot]);
                std::swap(P_inv[P[step.deferred_step()]], P_inv[P[pivot]]);

                // swap L
                L.interchange_rows(step, pivot);
                // adjust L_start
                internal::adjust_start_pos(step, L, pivot, L_start);
                // swap l by interchange it with diagonal
                std::swap(d[step], l.vals()[pivot]);
              }
              ++info_counter[1];
            } else {
              // restore L
              step.uncompress_L(L, str_bak, end_bak, counts_bak);
              // restore sparse vector
              l.restore_cur_state();
              // restore L_start
              for (size_type i(0); i < nbaks; ++i) --L_start[start_bak[i]];
              do_defer = true;
              ++info_counter[2];
              break;  // do-while
            }
          }  // if bad thing

          // handle U part

          // update U
          step.update_U_start_and_compress_U(U, U_start);
          // compute Uk'
          step.compute_ut(s, A_crs, t, P[step.deferred_step()], Q_inv, L, d, U,
                          U_start, ut);
        } else {
          // backup start/end/counts
          const size_type str_bak    = U.col_start()[step],
                          end_bak    = U.col_end()[step],
                          counts_bak = U.col_counts()[step];
          // update U
          size_type nbaks;
          step.update_U_start_and_compress_U_wbak(U, U_start, start_bak, nbaks);
          // compute Uk'
          step.compute_ut_diag(s, A_crs, t, P[step.deferred_step()], Q_inv, L,
                               d, U, U_start, ut, d[step]);
          // check if "bad thing" happens
          if (k_ut > tau_kappa || is_bad_diag(d[step])) {
            // size_type pivot = static_cast<size_type>(-1), sz(n);
            // for (index_type i(0); i < ut.size(); ++i) {
            //   const size_type pvt = ut.c_idx(i);
            //   if (pvt < n)
            //     if (!is_bad_diag(ut.val(i)) && U.nnz_in_col(pvt) < sz) {
            //       // check inverse conditioning
            //       update_kappa_ut(step, U, kappa_ut, pvt);
            //       if (std::abs(kappa_ut[step]) <= tau_kappa) {
            //         sz    = U.nnz_in_col(pvt);
            //         pivot = pvt;
            //         k_ut  = std::abs(kappa_ut[step]);
            //       }
            //     }
            value_type tmp(std::abs(d[step]));
            size_type  pivot = static_cast<size_type>(-1);
            for (index_type i(0); i < ut.size(); ++i) {
              const size_type pvt = ut.c_idx(i);
              // we want to ignore the deferals
              if (pvt < n)
                if (tmp < std::abs(ut.val(i))) {
                  // check inverse conditioning
                  update_kappa_ut(step, U, kappa_ut, pvt);
                  if (std::abs(kappa_ut[step]) <= tau_kappa) {
                    tmp   = std::abs(ut.val(i));
                    pivot = pvt;
                    k_ut  = std::abs(kappa_ut[step]);
                  }
                }
            }  // for
            if (pivot == static_cast<size_type>(-1) && k_ut <= tau_kappa &&
                std::abs(1. / d[step]) <= 100.0)
              pivot = step;
            if (pivot != static_cast<size_type>(-1)) {
              if (pivot != step) {
                // swap q
                std::swap(Q[step.deferred_step()], Q[pivot]);
                std::swap(Q_inv[Q[step.deferred_step()]], Q_inv[Q[pivot]]);
                // interchange U
                U.interchange_cols(step, pivot);
                // adjust U_start
                internal::adjust_start_pos(step, U, pivot, U_start);
                // swap ut with diagonal
                std::swap(d[step], ut.vals()[pivot]);
              }
              ++info_counter[3];
            } else {
              // restore U
              step.uncompress_U(U, str_bak, end_bak, counts_bak);
              // restore sparse vector
              ut.restore_cur_state();
              // restore U_start
              for (size_type i(0); i < nbaks; ++i) --U_start[start_bak[i]];
              do_defer = true;
              ++info_counter[4];
              break;  // do-while
            }
          }  // if bad thing

          // handle L part

          // then update L
          step.update_L_start_and_compress_L<false>(L, m2, L_start);
          // compute Lk
          step.compute_l<false>(s, A_ccs, t, P_inv, Q[step.deferred_step()], m2,
                                L, L_start, d, U, l);
        }
      } while (false);
      // if we do deferring
      if (do_defer) {
        --m;
        const auto tail_pos = n + step.defers();
        U.defer_col(step.deferred_step(), tail_pos);
        L.defer_row(step.deferred_step(), tail_pos);
        P[tail_pos]                    = P[step.deferred_step()];
        Q[tail_pos]                    = Q[step.deferred_step()];
        P_inv[P[step.deferred_step()]] = tail_pos;
        Q_inv[Q[step.deferred_step()]] = tail_pos;
        // mark as empty entries
        P[step.deferred_step()] = Q[step.deferred_step()] = -1;
        step.increment_defer_counter();  // increment defers here
        // handle the last step
        if (step.deferred_step() >= m2) m = step;
        if (step == m) break;
        continue;
      }
      break;
    }  // inf loop
    if (step == m) break;

    // compress permutation vectors
    step.assign_gap_array(P, p);
    step.assign_gap_array(Q, q);

#ifndef NDEBUG
    const bool u_is_nonsingular =
#else
    (void)
#endif
        step.scale_inv_diag(d, ut);
    psmilu_assert(!u_is_nonsingular, "u is singular at level %zd step %zd",
                  cur_level, step);
#ifndef NDEBUG
    const bool l_is_nonsingular =
#else
    (void)
#endif
        step.scale_inv_diag(d, l);
    psmilu_assert(!l_is_nonsingular, "l is singular at level %zd step %zd",
                  cur_level, step);

    //----------------
    // inverse thres
    //----------------

    Crout_info("  kappa_ut=%g, kappa_l=%g", (double)k_ut, (double)k_l);
    Crout_info(
        "  previous/current leading block sizes %zd/%zd, local/total "
        "defers=%zd/%zd",
        m_prev, m, step.defers() - defers_prev, step.defers());

    const auto kappa_sq = k_ut * k_l;

    //---------------
    // drop and sort
    //---------------

    const size_type ori_ut_size = ut.size(), ori_l_size = l.size();

    apply_num_dropping(tau_U, kappa_sq, ut);
#ifndef PSMILU_DISABLE_SPACE_DROP
    const size_type n_ut = ut.size();
    apply_space_dropping(row_sizes[p[step]], alpha_U, ut);
    info_counter[5] += n_ut - ut.size();
#endif  // PSMILU_DISABLE_SPACE_DROP
    info_counter[6] += ori_ut_size - ut.size();
    ut.sort_indices();

    // push back rows to U
    U.push_back_row(step, ut.inds().cbegin(), ut.inds().cbegin() + ut.size(),
                    ut.vals());

    Crout_info("  ut sizes before/after dropping %zd/%zd, drops=%zd",
               ori_ut_size, ut.size(), ori_ut_size - ut.size());

    // apply numerical dropping on L
    apply_num_dropping(tau_L, kappa_sq, l);
#ifndef PSMILU_DISABLE_SPACE_DROP
    const size_type n_l = l.size();
    apply_space_dropping(col_sizes[q[step]], alpha_L, l);
    info_counter[7] += n_l - l.size();
#endif  // PSMILU_DISABLE_SPACE_DROP
    info_counter[8] += ori_l_size - l.size();
    l.sort_indices();

    Crout_info("  l sizes before/after dropping %zd/%zd, drops=%zd", ori_l_size,
               l.size(), ori_l_size - l.size());

    L.push_back_col(step, l.inds().cbegin(), l.inds().cbegin() + l.size(),
                    l.vals());

    Crout_info(" Crout step %zd done!", step);
  }

  // compress permutation vectors
  for (; step < n; ++step) {
    step.assign_gap_array(P, p);
    step.assign_gap_array(Q, q);
    // NOTE important to compress the start/end in augmented DS
    // the tails were updated while deferring and the leading
    // block is compressed while updating L and U starts
    // thus, we only need to compress the offsets
    step.compress_array(U.col_start());
    step.compress_array(U.col_end());
    step.compress_array(U.col_counts());
    step.compress_array(L.row_start());
    step.compress_array(L.row_end());
    step.compress_array(L.row_counts());
  }
  // rebuild the inverse mappings
  p.build_inv();
  q.build_inv();

  U.end_assemble_rows();
  L.end_assemble_cols();

  // finalize start positions
  if (m) {
    U_start[m - 1] = U.row_start()[m - 1];
    L_start[m - 1] = L.col_start()[m - 1];
  }
  // compress tails
  internal::compress_tails(U, L, U_start, L_start, m, step.defers());

  timer.finish();  // profile Crout update

  // now we are done
  if (psmilu_verbose(INFO, opts)) {
    psmilu_info(
        "finish Crout update...\n"
        "\ttotal defers=%zd\n"
        "\tleading block size in=%zd\n"
        "\tleading block size out=%zd\n"
        "\tdiff=%zd\n"
        "\tstrict defers=%zd\n"
        "\trow pivots=%zd\n"
        "\tfallback row defers=%zd\n"
        "\tcol pivots=%zd\n"
        "\tfallback col defers=%zd\n"
        "\tdrop ut=%zd\n"
        "\tspace drop ut=%zd\n"
        "\tdrop l=%zd\n"
        "\tspace drop l=%zd",
        step.defers(), m2, m, m2 - m, (size_type)info_counter[0],
        (size_type)info_counter[1], (size_type)info_counter[2],
        (size_type)info_counter[3], (size_type)info_counter[4],
        (size_type)info_counter[5], (size_type)info_counter[6],
        (size_type)info_counter[7], (size_type)info_counter[8]);
#ifndef NDEBUG
    _GET_MAX_MIN_MINABS(d, m);
    _SHOW_MAX_MIN_MINABS(d);
#endif
    psmilu_info("time: %gs", timer.time());
  }

  if (psmilu_verbose(INFO, opts))
    psmilu_info("computing Schur complement and assembling Prec...");

  timer.start();

  // drop
  auto     E   = crs_type(internal::extract_E(s, A_crs, t, m, p, q));
  auto     F   = internal::extract_F(s, A_ccs, t, m, p, q, ut.vals());
  auto     L_E = L.template split_crs<true>(m, L_start);
  crs_type U_F;
  do {
    auto            U_F2 = U.template split_ccs<true>(m, U_start);
    const size_type nnz1 = L_E.nnz(), nnz2 = U_F2.nnz();
#ifndef PSMILU_NO_DROP_LE_UF
    const double a_L = opts.alpha_L, a_U = opts.alpha_U;
    if (psmilu_verbose(INFO, opts))
      psmilu_info("applying dropping on L_E and U_F with alpha_{L,U}=%g,%g...",
                  a_L, a_U);
    if (m < n) {
#  ifdef PSMILU_USE_CUR_SIZES
      drop_L_E(E.row_start(), a_L, L_E, l.vals(), l.inds());
      drop_U_F(F.col_start(), a_U, U_F2, ut.vals(), ut.inds());
#  else
      if (cur_level == 1u) {
        drop_L_E(E.row_start(), a_L, L_E, l.vals(), l.inds());
        drop_U_F(F.col_start(), a_U, U_F2, ut.vals(), ut.inds());
      } else {
        // use P and Q as buffers
        P[0] = Q[0] = 0;
        for (size_type i(m); i < n; ++i) {
          P[i - m + 1] = P[i - m] + row_sizes[p[i]];
          Q[i - m + 1] = Q[i - m] + col_sizes[q[i]];
        }
        drop_L_E(P, a_L, L_E, l.vals(), l.inds());
        drop_U_F(Q, a_U, U_F2, ut.vals(), ut.inds());
      }
#  endif
    }
#endif  // PSMILU_NO_DROP_LE_UF
    U_F = crs_type(U_F2);
    if (psmilu_verbose(INFO, opts))
      psmilu_info("nnz(L_E)=%zd/%zd, nnz(U_F)=%zd/%zd...", nnz1, L_E.nnz(),
                  nnz2, U_F.nnz());
  } while (false);

  // compute the nnz(A)-nnz(B) for first level
  size_type AmB_nnz(0);
  for (size_type i(m); i < n; ++i) AmB_nnz += row_sizes[p[i]] + col_sizes[q[i]];

  // compute S version of Schur complement
  bool             use_h_ver = false;
  const input_type S =
      input_type(compute_Schur_simple(s, A_crs, t, p, q, m, L_E, d, U_F, l));

  // compute L_B and U_B
  auto L_B = L.template split<false>(m, L_start);
  auto U_B = U.template split_ccs<false>(m, U_start);

  const size_type dense_thres1 = static_cast<size_type>(
                      std::max(opts.alpha_L, opts.alpha_U) * AmB_nnz),
                  dense_thres2 = std::max(static_cast<size_type>(std::ceil(
                                              opts.c_d * std::cbrt(N))),
                                          size_type(1000));

  if (psmilu_verbose(INFO, opts))
    psmilu_info(
        "nnz(S_C)=%zd, nnz(L/L_B)=%zd/%zd, nnz(U/U_B)=%zd/%zd\n"
        "dense_thres{1,2}=%zd/%zd...",
        S.nnz(), L.nnz(), L_B.nnz(), U.nnz(), U_B.nnz(), dense_thres1,
        dense_thres2);

  // test H version
  const size_type nm = n - m;
  dense_type      S_D;
  psmilu_assert(S_D.empty(), "fatal!");

  if ((size_type)std::ceil(nm * nm * opts.rho) <= dense_thres1 ||
      nm <= dense_thres2 || !m) {
    S_D = dense_type::from_sparse(S);
    if (psmilu_verbose(INFO, opts))
      psmilu_info("converted Schur complement (%s) to dense for last level...",
                  (use_h_ver ? "H" : "S"));
  }

#ifndef PSMILU_USE_CUR_SIZES
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
#endif  // PSMILU_USE_CUR_SIZES

  precs.emplace_back(m, n, std::move(L_B), std::move(d), std::move(U_B),
                     std::move(E), std::move(F), std::move(s), std::move(t),
                     std::move(p()), std::move(q.inv()));

  // if dense is not empty, then push it back
  if (!S_D.empty()) {
    auto &last_level = precs.back().dense_solver;
    last_level.set_matrix(std::move(S_D));
    last_level.factorize();
    if (psmilu_verbose(INFO, opts))
      psmilu_info("successfully factorized the dense component...");
  }

  timer.finish();  // profile post-processing

  if (psmilu_verbose(INFO, opts)) psmilu_info("time: %gs", timer.time());

  if (psmilu_verbose(INFO, opts)) psmilu_info("\nfinish level %zd.", cur_level);

  return S;
}

}  // namespace psmilu

#endif  // _PSMILU_FACPVT_HPP
