//@HEADER
//----------------------------------------------------------------------------
//                Copyright (C) 2019 The PSMILU AUTHORS
//----------------------------------------------------------------------------
//@HEADER

/// \file psmilu_fac_defer.hpp
/// \brief Implementation of incomplete multilevel deferred factorization
/// \authors Qiao,

#ifndef _PSMILU_FACDEFERRED_HPP
#define _PSMILU_FACDEFERRED_HPP

#include "psmilu_fac.hpp"

#include "psmilu_DeferredCrout.hpp"

namespace psmilu {
namespace internal {

template <class L_AugType, class U_AugType, class PosArray>
inline void compress_tails(U_AugType &U, L_AugType &L, const PosArray &U_start,
                           const PosArray &                   L_start,
                           const typename PosArray::size_type m,
                           const typename PosArray::size_type dfrs) {
  using size_type  = typename PosArray::size_type;
  using index_type = typename L_AugType::index_type;

  if (dfrs) {
    const auto comp_index = [=](index_type &j) { j -= dfrs; };
    auto       U_first = U.col_ind().begin(), L_first = L.row_ind().begin();
    for (size_type i(0); i < m; ++i) {
      std::for_each(U_first + U_start[i], U.col_ind_end(i), comp_index);
      std::for_each(L_first + L_start[i], L.row_ind_end(i), comp_index);
    }
  }

  L.resize_nrows(L.nrows() / 2);
  U.resize_ncols(U.ncols() / 2);

#ifndef NDEBUG
  L.check_validity();
  U.check_validity();
#endif
}

template <class L_AugCcsType, class PosArray>
inline void search_back_start_symm(const L_AugCcsType &               L,
                                   const typename PosArray::size_type back_step,
                                   const typename PosArray::size_type m,
                                   PosArray &                         L_start) {
  using index_type  = typename L_AugCcsType::index_type;
  index_type aug_id = L.start_row_id(back_step);
  while (!L.is_nil(aug_id)) {
    const auto col_idx = L.col_idx(aug_id);
    --L_start[col_idx];
    aug_id = L.next_row_id(aug_id);
  }
}

}  // namespace internal

template <bool IsSymm, class CsType, class CroutStreamer, class PrecsType>
inline CsType iludp_factor_defer(const CsType &                   A,
                                 const typename CsType::size_type m0,
                                 const typename CsType::size_type N,
                                 const Options &                  opts,
                                 const CroutStreamer &            Crout_info,
                                 PrecsType &                      precs) {
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

  // TODO put this into control parameters or function arg
  constexpr static bool check_zero_diag = true;

  psmilu_error_if(A.nrows() != A.ncols(), "only squared systems are supported");

  psmilu_assert(m0 <= std::min(A.nrows(), A.ncols()),
                "leading size should be smaller than size of A");
  const size_type cur_level = precs.size() + 1;
#ifndef NDEBUG
  if (IsSymm)
    psmilu_error_if(cur_level != 1u,
                    "symmetric must be applied to first level!");
#endif

  if (psmilu_verbose(INFO, opts))
    psmilu_info("\nenter level %zd.\n", cur_level);

  DefaultTimer timer;

  // build counterpart type
  const other_type A_counterpart(A);

  // now use our trait and its static methods to precisely determine the ccs
  // and crs components.
  const crs_type &A_crs = cs_trait::select_crs(A, A_counterpart);
  const ccs_type &A_ccs = cs_trait::select_ccs(A, A_counterpart);

  if (psmilu_verbose(INFO, opts))
    psmilu_info("performing preprocessing with leading block size %zd...", m0);

  // preprocessing
  timer.start();
  Array<value_type>        s, t;
  BiPermMatrix<index_type> p, q;
#ifndef PSMILU_DISABLE_PRE
  size_type m =
      do_preprocessing<IsSymm>(A_ccs, m0, opts, s, t, p, q, check_zero_diag);
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

  // extract diagonal
  auto d = internal::extract_perm_diag(s, A_ccs, t, m, p, q);

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
  Array<index_type> L_start(m), U_start(m);
  psmilu_error_if(
      L_start.status() == DATA_UNDEF || U_start.status() == DATA_UNDEF,
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

  // localize parameters
  // const auto tau_d = opts.tau_d, tau_kappa = opts.tau_kappa, tau_U =
  // opts.tau_U,
  //            tau_L   = opts.tau_L;
  // const auto alpha_L = opts.alpha_L, alpha_U = opts.alpha_U;
  DETERMINE_LEVEL_PARS(tau_d, tau_kappa, tau_U, tau_L, alpha_L, alpha_U, opts,
                       cur_level);

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

  // from L/U index to deferred fac mapping
  // Array<index_type> ori2def(n);
  // psmilu_error_if(ori2def.status() == DATA_UNDEF,
  //                 "memory allocation failed for ori2def at level %zd",
  //                 cur_level);
  // for (size_type i(0); i < n; ++i) ori2def[i] = i;  // build identity

  // 0 for defer due to diagonal, 1 for defer due to bad inverse conditioning
  index_type info_counter[] = {0, 0, 0};

  if (psmilu_verbose(INFO, opts)) psmilu_info("start Crout update...");
  DeferredCrout step;
  for (; step < m; ++step) {
    // first check diagonal
    bool            pvt         = is_bad_diag(d[step.deferred_step()]);
    const size_type m_prev      = m;
    const size_type defers_prev = step.defers();
    info_counter[0] += pvt;

    Crout_info(" Crout step %zd, leading block size %zd", step, m);

    // compute kappa for u wrp deferred index
    update_kappa_ut(step, U, kappa_ut, step.deferred_step());
    // then compute kappa for l
    update_kappa_l<IsSymm>(step, L, kappa_ut, kappa_l, step.deferred_step());

#ifdef PSMILU_DEFERREDFAC_VERBOSE_STAT
    info_counter[2] += (std::abs(kappa_ut[step]) > tau_kappa &&
                        std::abs(kappa_l[step]) > tau_kappa);
#endif  // PSMILU_DEFERREDFAC_VERBOSE_STAT

    // check condition number if diagonal doesn't satisfy
    if (!pvt) {
      pvt = std::abs(kappa_ut[step]) > tau_kappa ||
            std::abs(kappa_l[step]) > tau_kappa;
      info_counter[1] += pvt;
    }

    if (pvt) {
      while (m > step) {
        --m;
        const auto tail_pos = n + step.defers();
        U.defer_col(step.deferred_step(), tail_pos);
        L.defer_row(step.deferred_step(), tail_pos);
        if (IsSymm) internal::search_back_start_symm(L, tail_pos, m2, L_start);
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
#ifdef PSMILU_DEFERREDFAC_VERBOSE_STAT
          // compute kappa for u wrp deferred index
          update_kappa_ut(step, U, kappa_ut, step.deferred_step());
          // then compute kappa for l
          update_kappa_l<IsSymm>(step, L, kappa_ut, kappa_l,
                                 step.deferred_step());
          info_counter[2] += (std::abs(kappa_ut[step]) > tau_kappa &&
                              std::abs(kappa_l[step]) > tau_kappa);
#endif  // PSMILU_DEFERREDFAC_VERBOSE_STAT
          continue;
        }

        // compute kappa for u wrp deferred index
        update_kappa_ut(step, U, kappa_ut, step.deferred_step());
        // then compute kappa for l
        update_kappa_l<IsSymm>(step, L, kappa_ut, kappa_l,
                               step.deferred_step());
        pvt = std::abs(kappa_ut[step]) > tau_kappa ||
              std::abs(kappa_l[step]) > tau_kappa;
        if (pvt) {
          ++info_counter[1];
#ifdef PSMILU_DEFERREDFAC_VERBOSE_STAT
          info_counter[2] += (std::abs(kappa_ut[step]) > tau_kappa &&
                              std::abs(kappa_l[step]) > tau_kappa);
#endif  // PSMILU_DEFERREDFAC_VERBOSE_STAT
          continue;
        }
        break;
      }                      // while
      if (m == step) break;  // break for
    }

    //----------------
    // inverse thres
    //----------------

    const auto k_ut = std::abs(kappa_ut[step]), k_l = std::abs(kappa_l[step]);

    // check pivoting
    psmilu_assert(!(k_ut > tau_kappa || k_l > tau_kappa), "should not happen!");

    Crout_info("  kappa_ut=%g, kappa_l=%g", (double)k_ut, (double)k_l);

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

    // update U
    step.update_U_start_and_compress_U(U, U_start);
    // then update L
    step.update_L_start_and_compress_L<IsSymm>(L, m2, L_start);

    //----------------------
    // compute Crout updates
    //----------------------

    // compute Uk'
    step.compute_ut(s, A_crs, t, p[step], Q_inv, L, d, U, U_start, ut);
    // compute Lk
    step.compute_l<IsSymm>(s, A_ccs, t, P_inv, q[step], m2, L, L_start, d, U,
                           l);

    // update diagonal entries for u first
#ifndef NDEBUG
    const bool u_is_nonsingular =
#else
    (void)
#endif
        step.scale_inv_diag(d, ut);
    psmilu_assert(!u_is_nonsingular, "u is singular at level %zd step %zd",
                  cur_level, step);

    // update diagonals b4 dropping
    step.update_B_diag<IsSymm>(l, ut, m2, d);
    // step.update_B_diag<IsSymm>(l, ut, n * 2, d2);

#ifndef NDEBUG
    const bool l_is_nonsingular =
#else
    (void)
#endif
        step.scale_inv_diag(d, l);
    psmilu_assert(!l_is_nonsingular, "l is singular at level %zd step %zd",
                  cur_level, step);

    //---------------
    // drop and sort
    //---------------

    const size_type ori_ut_size = ut.size(), ori_l_size = l.size();

    // apply drop for U
    apply_dropping_and_sort(tau_U, k_ut, A_crs.nnz_in_row(p[step]), alpha_U,
                            ut);
    // ut.sort_indices();

    // push back rows to U
    U.push_back_row(step, ut.inds().cbegin(), ut.inds().cbegin() + ut.size(),
                    ut.vals());

    Crout_info("  ut sizes before/after dropping %zd/%zd, drops=%zd",
               ori_ut_size, ut.size(), ori_ut_size - ut.size());

    if (IsSymm) {
      // for symmetric cases, we need first find the leading block size
      auto info = find_sorted(ut.inds().cbegin(),
                              ut.inds().cbegin() + ut.size(), m2 + ONE_BASED);
      apply_dropping_and_sort(tau_L, k_l, A_ccs.nnz_in_col(q[step]), alpha_L, l,
                              info.second - ut.inds().cbegin());

#ifndef NDEBUG
      if (info.second != ut.inds().cbegin() &&
          info.second != ut.inds().cbegin() + ut.size() && l.size())
        psmilu_error_if(*(info.second - 1) >= *l.inds().cbegin() ||
                            *(info.second - 1) - ONE_BASED >= m2,
                        "l contains symm part (%zd,%zd,%zd)",
                        (size_type)(*(info.second - 1)),
                        (size_type)*l.inds().cbegin(), m2);
#endif

      Crout_info(
          "  l sizes (asymm parts) before/after dropping %zd/%zd, drops=%zd",
          ori_l_size, l.size(), ori_l_size - l.size());

      // push back symmetric entries and offsets
      L.push_back_col(step, ut.inds().cbegin(), info.second, ut.vals(),
                      l.inds().cbegin(), l.inds().cbegin() + l.size(),
                      l.vals());
    } else {
      // for asymmetric cases, just do exactly the same things as ut
      apply_dropping_and_sort(tau_L, k_l, A_ccs.nnz_in_col(q[step]), alpha_L,
                              l);
      // l.sort_indices();

      Crout_info("  l sizes before/after dropping %zd/%zd, drops=%zd",
                 ori_l_size, l.size(), ori_l_size - l.size());

      L.push_back_col(step, l.inds().cbegin(), l.inds().cbegin() + l.size(),
                      l.vals());
    }

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
    step.compress_array(L.row_start());
    step.compress_array(L.row_end());
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
        "\tdiag defers=%zd\n"
        "\tinv-norm defers=%zd"
#ifdef PSMILU_DEFERREDFAC_VERBOSE_STAT
        "\n\tboth-inv-bad defers=%zd"
#endif  // PSMILU_DEFERREDFAC_VERBOSE_STAT
        ,
        step.defers(), m2, m, m2 - m, (size_type)info_counter[0],
        (size_type)info_counter[1]
#ifdef PSMILU_DEFERREDFAC_VERBOSE_STAT
        ,
        (size_type)info_counter[2]
#endif  // PSMILU_DEFERREDFAC_VERBOSE_STAT
    );
#ifndef NDEBUG
    _GET_MAX_MIN_MINABS(d, m);
    _SHOW_MAX_MIN_MINABS(d);
#endif
    psmilu_info("time: %gs", timer.time());
  }

  if (psmilu_verbose(INFO, opts))
    psmilu_info("computing Schur complement (C) and assembling Prec...");

  timer.start();

  // compute C version of Schur complement
  crs_type S_tmp;
  compute_Schur_C(s, A_crs, t, p, q, m, A.nrows(), L, d, U, U_start, S_tmp);
  const input_type S(S_tmp);  // if input==crs, then wrap, ow copy

  // compute L_B and U_B
  auto L_B = L.template split<false>(m, L_start);
  auto U_B = U.template split_ccs<false>(m, U_start);

  if (psmilu_verbose(INFO, opts))
    psmilu_info(
        "nnz(S_C)=%zd, nnz(L)=%zd, nnz(L_B)=%zd, nnz(U)=%zd, nnz(U_B)=%zd...",
        S.nnz(), L.nnz(), L_B.nnz(), U.nnz(), U_B.nnz());

  // test H version
  const size_type nm     = A.nrows() - m;
  const auto      cbrt_N = std::cbrt(N);
  dense_type      S_D;
  psmilu_assert(S_D.empty(), "fatal!");
  if (S.nnz() >= static_cast<size_type>(opts.rho * nm * nm) ||
      nm <= static_cast<size_type>(opts.c_d * cbrt_N) || !m) {
    bool use_h_ver = false;
    S_D            = dense_type::from_sparse(S);
    if (m <= static_cast<size_type>(opts.c_h * cbrt_N) && m) {
#ifdef PSMILU_UNIT_TESTING
      ccs_type T_E, T_F;
#endif
      compute_Schur_H(L, L_start, L_B, s, A_ccs, t, p, q, d, U_B, U, S_D
#ifdef PSMILU_UNIT_TESTING
                      ,
                      T_E, T_F
#endif
      );
      use_h_ver = true;
    }  // H version check
    if (psmilu_verbose(INFO, opts))
      psmilu_info("converted Schur complement (%s) to dense for last level...",
                  (use_h_ver ? "H" : "C"));
  }

  // NOTE that L_B/U_B are CCS, we need CRS, we can save computation with
  // symmetric case
  crs_type L_B2, U_B2;
  if (IsSymm) {
    L_B2.resize(m, m);
    U_B2.resize(m, m);
    L_B2.row_start() = std::move(U_B.col_start());
    U_B2.row_start() = std::move(L_B.col_start());
    L_B2.col_ind()   = std::move(U_B.row_ind());
    U_B2.col_ind()   = std::move(L_B.row_ind());
    L_B2.vals()      = std::move(U_B.vals());
    U_B2.vals()      = std::move(L_B.vals());
  } else {
    L_B2 = crs_type(L_B);
    U_B2 = crs_type(U_B);
  }
  precs.emplace_back(
      m, A.nrows(), std::move(L_B2), std::move(d), std::move(U_B2),
      crs_type(internal::extract_E(s, A_crs, t, m, p, q)),
      crs_type(internal::extract_F(s, A_ccs, t, m, p, q, ut.vals())),
      std::move(s), std::move(t), std::move(p()), std::move(q.inv()));

  // if dense is not empty, then push it back
  if (!S_D.empty()) {
    auto &last_level = precs.back().dense_solver;
    last_level.set_matrix(std::move(S_D));
    last_level.factorize();
    if (psmilu_verbose(INFO, opts))
      psmilu_info("successfully factorized the dense complement...");
  }
#ifndef NDEBUG
  else
    psmilu_error_if(!precs.back().dense_solver.empty(), "should be empty!");
#endif

  timer.finish();  // profile post-processing

  if (psmilu_verbose(INFO, opts)) psmilu_info("time: %gs", timer.time());

  if (psmilu_verbose(INFO, opts)) psmilu_info("\nfinish level %zd.", cur_level);

  return S;
}

}  // namespace psmilu

#endif  // _PSMILU_FACDEFERRED_HPP
