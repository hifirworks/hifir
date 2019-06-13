//@HEADER
//----------------------------------------------------------------------------
//                Copyright (C) 2019 The PSMILU AUTHORS
//----------------------------------------------------------------------------
//@HEADER

/// \file psmilu_fac_defer_thin.hpp
/// \brief Kernels for deferred factorization in thin data structure
/// \authors Qiao,

#ifndef _PSMILU_FACDEFERTHIN_HPP
#define _PSMILU_FACDEFERTHIN_HPP

#include "psmilu_DeferredCrout_thin.hpp"
#include "psmilu_fac_defer.hpp"

namespace psmilu {

template <bool IsSymm, class CsType, class CroutStreamer, class PrecsType,
          class IntArray>
inline CsType iludp_factor_defer_thin(const CsType &                   A,
                                      const typename CsType::size_type m0,
                                      const typename CsType::size_type N,
                                      const Options &                  opts,
                                      const CroutStreamer &Crout_info,
                                      PrecsType &precs, IntArray &row_sizes,
                                      IntArray &col_sizes) {
  typedef CsType                      input_type;
  typedef typename CsType::other_type other_type;
  using cs_trait = internal::CompressedTypeTrait<input_type, other_type>;
  typedef typename cs_trait::crs_type crs_type;
  typedef typename cs_trait::ccs_type ccs_type;
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
  size_type m = do_preprocessing<IsSymm>(A_ccs, A_crs, m0, opts, cur_level, s,
                                         t, p, q, opts.saddle);
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
  crs_type U(m, A.ncols() * 2);
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
  ccs_type L(A.nrows() * 2, m);
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

  // create buffer for L and U starts
  Array<index_type> L_start(m), U_start(m);
  psmilu_error_if(
      L_start.status() == DATA_UNDEF || U_start.status() == DATA_UNDEF,
      "memory allocation failed for L_start and/or U_start at level %zd.",
      cur_level);

  Array<index_type> L_offsets;
  if (IsSymm) {
    L_offsets.resize(m);
    psmilu_error_if(L_offsets.status() == DATA_UNDEF,
                    "memory allocation failed for L_offsets at level %zd.",
                    cur_level);
  }

  const Array<index_type> &Crout_L_start = !IsSymm ? L_start : L_offsets;

  // create buffer for L and U lists
  Array<index_type> L_list(m * 2), U_list(m * 2);
  psmilu_error_if(
      L_list.status() == DATA_UNDEF || U_list.status() == DATA_UNDEF,
      "memory allocation failed for L_list and/or U_list at level %zd.",
      cur_level);

  // set default value
  std::fill(L_list.begin(), L_list.end(), static_cast<index_type>(-1));
  std::fill(U_list.begin(), U_list.end(), static_cast<index_type>(-1));

  // create storage for kappa's
  Array<value_type> kappa_l(m), kappa_ut(m);
  psmilu_error_if(
      kappa_l.status() == DATA_UNDEF || kappa_ut.status() == DATA_UNDEF,
      "memory allocation failed for kappa_l and/or kappa_ut at level %zd.",
      cur_level);

#ifdef PSMILU_ENABLE_NORM_STAT
  Array<value_type> ut_norm_ratios(m), l_norm_ratios(m);
#endif  // PSMILU_ENABLE_NORM_STAT

  U.begin_assemble_rows();
  L.begin_assemble_cols();

  // localize parameters
  double tau_d, tau_kappa, tau_L, tau_U;
  int    alpha_L, alpha_U;
  std::tie(tau_d, tau_kappa, tau_L, tau_U, alpha_L, alpha_U) =
      determine_fac_pars(opts, cur_level);
  const auto kappa_sq = tau_kappa * tau_kappa;

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

  // 0 for defer due to diagonal, 1 for defer due to bad inverse conditioning
  index_type info_counter[] = {0, 0, 0, 0, 0, 0, 0};

  if (psmilu_verbose(INFO, opts)) psmilu_info("start Crout update...");
  DeferredCrout_thin step;
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

    // handle defer
    if (pvt) {
      while (m > step) {
        --m;
        const auto tail_pos = n + step.defers();
        step.defer_entry(tail_pos, U_start, U, U_list);
        if (!IsSymm)
          step.defer_entry(tail_pos, L_start, L, L_list);
        else
          step.defer_L_and_fix_offsets(tail_pos, L_start, L, L_list, L_offsets);
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
          step.update_kappa(U, U_list, U_start, kappa_ut);
          // then compute kappa for l
          if (!IsSymm)
            step.update_kappa(L, L_list, L_start, kappa_l);
          else
            kappa_l[step] = kappa_ut[step];
          info_counter[2] += (std::abs(kappa_ut[step]) > tau_kappa &&
                              std::abs(kappa_l[step]) > tau_kappa);
#endif  // PSMILU_DEFERREDFAC_VERBOSE_STAT
          continue;
        }
        // compute kappa for u wrp deferred index
        step.update_kappa(U, U_list, U_start, kappa_ut);
        // then compute kappa for l
        if (!IsSymm)
          step.update_kappa(L, L_list, L_start, kappa_l);
        else
          kappa_l[step] = kappa_ut[step];
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

    const auto k_ut = kappa_ut[step], k_l = kappa_l[step];

    // check pivoting
    psmilu_assert(!(std::abs(k_ut) > tau_kappa || std::abs(k_l) > tau_kappa),
                  "should not happen!");

    Crout_info("  kappa_ut=%g, kappa_l=%g", (double)std::abs(k_ut),
               (double)std::abs(k_l));

    Crout_info(
        "  previous/current leading block sizes %zd/%zd, local/total "
        "defers=%zd/%zd",
        m_prev, m, step.defers() - defers_prev, step.defers());

#ifndef PSMILU_DISABLE_DYN_PVT_THRES
    const auto kappa_sq = std::abs(k_ut * k_l);
#endif  // PSMILU_DISABLE_DYN_PVT_THRES

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
    psmilu_assert(!u_is_nonsingular, "u is singular at level %zd step %zd",
                  cur_level, step);

    // update diagonals b4 dropping
    step.update_B_diag<IsSymm>(l, ut, m2, d);

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
    apply_num_dropping(tau_U, kappa_sq, ut);
#ifdef PSMILU_ENABLE_NORM_STAT
    ut_norm_ratios[step] = ut.norm1();
#endif  // PSMILU_ENABLE_NORM_STAT
#ifndef PSMILU_DISABLE_SPACE_DROP
    const size_type n_ut = ut.size();
    apply_space_dropping(row_sizes[p[step]], alpha_U, ut);
    info_counter[3] += n_ut - ut.size();
#endif  // PSMILU_DISABLE_SPACE_DROP
    info_counter[5] += ori_ut_size - ut.size();
    ut.sort_indices();
#ifdef PSMILU_ENABLE_NORM_STAT
    ut_norm_ratios[step] = ut.norm1() / ut_norm_ratios[step];
#endif  // PSMILU_ENABLE_NORM_STAT

    // push back rows to U
    U.push_back_row(step, ut.inds().cbegin(), ut.inds().cbegin() + ut.size(),
                    ut.vals());

    Crout_info("  ut sizes before/after dropping %zd/%zd, drops=%zd",
               ori_ut_size, ut.size(), ori_ut_size - ut.size());

    // apply numerical dropping on L
    apply_num_dropping(tau_L, kappa_sq, l);

#ifdef PSMILU_ENABLE_NORM_STAT
    l_norm_ratios[step] = l.norm1();
#endif  // PSMILU_ENABLE_NORM_STAT

    if (IsSymm) {
#ifndef PSMILU_DISABLE_SPACE_DROP
      // for symmetric cases, we need first find the leading block size
      auto info = find_sorted(ut.inds().cbegin(),
                              ut.inds().cbegin() + ut.size(), m2 + ONE_BASED);
      apply_space_dropping(col_sizes[q[step]], alpha_L, l,
                           info.second - ut.inds().cbegin());

      auto u_last = info.second;
#else   // !PSMILU_DISABLE_SPACE_DROP
      auto u_last = ut.inds().cbegin() + ut.size();
#endif  // PSMILU_DISABLE_SPACE_DROP

      l.sort_indices();
      Crout_info(
          "  l sizes (asymm parts) before/after dropping %zd/%zd, drops=%zd",
          ori_l_size, l.size(), ori_l_size - l.size());

      // push back symmetric entries and offsets
      L.push_back_col(step, ut.inds().cbegin(), u_last, ut.vals(),
                      l.inds().cbegin(), l.inds().cbegin() + l.size(),
                      l.vals());
    } else {
#ifndef PSMILU_DISABLE_SPACE_DROP
      const size_type n_l = l.size();
      apply_space_dropping(col_sizes[q[step]], alpha_L, l);
      info_counter[4] += n_l - l.size();
#endif  // PSMILU_DISABLE_SPACE_DROP
      info_counter[6] += ori_l_size - l.size();
      l.sort_indices();
      Crout_info("  l sizes before/after dropping %zd/%zd, drops=%zd",
                 ori_l_size, l.size(), ori_l_size - l.size());
      L.push_back_col(step, l.inds().cbegin(), l.inds().cbegin() + l.size(),
                      l.vals());
    }
#ifdef PSMILU_ENABLE_NORM_STAT
    l_norm_ratios[step] = l.norm1() / l_norm_ratios[step];
#endif  // PSMILU_ENABLE_NORM_STAT

    // update position
    step.update_compress(U, U_list, U_start);
    step.update_compress(L, L_list, L_start);
    if (IsSymm) {
      if (!step.defers() && m2 == n)
        L_offsets[step] = L.nnz_in_col(step);
      else
        step.update_L_start_offset_symm(L, m2, L_offsets);
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
    L_start[i] += L.col_start()[i] - ONE_BASED;
    U_start[i] += U.row_start()[i] - ONE_BASED;
  }

  // compress tails
  internal::compress_tails(U, L, U_start, L_start, m, step.defers());

  timer.finish();  // profile Crout update

  // now we are done
  if (psmilu_verbose(INFO, opts)) {
    psmilu_info(
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
        "\tmax |kappa_l|=%g"
#ifdef PSMILU_DEFERREDFAC_VERBOSE_STAT
        "\n\tboth-inv-bad deferrals=%zd"
#endif  // PSMILU_DEFERREDFAC_VERBOSE_STAT
        ,
        step.defers(), m2, m, m2 - m, (size_type)info_counter[0],
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
                              }))
#ifdef PSMILU_DEFERREDFAC_VERBOSE_STAT
            ,
        (size_type)info_counter[2]
#endif  // PSMILU_DEFERREDFAC_VERBOSE_STAT
    );
#ifndef NDEBUG
    _GET_MAX_MIN_MINABS(d, m);
    _SHOW_MAX_MIN_MINABS(d);
#endif
#ifdef PSMILU_ENABLE_NORM_STAT
    psmilu_info(
        "analyzing the behaviors of l and ut after/before space dropping...");
    std::sort(ut_norm_ratios.begin(), ut_norm_ratios.begin() + m);
    std::sort(l_norm_ratios.begin(), l_norm_ratios.begin() + m);
    const size_type i25 = 25.0 * m / 100, i50 = 50.0 * m / 100,
                    i75 = 75.0 * m / 100;
    psmilu_info(
        "\tut 1-norm ratio:\n"
        "\t\tmin=%g\n"
        "\t\t25%%=%g\n"
        "\t\tmedian=%g\n"
        "\t\t75%%=%g\n"
        "\t\tmax=%g\n"
        "\t\t<=99%%=%zd\n"
        "\tl 1-norm ratio:\n"
        "\t\tmin=%g\n"
        "\t\t25%%=%g\n"
        "\t\tmedian=%g\n"
        "\t\t75%%=%g\n"
        "\t\tmax=%g\n"
        "\t\t<=99%%=%zd",
        (double)ut_norm_ratios.front(), (double)ut_norm_ratios[i25],
        (double)ut_norm_ratios[i50], (double)ut_norm_ratios[i75],
        (double)ut_norm_ratios[m - 1],
        size_type(std::upper_bound(ut_norm_ratios.cbegin(),
                                   ut_norm_ratios.cend(), 0.99) -
                  ut_norm_ratios.cbegin()),
        (double)l_norm_ratios.front(), (double)l_norm_ratios[i25],
        (double)l_norm_ratios[i50], (double)l_norm_ratios[i75],
        (double)l_norm_ratios[m - 1],
        size_type(std::upper_bound(l_norm_ratios.cbegin(), l_norm_ratios.cend(),
                                   0.99) -
                  l_norm_ratios.cbegin()));
#endif  // PSMILU_ENABLE_NORM_STAT
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

#endif  // _PSMILU_FACDEFERTHIN_HPP