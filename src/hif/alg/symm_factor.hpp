///////////////////////////////////////////////////////////////////////////////
//                  This file is part of the HIFIR library                   //
///////////////////////////////////////////////////////////////////////////////

/*!
 * \file hif/alg/symm_factor.hpp
 * \brief Kernels for deferred incomplete LDL' factorization
 * \author Qiao Chen

\verbatim
Copyright (C) 2020 NumGeom Group at Stony Brook University

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

#ifndef _HIF_ALG_SYMMFACTOR_HPP
#define _HIF_ALG_SYMMFACTOR_HPP

#include "hif/alg/factor.hpp"

namespace hif {

namespace internal {

/// \brief compress offsets to have a compact L
/// \tparam L_Type storage for \a L, see \ref CCS
/// \tparam IndPtrType array for storing starting positions, see \ref Array
/// \param[in,out] L uncompressed \a L part
/// \param[in] L_start starting positions of the offset of \a L
/// \param[in] m leading block size
/// \param[in] dfrs total number of deferrals
template <class L_Type, class IndPtrType>
inline void compress_tail(L_Type &L, const IndPtrType &L_start,
                          const typename IndPtrType::size_type m,
                          const typename IndPtrType::size_type dfrs) {
  using size_type  = typename IndPtrType::size_type;
  using index_type = typename L_Type::index_type;

  if (dfrs) {
    const auto comp_index = [=](index_type &j) { j -= dfrs; };
    auto       L_first    = L.row_ind().begin();
    for (size_type i(0); i < m; ++i)
      std::for_each(L_first + L_start[i], L.row_ind_end(i), comp_index);
  }

  // reshape the secondary axis of the matrices
  L.resize_nrows(L.nrows() / 2);

#ifdef HIF_DEBUG
  L.check_validity();
#endif
}

/// \brief efficiently create transpose
/// \tparam FromType CS type of which we compute transpose
/// \tparam ToType CS type of which we store the resulting transpose
/// \param[in] A input coefficient matrix A that will be transposed
/// \param[out] AT stores the transpose of A
/// \param[in] do_conj (optional) do conjugate for complex, default is false
template <class FromType, class ToType>
inline void make_transpose(const FromType &A, ToType &AT,
                           const bool do_conj = false) {
  AT.resize(A.ncols(), A.nrows());
  // NOTE: Array assignment is shallow copy
  if (!std::is_same<FromType, ToType>::value) {
    // if not the same, then we can create efficient transpose by aliasing
    // the three arrays in A
    AT.inds()      = A.inds();
    AT.ind_start() = A.ind_start();
    if (std::is_floating_point<typename FromType::value_type>::value ||
        !do_conj)
      AT.vals() = A.vals();
    else if (do_conj) {
      const auto n = A.vals().size();
      AT.vals().resize(n);
      for (auto i(0ul); i < n; ++i) AT.vals()[i] = conjugate(A.vals()[i]);
    }
  } else {
    // for homogeneous class type, we need to create an intermidate storage
    // for the dual storage scheme format of A
    typename FromType::other_type A_dual(A);
    AT.inds()      = A_dual.inds();
    AT.ind_start() = A_dual.ind_start();
    AT.vals()      = A_dual.vals();
    if (!std::is_floating_point<typename FromType::value_type>::value &&
        do_conj) {
      const auto n = AT.vals().size();
      for (auto i(0ul); i < n; ++i) AT.vals()[i] = conjugate(AT.vals()[i]);
    }
  }
}
}  // namespace internal

/// \brief perform partial incomplete LU for a level
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
/// \param[in,out] stats hierarchical stats
/// \param[in] schur_threads threads usedin Schur-related computations
/// \return Schur complement for next level (if needed), in the same type as
///         that of the input, i.e. \a CsType
/// \ingroup fac
template <class CsType, class CroutStreamer, class PrecsType, class IntArray>
inline CsType symm_level_factorize(
    const CsType &A, const typename CsType::size_type m0,
    const typename CsType::size_type N, const Options &opts,
    const CroutStreamer &Crout_info, PrecsType &precs, IntArray &row_sizes,
    typename CsType::size_type *stats, const int schur_threads = 1) {
  typedef CsType                      input_type;
  typedef typename CsType::other_type other_type;
  using cs_trait = internal::CompressedTypeTrait<input_type, other_type>;
  typedef typename cs_trait::crs_type                     crs_type;
  typedef typename cs_trait::ccs_type                     ccs_type;
  typedef typename CsType::index_type                     index_type;
  typedef typename CsType::indptr_type                    indptr_type;
  typedef typename CsType::size_type                      size_type;
  typedef typename CsType::value_type                     value_type;
  typedef DenseMatrix<value_type>                         dense_type;
  typedef typename ValueTypeTrait<value_type>::value_type scalar_type;

  hif_error_if(A.nrows() != A.ncols(), "only squared systems are supported");

  hif_assert(m0 <= A.nrows(), "leading size should be smaller than size of A");
  const size_type cur_level = precs.size() + 1;

  if (hif_verbose(INFO, opts))
    hif_info("\nenter symmetric level %zd.\n", cur_level);

  DefaultTimer timer;

  // build counterpart type
  other_type A_counterpart;
  // NOTE: This can be done efficiently for symmetric real systems
  // NOTE: For complex Hermitian matrix, the following will not work
  internal::make_transpose(A, A_counterpart);

  // now use our trait and its static methods to precisely determine the ccs
  // and crs components.
  const crs_type &A_crs = cs_trait::select_crs(A, A_counterpart);
  const ccs_type &A_ccs = cs_trait::select_ccs(A, A_counterpart);

  // handle row and column sizes
  if (cur_level == 1u) {
    row_sizes.resize(A.nrows());
    constexpr static double min_local_size_ratio =
        HIF_MIN_LOCAL_SIZE_PERCTG / 100.0;
    for (size_type i(0); i < A.nrows(); ++i) row_sizes[i] = A_crs.nnz_in_row(i);
    // filter out too small terms
    const size_type lower_row =
        std::ceil(min_local_size_ratio * A.nnz() / A.nrows());
    std::replace_if(
        row_sizes.begin(), row_sizes.begin() + A.nrows(),
        [=](const index_type i) { return (size_type)i < lower_row; },
        lower_row);
  }

  // preprocessing
  timer.start();
  Array<scalar_type>       s, t;
  BiPermMatrix<index_type> p, q;
  size_type                m;
  if (!opts.no_pre) {
    // only use symmetric preprocessing
    if (hif_verbose(INFO, opts))
      hif_info("performing symm preprocessing with leading block size %zd...",
               m0);
    m = do_preprocessing<true>(A_ccs, A_crs, m0, cur_level, opts, s, t, p, q);
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

#ifdef HIF_SAVE_FIRST_LEVEL_PERM_A
  if (cur_level == 1u) {
    std::mt19937                       eng(std::time(0));
    std::uniform_int_distribution<int> d(1000, 1000000);
    const std::string fname  = "Perm_A_" + std::to_string(d(eng)) + ".hif";
    auto              A_perm = A_crs.compute_perm(p(), q.inv(), m);
    Array<value_type> s2(m), t2(m);
    for (size_type i = 0; i < m; ++i) {
      s2[i] = s[p[i]];
      t2[i] = t[q[i]];
    }
    A_perm.scale_diag_left(s2);
    A_perm.scale_diag_right(t2);
    hif_info("\nsaving first level permutated matrix to file %s\n",
             fname.c_str());
    A_perm.write_bin(fname.c_str(), IsSymm ? m : size_type(0));
  }
#endif

  if (hif_verbose(INFO, opts)) hif_info("preparing data variables...");

  timer.start();

  // extract diagonal
  auto d = internal::extract_perm_diag(s, A_ccs, t, m, p, q);

  // create L storage with deferred
  ccs_type L(A.nrows() * 2, m);
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

  // create l buffer
  SparseVector<value_type, index_type> l(A.nrows() * 2);

  // create buffer for L start
  Array<indptr_type> L_start(m);
  hif_error_if(L_start.status() == DATA_UNDEF,
               "memory allocation failed for L_start at level %zd.", cur_level);

  // create buffer for L list
  Array<index_type> L_list(A.nrows() * 2);
  hif_error_if(L_list.status() == DATA_UNDEF,
               "memory allocation failed for L_list at level %zd.", cur_level);

  // set default value
  std::fill(L_list.begin(), L_list.end(), static_cast<index_type>(-1));

  // create storage for kappa's
  Array<value_type> kappa_l(m);
  hif_error_if(kappa_l.status() == DATA_UNDEF,
               "memory allocation failed for kappa_l at level %zd.", cur_level);

  L.begin_assemble_cols();

  // localize parameters
  double kappa_d, kappa, tau_L, tau_U, alpha_L, alpha_U;
  std::tie(kappa_d, kappa, tau_L, tau_U, alpha_L, alpha_U) =
      internal::determine_fac_pars(opts, cur_level);
  // get spd flag, notice that 0 for indefinite, >0 for pd, and <0 for ng
  const int spd_flag = opts.spd;

  // Removing bounding the large diagonal values
  const auto is_bad_diag = [=](const value_type a) -> bool {
    return std::abs(1. / real(a)) > kappa_d ||
           (spd_flag > 0 && real(a) <= 0.0) || (spd_flag < 0 && real(a) >= 0.0);
  };

  const size_type m2(m), n(A.nrows());

  // deferred permutations
  Array<indptr_type> P(n * 2);
  hif_error_if(P.status() == DATA_UNDEF,
               "memory allocation failed for P at level %zd", cur_level);
  std::copy_n(p().cbegin(), n, P.begin());
  auto &P_inv = p.inv();

  // 0 for defer due to diagonal, 1 for defer due to bad inverse conditioning
  index_type info_counter[] = {0, 0, 0, 0, 0, 0, 0};

  if (hif_verbose(INFO, opts)) hif_info("start Crout update...");
  Crout step;
  for (; step < m; ++step) {
    // first check diagonal
    bool            pvt         = is_bad_diag(d[step.deferred_step()]);
    const size_type m_prev      = m;
    const size_type defers_prev = step.defers();
    info_counter[0] += pvt;

    Crout_info(" Crout step %zd, leading block size %zd", step, m);

    // compute kappa for l wrt deferred index
    step.update_kappa(L, L_list, L_start, kappa_l);

    // check condition number if diagonal doesn't satisfy
    if (!pvt) {
      pvt = std::abs(kappa_l[step]) > kappa;
      info_counter[1] += pvt;
    }

    // handle defer
    if (pvt) {
      while (m > step) {
        --m;
        const auto tail_pos = n + step.defers();
        step.defer_entry(tail_pos, L_start, L, L_list);
        P[tail_pos]        = p[step.deferred_step()];
        P_inv[P[tail_pos]] = tail_pos;
        // mark as empty entries
        P[step.deferred_step()] = -1;

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
        // then compute kappa for l
        step.update_kappa(L, L_list, L_start, kappa_l);
        pvt = std::abs(kappa_l[step]) > kappa;
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

    const auto k_l = kappa_l[step];

    // check pivoting
    hif_assert(!(std::abs(k_l) > kappa), "should not happen!");

    Crout_info("  kappa=%g", (double)std::abs(k_l));

    Crout_info(
        "  previous/current leading block sizes %zd/%zd, local/total "
        "defers=%zd/%zd",
        m_prev, m, step.defers() - defers_prev, step.defers());

    //------------------------
    // update start positions
    //------------------------

    Crout_info("  updating start and performing Crout update");

    // compress diagonal
    step.compress_array(d);
    // compress permutation vectors
    step.compress_array(p);
    // compute l
    step.compute_symm(s, A_ccs, t, P_inv, p[step], m2, L, L_start, L_list, d,
                      l);

    // update diagonal entries
#ifdef HIF_DEBUG
    const bool is_nonsingular =
#else
    (void)
#endif
        step.scale_inv_diag(d, l);
    hif_assert(!is_nonsingular, "l is singular at level %zd step %zd",
               cur_level, step);

    // update diagonals b4 dropping
    step.update_diag<true>(l, l, m2, d);

    //---------------
    // drop and sort
    //---------------

    const size_type ori_l_size = l.size();

    // apply numerical dropping on L
    apply_num_dropping(tau_L, std::abs(k_l) * kappa_d, l);
#ifndef HIF_DISABLE_SPACE_DROP
    const size_type n_l = l.size();
    apply_space_dropping(row_sizes[p[step]], alpha_L, l);
    info_counter[4] += n_l - l.size();
#endif  // HIF_DISABLE_SPACE_DROP
    info_counter[6] += ori_l_size - l.size();
    l.sort_indices();
    Crout_info("  sizes before/after dropping %zd/%zd, drops=%zd", ori_l_size,
               l.size(), ori_l_size - l.size());
    L.push_back_col(step, l.inds().cbegin(), l.inds().cbegin() + l.size(),
                    l.vals());

    // update position
    step.update_compress(L, L_list, L_start);
    Crout_info(" Crout step %zd done!", step);
  }  // for

  // compress permutation vectors
  for (; step < n; ++step) step.assign_gap_array(P, p);

  // rebuild the inverse mappings
  p.build_inv();

  L.end_assemble_cols();

  // Important! Revert the starting positions to global index that is required
  // in older routines
  for (size_type i(0); i < m; ++i) L_start[i] += L.col_start()[i];

  // compress tails
  internal::compress_tail(L, L_start, m, step.defers());

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
        "\tdrop l=%zd\n"
        "\tspace drop l=%zd\n"
        "\tmin |kappa|=%g\n"
        "\tmax |kappa|=%g\n"
        "\tmax |d|=%g",
        step.defers(), m0, m, m0 - m, (size_type)info_counter[0],
        (size_type)info_counter[1], (size_type)info_counter[6],
        (size_type)info_counter[4],
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
    // logging for factorization finished here
    hif_info("computing Schur complement and assembling Prec...");
    internal::print_post_flag(post_flag);
  }

  timer.start();

  crs_type S;

  const auto L_nnz = L.nnz();

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
      internal::make_transpose(L_B, U_B, true);
      timer2.finish();
      if (hif_verbose(INFO2, opts))
        hif_info("splitting LB, copying to UB, and freeing L took %gs.",
                 timer2.time());
      do {
        timer2.start();
        const size_type nnz1 = L_E.nnz();
#ifndef HIF_NO_DROP_LE_UF
        double a_L = opts.alpha_L;
        if (cur_level == 1u && opts.fat_schur_1st) a_L *= 2;
        if (hif_verbose(INFO2, opts))
          hif_info("applying dropping on L_E with alpha=%g...", a_L);
        if (m < n) {
          // use P as buffer
          P[0] = 0;
          for (size_type i(m); i < n; ++i)
            P[i - m + 1] = P[i - m] + row_sizes[p[i]];
          drop_L_E(P, a_L, L_E, l.vals(), l.inds());
        }
#endif  // HIF_NO_DROP_LE_UF
        timer2.finish();
        if (hif_verbose(INFO2, opts))
          hif_info("nnz(L_E)=%zd/%zd, time: %gs...", nnz1, L_E.nnz(),
                   timer2.time());
      } while (false);  // U_F2 got freed

      // NOTE: for symmetric real systems, we need to construct U_F differently
      crs_type U_F;
      internal::make_transpose(L_E, U_F, true);

      timer2.start();
// compute S version of Schur complement
#ifndef _OPENMP
      (void)schur_threads;
      S = compute_Schur_simple(s, A_crs, t, p, p, m, L_E, d, U_F, l);
#else
      if (hif_verbose(INFO, opts))
        hif_info("using %d threads for Schur computation...", schur_threads);
      S = mt::compute_Schur_simple(s, A_crs, t, p, p, m, L_E, d, U_F, l,
                                   schur_threads);
#endif
      timer2.finish();
      if (hif_verbose(INFO, opts))
        hif_info("pure Schur computation time: %gs...", timer2.time());
    } while (false);
  } else {
    S = A_crs;
    p.make_eye();
    std::fill(s.begin(), s.end(), 1);
    std::fill(t.begin(), t.end(), 1);
  }

  // compute the nnz(A)-nnz(B) from the first level
  size_type AmB_nnz(0);
  for (size_type i(m); i < n; ++i) AmB_nnz += row_sizes[p[i]] * 2;

  // L and U got freed, only L_B and U_B exist

  const size_type dense_thres1 = static_cast<size_type>(opts.alpha_L * AmB_nnz),
                  dense_thres2 = std::max(
                      static_cast<size_type>(
                          std::ceil(opts.c_d * std::cbrt(N))),
                      size_type(opts.dense_thres <= 0 ? 2000
                                                      : opts.dense_thres));

  if (hif_verbose(INFO, opts))
    hif_info(
        "nnz(S_C)=%zd, nnz(L/L_B)=%zd/%zd\n"
        "dense_thres{1,2}=%zd/%zd...",
        S.nnz(), L_nnz, L_B.nnz(), dense_thres1, dense_thres2);

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
    // NOTE use P buffer
    for (size_type i(m); i < n; ++i) P[i] = row_sizes[p[i]];
    for (size_type i(m); i < n; ++i) row_sizes[i - m] = P[i];
  }

  ccs_type E, F;

  if (input_type::ROW_MAJOR) {
    A_counterpart.destroy();
    E = internal::extract_E(s, A_crs, t, m, p, p);
  } else {
    E = internal::extract_E(s, A_crs, t, m, p, p);
    A_counterpart.destroy();
  }

  // construct F differently for symmetric real systems
  crs_type F_ccs(E);
  F.resize(E.ncols(), E.nrows());
  F.inds()      = F_ccs.inds();
  F.ind_start() = F_ccs.ind_start();
  F.vals()      = F_ccs.vals();

  // recursive function free
  if (cur_level > 1u) const_cast<input_type &>(A).destroy();

  // copy p to q
  for (size_type i(0); i < p.size(); ++i) {
    q[i]       = p[i];
    q.inv()[i] = p.inv()[i];
  }

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
    auto &last_level = precs.back().symm_dense_solver;
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

#endif  // _HIF_ALG_SYMMFACTOR_HPP
