//@HEADER
//----------------------------------------------------------------------------
//                Copyright (C) 2019 The PSMILU AUTHORS
//----------------------------------------------------------------------------
//@HEADER

/// \file psmilu_MT/fac.hpp
/// \brief Implementation of factorization in MT setting
/// \authors Qiao,

#ifndef _PSMILU_MT_FAC_HPP
#define _PSMILU_MT_FAC_HPP

#include <omp.h>
#include <memory>

#include "Crout.hpp"
#include "psmilu_fac.hpp"

namespace psmilu {
namespace mt {
/// \ingroup mt
template <bool IsSymm, class CsType, class CroutStreamer, class PrecsType>
inline CsType iludp_factor(const CsType &A, const typename CsType::size_type m0,
                           const typename CsType::size_type N,
                           const int threads, const Options &opts,
                           const CroutStreamer &Crout_info, PrecsType &precs) {
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

  psmilu_error_if(threads < 2, "must be called with multiple threads!");

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

  if (psmilu_verbose(INFO, opts)) psmilu_info("performing preprocessing...");

  // preprocessing
  timer.start();
  Array<value_type>        s, t;
  BiPermMatrix<index_type> p, q;
  size_type m = do_preprocessing<IsSymm>(A_ccs, m0, opts, s, t, p, q);
  timer.finish();  // prefile pre-processing

  if (psmilu_verbose(INFO, opts)) psmilu_info("time: %gs", timer.time());

  const int l_start_id = Crout_MT::partition<IsSymm>(A.nrows(), m, threads);
  const int u_threads = l_start_id, l_threads = threads - u_threads;
  const int u_sec_id = l_start_id - 1, l_sec_id = threads - 1;

  const auto in_u_group = [=](const int thread) -> bool {
    return thread < l_start_id;
  };
  const auto is_u_master = [](const int thread) -> bool { return thread == 0; };
  const auto is_l_master = [=](const int thread) -> bool {
    return thread == l_start_id;
  };

  if (psmilu_verbose(INFO, opts)) {
    psmilu_info("partition the problem into %d threads...", threads);
    psmilu_info("starting thread ID for U part is 0...");
    psmilu_info("starting thread ID for L part is %d...", l_start_id);
  }

  if (psmilu_verbose(INFO, opts)) psmilu_info("preparing data variables...");

  timer.start();

  // extract diagonal
  auto d = internal::extract_perm_diag(s, A_ccs, t, m, p, q);

  // create U storage
  aug_crs_type U(m, A.ncols());
  psmilu_error_if(U.row_start().status() == DATA_UNDEF,
                  "memory allocation failed for U:row_start at level %zd.",
                  cur_level);
  U.reserve(A.nnz() * opts.alpha_U);
  psmilu_error_if(
      U.col_ind().status() == DATA_UNDEF || U.vals().status() == DATA_UNDEF,
      "memory allocation failed for U-nnz arrays at level %zd.", cur_level);

  // create L storage
  aug_ccs_type L(A.nrows(), m);
  psmilu_error_if(L.col_start().status() == DATA_UNDEF,
                  "memory allocation failed for L:col_start at level %zd.",
                  cur_level);
  L.reserve(A.nnz() * opts.alpha_L);
  psmilu_error_if(
      L.row_ind().status() == DATA_UNDEF || L.vals().status() == DATA_UNDEF,
      "memory allocation failed for L-nnz arrays at level %zd.", cur_level);

  // create l and ut buffer
  SparseVector<value_type, index_type, ONE_BASED> l(A.nrows()), ut(A.ncols());

  // create buffer for L and U start
  const std::unique_ptr<Array<index_type>[]> starts(
      new (std::nothrow) Array<index_type>[threads]);
  psmilu_error_if(!starts.get(), "memory allocation failed");
  for (int i = 0; i < threads; ++i) {
    auto &start = starts[i];
    start.resize(m);
    psmilu_error_if(
        start.status() == DATA_UNDEF,
        "memory allocation failed for L_start and/or U_start at level %zd.",
        cur_level);
  }
  auto &L_start = starts[l_start_id], &U_start = starts[0];

  // create storage for kappa's
  Array<value_type> kappa_l(m), kappa_ut(m);
  psmilu_error_if(
      kappa_l.status() == DATA_UNDEF || kappa_ut.status() == DATA_UNDEF,
      "memory allocation failed for kappa_l and/or kappa_ut at level %zd.",
      cur_level);

  U.begin_assemble_rows();
  L.begin_assemble_cols();

  // localize parameters
  const auto tau_d = opts.tau_d, tau_kappa = opts.tau_kappa, tau_U = opts.tau_U,
             tau_L   = opts.tau_L;
  const auto alpha_L = opts.alpha_L, alpha_U = opts.alpha_U;

  size_type       interchanges_;
  const size_type m_in(m), n(A.nrows());

  if (psmilu_verbose(INFO, opts)) psmilu_info("start Crout update...");

#pragma omp parallel num_threads(threads)
  {
    size_type         interchanges(0);
    const int         thread  = omp_get_thread_num();
    auto &            pos     = starts[thread];
    const index_type *pos_end = nullptr;
    if (thread != l_start_id - 1 && thread != threads - 1)
      pos_end = starts[thread + 1].data();
    else if (thread == l_start_id - 1)
      pos_end = U.row_start().data() + 1;
    else
      pos_end = L.col_start().data() + 1;

    const bool u_master = is_u_master(thread), l_master = is_l_master(thread),
               u_group = in_u_group(thread);

    for (Crout_MT step; step < m; ++step) {
      // first check diagonal
      bool            pvt    = std::abs(1. / d[step]) > tau_d;
      const size_type m_prev = m;

#pragma omp master
      do {
        Crout_info(" Crout step %zd, leading block size %zd", step, m_prev);
      } while (false);  // master

      size_type local_interchanges = 0;

      // inf loop
      for (;;) {
        //----------------
        // pivoting
        //---------------

        if (pvt) {
#pragma omp master
          // test m value before plugin m-1 to array accessing
          while (m > step && std::abs(1. / d[m - 1]) > tau_d) --m;
#pragma omp barrier
          if (m == step) break;
          if (u_master)
            U.interchange_cols(step, m - 1);
          else if (l_master)
            L.interchange_rows(step, m - 1);

          if (thread == l_sec_id) {
            // udpate p and q; be aware that the inverse mappings are also
            // updated
            p.interchange(step, m - 1);
            q.interchange(step, m - 1);
            // update diagonal since we maintain a permutated version of it
            std::swap(d[step], d[m - 1]);
            --m;
            if (IsSymm) internal::update_L_start_symm(L, m, L_start);
          }
          ++local_interchanges;
#pragma omp barrier
        }

        //----------------
        // inverse thres
        //----------------

        if (!IsSymm) {
          if (u_master)
            update_kappa_ut(step, U, kappa_ut);
          else if (l_master)
            update_kappa_l<false>(step, L, kappa_ut, kappa_l);

        } else {
          if (u_master) {
            update_kappa_ut(step, U, kappa_ut);
            kappa_l[step] = kappa_ut[step];
          }
        }

        // check pivoting
        const auto k_ut = std::abs(kappa_ut[step]),
                   k_l  = std::abs(kappa_l[step]);
        pvt             = k_ut > tau_kappa || k_l > tau_kappa;

        // #pragma omp master
        //         do {
        //           Crout_info("  kappa_ut=%g, kappa_l=%g, pvt=%s",
        //           (double)k_ut,
        //                      (double)k_l, (pvt ? "yes" : "no"));
        //         } while (false);  // master
        // #pragma omp barrier
        if (pvt) continue;

#pragma omp master
        do {
          Crout_info(
              "  previous/current leading block sizes %zd/%zd, "
              "interchanges=%zd",
              m_prev, m, local_interchanges);
          Crout_info("  updating L_start/U_start and performing Crout update");
        } while (false);                     // master
        interchanges += local_interchanges;  // accumulate global interchanges

        //------------------------
        // update start positions
        //------------------------

        // runtime partition
        const size_type cur_size = n - step, u_stride = cur_size / u_threads,
                        l_stride =
                            IsSymm ? (n - m) / l_threads : cur_size / l_threads;

        if (u_group) {
          if (u_master)
            step.update_U_start(U, U_start);
          else
            step.update_U_pos(U, L, u_stride, 0, thread, pos);
        } else {
          if (l_master)
            step.update_L_start<IsSymm>(L, m, L_start);
          else
            step.update_L_pos<IsSymm>(L, U, m, l_stride, l_start_id, thread,
                                      pos);
        }

        step.template load_ut_l<IsSymm>(s, A_crs, A_ccs, t, p, q, m, u_sec_id,
                                        l_sec_id, thread, ut, l);
        // #pragma omp barrier
        if (u_group)
          step.compute_ut(L, d, U, pos, pos_end, ut);
        else
          step.compute_l(L, pos, pos_end, d, U, l);

#pragma omp barrier
        if (u_master)
          step.scale_inv_diag(d, ut);
        else if (l_master)
          step.scale_inv_diag(d, l);

#pragma omp barrier

          // compute and update the diagonal
#pragma omp master
        do {
          step.update_B_diag<IsSymm>(l, ut, m, d);
        } while (false);  // master
#pragma omp barrier

        // droping and assembling
        const size_type ori_ut_size = ut.size(), ori_l_size = l.size();
        if (!IsSymm) {
          if (u_master) {
            // apply drop for U
            apply_dropping_and_sort(tau_U, k_ut, A_crs.nnz_in_row(p[step]),
                                    alpha_U, ut);
            // push back rows to U
            U.push_back_row(step, ut.inds().cbegin(),
                            ut.inds().cbegin() + ut.size(), ut.vals());
          } else if (l_master) {
            // for asymmetric cases, just do exactly the same things as ut
            apply_dropping_and_sort(tau_L, k_l, A_ccs.nnz_in_col(q[step]),
                                    alpha_L, l);
            L.push_back_col(step, l.inds().cbegin(),
                            l.inds().cbegin() + l.size(), l.vals());
          }
#pragma omp master
          do {
            Crout_info("  ut sizes before/after dropping %zd/%zd, drops=%zd",
                       ori_ut_size, ut.size(), ori_ut_size - ut.size());
            Crout_info("  l sizes before/after dropping %zd/%zd, drops=%zd",
                       ori_l_size, l.size(), ori_l_size - l.size());
          } while (false);  // master
        } else {            // symmetric case
#pragma omp master
          do {
            apply_dropping_and_sort(tau_U, k_ut, A_crs.nnz_in_row(p[step]),
                                    alpha_U, ut);
            // push back rows to U
            U.push_back_row(step, ut.inds().cbegin(),
                            ut.inds().cbegin() + ut.size(), ut.vals());
            Crout_info("  ut sizes before/after dropping %zd/%zd, drops=%zd",
                       ori_ut_size, ut.size(), ori_ut_size - ut.size());
            // for symmetric cases, we need first find the leading block size
            auto info =
                find_sorted(ut.inds().cbegin(), ut.inds().cbegin() + ut.size(),
                            m + ONE_BASED);
            apply_dropping_and_sort(tau_L, k_l, A_ccs.nnz_in_col(q[step]),
                                    alpha_L, l,
                                    info.second - ut.inds().cbegin());

#ifndef NDEBUG
            if (info.second != ut.inds().cbegin() &&
                info.second != ut.inds().cbegin() + ut.size() && l.size())
              psmilu_error_if(*(info.second - 1) >= *l.inds().cbegin() ||
                                  *(info.second - 1) - ONE_BASED >= m,
                              "l contains symm part (%zd,%zd,%zd)",
                              (size_type)(*(info.second - 1)),
                              (size_type)*l.inds().cbegin(), m);
#endif

            Crout_info(
                "  l sizes (asymm parts) before/after dropping %zd/%zd, "
                "drops=%zd",
                ori_l_size, l.size(), ori_l_size - l.size());

            // push back symmetric entries and offsets
            L.push_back_col(step, ut.inds().cbegin(), info.second, ut.vals(),
                            l.inds().cbegin(), l.inds().cbegin() + l.size(),
                            l.vals());
          } while (false);  // master
        }

        break;
      }  // inf loop
#pragma omp barrier
    }
#pragma omp master
    interchanges_ = interchanges;
  }

  U.end_assemble_rows();
  L.end_assemble_cols();

  // finalize start positions
  U_start[m - 1] = U.row_start()[m - 1];
  L_start[m - 1] = L.col_start()[m - 1];

  timer.finish();  // profile Crout update

  // now we are done
  if (psmilu_verbose(INFO, opts)) {
    psmilu_info(
        "finish Crout update...\n"
        "\ttotal interchanges=%zd\n"
        "\tleading block size in=%zd\n"
        "\tleading block size out=%zd\n"
        "\tdiff=%zd",
        interchanges_, m_in, m, m_in - m);
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
  auto L_B = internal::extract_L_B(L, m, L_start);
  auto U_B = internal::extract_U_B(U, m, U_start);

  if (psmilu_verbose(INFO, opts))
    psmilu_info("nnz(S_C)=%zd, nnz(L_B)=%zd, nnz(U_B)=%zd...", S.nnz(),
                L_B.nnz(), U_B.nnz());

  // test H version
  const size_type nm     = A.nrows() - m;
  const auto      cbrt_N = std::cbrt(N);
  dense_type      S_D;
  psmilu_assert(S_D.empty(), "fatal!");
  if (S.nnz() >= static_cast<size_type>(opts.rho * nm * nm) ||
      nm <= static_cast<size_type>(opts.c_d * cbrt_N)) {
    bool use_h_ver = false;
    S_D            = dense_type::from_sparse(S);
    if (m <= static_cast<size_type>(opts.c_h * cbrt_N)) {
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
}  // namespace mt
}  // namespace psmilu

#endif  // _PSMILU_MT_FAC_HPP
