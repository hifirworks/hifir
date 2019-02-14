//@HEADER
//----------------------------------------------------------------------------
//                Copyright (C) 2019 The PSMILU AUTHORS
//----------------------------------------------------------------------------
//@HEADER

#include "common.hpp"
// line break to avoid sorting
#include "psmilu_Array.hpp"
#include "psmilu_AugmentedStorage.hpp"
#include "psmilu_CompressedStorage.hpp"
#include "psmilu_Crout.hpp"
#include "psmilu_PermMatrix.hpp"
#include "psmilu_Schur.hpp"
#include "psmilu_SparseVec.hpp"
#include "psmilu_diag_pivot.hpp"
#include "psmilu_inv_thres.hpp"

#include <gtest/gtest.h>

using namespace psmilu;

constexpr static int N       = 100;
constexpr static int START_M = 80;

static const RandRealGen r_rand(0.0, 2.0);
static const RandIntGen  i_rand(0, N);

TEST(CRS_api, test_core_c) {
  typedef CRS<double, int> crs_t;
  crs_t                    A(N, N);
  Array<double>            d(N);

  // construct matrix
  {
    std::vector<std::vector<int>> col_inds(N);
    std::vector<bool>             tags(N, false);
    for (int i = 0u; i < N; ++i) {
      const int nnz    = std::max(5, i_rand() % N);
      int       counts = 0, guard = 0;
      auto&     col_ind = col_inds[i];
      for (;;) {
        if (counts == nnz || guard > 2 * N) break;
        const int idx = i_rand() % N;
        ++guard;
        if (tags[idx]) continue;
        ++counts;
        col_ind.push_back(idx);
        tags[idx] = true;
      }
      if (std::find(col_ind.cbegin(), col_ind.cend(), i) == col_ind.cend())
        col_ind.push_back(i);
      std::sort(col_ind.begin(), col_ind.end());
      for (const auto j : col_ind) tags[j] = false;
    }
    std::vector<double> buf(N);
    // construct sparse matrix
    A.begin_assemble_rows();
    for (int i = 0; i < N; ++i) {
      const auto& col_ind = col_inds[i];
      for (const auto col : col_ind) buf[col] = r_rand();
      A.push_back_row(i, col_ind.cbegin(), col_ind.cend(), buf);
      d[i] = buf[i];
      ASSERT_EQ(A.nnz_in_row(i), col_ind.size());
    }
    A.end_assemble_rows();
  }

  const auto nnz = A.nnz();

  using ccs_t = crs_t::other_type;
  ccs_t Ap(A);

  ASSERT_EQ(nnz, Ap.nnz());

  int m = START_M;  // prepare start m

  // augmented ds for crout
  AugCRS<crs_t> U(m, N);
  U.reserve(nnz * 4);
  AugCCS<ccs_t> L(N, m);
  L.reserve(nnz * 4);
  Array<int> L_start(N), U_start(N);

  // buffer
  SparseVector<double, int> l(N), ut(N);

  // permutation
  BiPermMatrix<int> p(N), q(N);
  p.make_eye();
  q.make_eye();

  // dummy scaling
  Array<double> s(N, 1.0), t(N, 1.0);

  // cond
  Array<double> kappa_l(N), kappa_ut(N);

  U.begin_assemble_rows();
  L.begin_assemble_cols();
  for (Crout crout; (int)crout < m; ++crout) {
    bool pvt = std::abs(1. / d[crout]) > 10.0;
    for (;;) {
      if (pvt) {
        while (std::abs(1. / d[m - 1]) > 10.0 && m > (int)crout) --m;
        if (m == (int)crout) break;
        U.interchange_cols(crout, m - 1);
        L.interchange_rows(crout, m - 1);
        p.interchange(crout, m - 1);
        q.interchange(crout, m - 1);
        std::swap(d[crout], d[m - 1]);
        --m;
      }
      update_kappa_ut(crout, U, kappa_ut);
      update_kappa_l<false>(crout, L, kappa_ut, kappa_l);
      const double k_ut = std::abs(kappa_ut[crout]),
                   k_l  = std::abs(kappa_l[crout]);
      pvt               = k_ut > 100.0 || k_l > 100.0;
      if (pvt) continue;
      ut.reset_counter();
      crout.compute_ut(s, A, t, p[crout], q, L, d, U, U_start, ut);
      l.reset_counter();
      crout.compute_l<false>(s, Ap, t, p, q[crout], m, L, L_start, d, U, l);
      // scale inv d
      EXPECT_FALSE(crout.scale_inv_diag(d, ut))
          << "singular at step " << crout << " for ut\n";
      EXPECT_FALSE(crout.scale_inv_diag(d, l))
          << "singular at step " << crout << " for l\n";
      apply_dropping_and_sort(0.01, k_ut, A.nnz_in_row(p[crout]), 4, ut);
      apply_dropping_and_sort(0.01, k_l, Ap.nnz_in_col(q[crout]), 4, l);
      U.push_back_row(crout, ut.inds().cbegin(), ut.inds().cbegin() + ut.size(),
                      ut.vals());
      L.push_back_col(crout, l.inds().cbegin(), l.inds().cbegin() + l.size(),
                      l.vals());
      crout.update_U_start(U, U_start);
      crout.update_L_start<false>(L, m, L_start);
      break;
    }
  }
  U.end_assemble_rows();
  L.end_assemble_cols();

  std::cout << "M=" << START_M << ", m=" << m << std::endl;
  std::cout << "U_start=\n";
  auto itr = U.col_ind().cbegin();
  for (int i = 0; i < m; ++i) {
    std::cout << "row " << i << ':';
    for (auto j = itr + U_start[i]; j != U.col_ind_cend(i); ++j) {
      std::cout << *j << ' ';
      EXPECT_GE(*j, m);
    }
    std::cout << std::endl;
  }
  std::cout << "L_start=\n";
  for (int i = 0; i < m; ++i) {
    std::cout << "col " << i << ':';
    for (auto j = L.row_ind().cbegin() + L_start[i]; j != L.row_ind_cend(i);
         ++j) {
      EXPECT_GE(*j, m);
      std::cout << *j << ' ';
    }
    std::cout << std::endl;
  }

  crs_t Sc;
  compute_Schur_C(s, A, t, p, q, m, N, L, d, U, U_start, Sc);
}
