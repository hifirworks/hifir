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
#include "psmilu_PermMatrix.hpp"
#include "psmilu_SparseVec.hpp"
#include "psmilu_diag_pivot.hpp"

#include <gtest/gtest.h>
#include <numeric>

using namespace psmilu;

static const RandIntGen  i_rand(0, 100);
static const RandRealGen r_rand;

constexpr static double tol = 1e-13;

TEST(DIAG_COND, c) {
  const int n = i_rand() + 2;
  std::cout << "Problem size is " << n << '\n';
  using crs_t     = CRS<double, int>;
  using ccs_t     = CCS<double, int>;
  using aug_crs_t = AugCRS<crs_t>;
  using aug_ccs_t = AugCCS<ccs_t>;

  // construct crs (U)
  aug_crs_t                     crs(n, n);
  std::vector<std::vector<int>> col_inds(n);
  std::vector<bool>             tags(n, false);
  for (int i = 0; i < n; ++i) {
    const int nnz    = i_rand() % (n - i);
    int       counts = 0, guard = 0;
    for (;;) {
      if (counts == nnz || guard > 2 * nnz) break;
      int idx = i_rand() % (n - i) + i;
      if (idx == i) ++idx;
      ++guard;
      if (tags[idx]) continue;
      ++counts;
      tags[idx] = true;
      col_inds[i].push_back(idx);
    }
    std::cout << i << '\n';
    std::sort(col_inds[i].begin(), col_inds[i].end());
    for (const auto j : col_inds[i]) tags[j] = false;
    for (const auto j : col_inds[i]) ASSERT_GT(j, i);
  }
  std::cout << 1 << std::endl;
  std::vector<double> val_buf(n, 0.0);
  crs.begin_assemble_rows();
  for (int i = 0; i < n; ++i) {
    const auto &col_ind = col_inds[i];
    for (const auto col : col_ind) val_buf[col] = r_rand();
    crs.push_back_row(i, col_ind.cbegin(), col_ind.cend(), val_buf);
    ASSERT_EQ(crs.nnz_in_row(i), col_ind.size());
  }
  crs.end_assemble_rows();

  // construct ccs
  aug_ccs_t ccs(n, n);
  std::fill(tags.begin(), tags.end(), false);
  std::vector<std::vector<int>> &row_inds = col_inds;
  for (int i = 0; i < n; ++i) {
    row_inds[i].clear();
    const int nnz    = i_rand() % (n - i);
    int       counts = 0, guard = 0;
    for (;;) {
      if (counts == nnz || guard > 2 * nnz) break;
      int idx = i_rand() % (n - i) + i;
      if (idx == i) ++idx;
      ++guard;
      if (tags[idx]) continue;
      ++counts;
      tags[idx] = true;
      row_inds[i].push_back(idx);
    }
    std::sort(row_inds[i].begin(), row_inds[i].end());
    for (const auto j : row_inds[i]) tags[j] = false;
    for (const auto j : row_inds[i]) ASSERT_GT(j, i);
  }
  ccs.begin_assemble_cols();
  for (int i = 0; i < n; ++i) {
    const auto &row_ind = row_inds[i];
    for (const auto row : row_ind) val_buf[row] = r_rand();
    ccs.push_back_col(i, row_ind.cbegin(), row_ind.cend(), val_buf);
    ASSERT_EQ(ccs.nnz_in_col(i), row_ind.size());
  }
  ccs.end_assemble_cols();

  Array<double> c_l(n), c_u(n);
  Array<double> kappa_l(n), kappa_ut(n);

  const crs_t::size_type m = n;
  const auto &           L = ccs;
  const auto &           U = crs;
  for (crs_t::size_type step = 0u; step < m; ++step) {
    std::cout << "enter step " << step << '\n';
    update_kappa_ut(step, U, kappa_ut) ? c_u[step] = 1.0 : c_u[step] = -1.0;
    update_kappa_l<false>(step, L, kappa_ut, kappa_l) ? c_l[step]    = 1.0
                                                      : c_l[step]    = -1.0;
  }

  auto L2 = convert2dense(L);
  auto U2 = convert2dense(U);
  for (int i = 0; i < n; ++i) L2[i][i] = U2[i][i] = 1.0;  // set unit diag
  const auto bl = dense_mv(L2, kappa_l);
#define TRAN_U2 true
  const auto bu = dense_mv(U2, kappa_ut, TRAN_U2);
#undef TRAN_U2
  for (int i = 0; i < n; ++i) EXPECT_NEAR(bl[i], c_l[i], tol);
  for (int i = 0; i < n; ++i) EXPECT_NEAR(bu[i], c_u[i], tol);
}
