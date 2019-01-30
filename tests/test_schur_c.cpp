//@HEADER
//----------------------------------------------------------------------------
//                Copyright (C) 2019 The PSMILU AUTHORS
//----------------------------------------------------------------------------
//@HEADER

// for unit testing, using full L and U, thus
#define PSMILU_SCHUR_C_USE_FULL

#include "common.hpp"
// line break to avoid sorting
#include "psmilu_Array.hpp"
#include "psmilu_CompressedStorage.hpp"
#include "psmilu_PermMatrix.hpp"
#include "psmilu_Schur.hpp"
#include "psmilu_SparseVec.hpp"

#include "psmilu_AugmentedStorage.hpp"

#include <gtest/gtest.h>

constexpr static auto tol = 1e-12;

using namespace psmilu;

static const RandIntGen  i_rand(0, 200);
static const RandRealGen r_rand;

TEST(Schur, c_ver) {
  using crs_t     = CRS<double, int>;
  using aug_crs_t = AugCRS<crs_t>;
  using aug_ccs_t = AugCCS<crs_t::other_type>;
  using perm_t    = BiPermMatrix<int>;

  const int n = i_rand() + 1;
  // create A
  crs_t A(n, n);
  auto  Ad = create_mat<double>(n, n);
  {
    std::vector<std::vector<int>> col_inds(n);
    std::vector<bool>             tags(n, false);
    for (int i = 0; i < n; ++i) {
      const int nnz     = i_rand() % n;
      auto&     col_ind = col_inds[i];
      int       counts = 0, guard = 0;
      for (;;) {
        if (counts == nnz || guard > 2 * nnz) break;
        const int idx = i_rand() % n;
        ++guard;
        if (tags[idx]) continue;
        ++counts;
        col_ind.push_back(idx);
        tags[idx] = true;
      }
      std::sort(col_ind.begin(), col_ind.end());
      for (const auto j : col_ind) tags[j] = false;
    }
    A.begin_assemble_rows();
    for (int i = 0; i < n; ++i) {
      const auto& col_ind = col_inds[i];
      for (const auto col : col_ind) Ad[i][col] = r_rand();
      A.push_back_row(i, col_ind.cbegin(), col_ind.cend(), Ad[i]);
      ASSERT_EQ(A.nnz_in_row(i), col_ind.size());
    }
    A.end_assemble_rows();
  }
  // create L
  aug_ccs_t L(n, n);
  {
    std::vector<std::vector<int>> row_inds(n);
    std::vector<bool>             tags(n, false);
    for (int i = 0; i < n; ++i) {
      const int nnz     = i_rand() % n;
      auto&     row_ind = row_inds[i];
      int       counts = 0, guard = 0;
      for (;;) {
        if (counts == nnz || guard > 2 * nnz) break;
        const int idx = i_rand() % n;
        ++guard;
        if (tags[idx]) continue;
        ++counts;
        row_ind.push_back(idx);
        tags[idx] = true;
      }
      std::sort(row_ind.begin(), row_ind.end());
      for (const auto j : row_ind) tags[j] = false;
    }
    std::vector<double> buf(n);
    L.begin_assemble_cols();
    for (int i = 0; i < n; ++i) {
      const auto& row_ind = row_inds[i];
      for (const auto row : row_ind) buf[row] = r_rand();
      L.push_back_col(i, row_ind.cbegin(), row_ind.cend(), buf);
      ASSERT_EQ(L.nnz_in_col(i), row_ind.size());
    }
    L.end_assemble_cols();
  }
  const auto Ld = convert2dense(L);
  // create U
  aug_crs_t U(n, n);
  auto      Ud = create_mat<double>(n, n);
  {
    std::vector<std::vector<int>> col_inds(n);
    std::vector<bool>             tags(n, false);
    for (int i = 0; i < n; ++i) {
      const int nnz     = i_rand() % n;
      auto&     col_ind = col_inds[i];
      int       counts = 0, guard = 0;
      for (;;) {
        if (counts == nnz || guard > 2 * nnz) break;
        const int idx = i_rand() % n;
        ++guard;
        if (tags[idx]) continue;
        ++counts;
        col_ind.push_back(idx);
        tags[idx] = true;
      }
      std::sort(col_ind.begin(), col_ind.end());
      for (const auto j : col_ind) tags[j] = false;
    }
    U.begin_assemble_rows();
    for (int i = 0; i < n; ++i) {
      const auto& col_ind = col_inds[i];
      for (const auto col : col_ind) Ud[i][col] = r_rand();
      U.push_back_row(i, col_ind.cbegin(), col_ind.cend(), Ud[i]);
      ASSERT_EQ(U.nnz_in_row(i), col_ind.size());
    }
    U.end_assemble_rows();
  }

  const int m = i_rand() % n;
  const int N = n - m;
  perm_t    p(n), q(n);
  p.make_eye();
  q.make_eye();
  Array<int> U_start(n);
  auto       first = U.col_ind().cbegin();
  for (int i = 0; i < n; ++i)
    U_start[i] =
        std::lower_bound(U.col_ind_cbegin(i), U.col_ind_cend(i), m) - first;

  Array<double> d(n);
  for (auto& v : d) v = r_rand();
  crs_t Sc;
  compute_Schur_C(A, p, q, m, n, L, d, U, U_start, Sc);
  ASSERT_EQ((int)Sc.nrows(), N);
  ASSERT_EQ((int)Sc.ncols(), N);

  const auto Scd1 = convert2dense(Sc);

  const auto Scd2 = compute_dense_Schur_c(Ad, Ld, Ud, d, m);
  //   const auto print = [](const matrix<double>& m) {
  //     for (int i = 0; i < (int)m.size(); ++i) {
  //       for (int j = 0; j < (int)m.size(); ++j) std::cout << m[i][j] << ' ';
  //       std::cout << '\n';
  //     }
  //   };
  COMPARE_MATS_TOL(Scd1, Scd2, tol);
}
