//@HEADER
//----------------------------------------------------------------------------
//                Copyright (C) 2019 The PSMILU AUTHORS
//----------------------------------------------------------------------------
//@HEADER

// for unit testing, using full L and U, thus
#define PSMILU_SCHUR_USE_FULL

#include "common.hpp"
// line break to avoid sorting
#include "psmilu_Array.hpp"
#include "psmilu_CompressedStorage.hpp"
#include "psmilu_DenseMatrix.hpp"
#include "psmilu_PermMatrix.hpp"
#include "psmilu_Schur.hpp"
#include "psmilu_SparseVec.hpp"

#include "psmilu_AugmentedStorage.hpp"

#include <gtest/gtest.h>

#include <iomanip>

constexpr static auto tol = 1e-12;

using namespace psmilu;

static const RandIntGen  i_rand(0, 200);
static const RandRealGen r_rand;

TEST(Schur, h_ver) {
  using crs_t     = CRS<double, int>;
  using aug_crs_t = AugCRS<crs_t>;
  using ccs_t     = crs_t::other_type;
  using aug_ccs_t = AugCCS<ccs_t>;
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

  int m = i_rand() % n;
  if (!m) ++m;
  const int N = n - m;
  perm_t    p(n), q(n);
  p.make_eye();
  q.make_eye();
  Array<int> U_start(n);
  auto       first = U.col_ind().cbegin();
  for (int i = 0; i < n; ++i)
    U_start[i] =
        std::lower_bound(U.col_ind_cbegin(i), U.col_ind_cend(i), m) - first;
  Array<int> L_start(n);
  {
    auto first = L.row_ind().cbegin();
    for (int i = 0; i < n; ++i)
      L_start[i] =
          std::lower_bound(L.row_ind_cbegin(i), L.row_ind_cend(i), m) - first;
  }

  Array<double> d(n);
  for (auto& v : d) v = r_rand();
  crs_t         Sc;
  Array<double> s(n, 1.0), t(n, 1.0);

  DenseMatrix<double> HC(N, N, 0.0);

  // extract the lower part
  ccs_t L_B(m, m);
  {
    auto  first     = L.row_ind().cbegin();
    int   nnz       = 0;
    auto& col_start = L_B.col_start();
    col_start.resize(m + 1);
    col_start.front() = 0;
    for (int i = 0; i < m; ++i) {
      nnz += (first + L_start[i]) - L.row_ind_cbegin(i);
      col_start[i + 1] = nnz;
    }

    L_B.reserve(nnz);
    auto& row_ind = L_B.row_ind();
    auto& vals    = L_B.vals();
    row_ind.resize(nnz);
    vals.resize(nnz);
    auto itr   = row_ind.begin();
    auto v_itr = vals.begin();
    for (int i = 0; i < m; ++i) {
      itr   = std::copy(L.row_ind_cbegin(i), first + L_start[i], itr);
      v_itr = std::copy(L.val_cbegin(i), L.vals().cbegin() + L_start[i], v_itr);
    }
  }

  // extract the upper part
  ccs_t U_B(m, m);
  {
    auto  first     = U.col_ind().cbegin();
    auto& col_start = U_B.col_start();
    col_start.resize(m + 1);
    std::fill_n(col_start.begin(), m + 1, 0);
    for (int i = 0; i < m; ++i)
      std::for_each(U.col_ind_cbegin(i), first + U_start[i],
                    [&](const int j) { ++col_start[j + 1]; });
    for (int i = 0; i < m; ++i) col_start[i + 1] += col_start[i];
    U_B.reserve(col_start[m]);
    auto& row_ind = U_B.row_ind();
    auto& vals    = U_B.vals();
    row_ind.resize(col_start[m]);
    vals.resize(col_start[m]);
    for (int i = 0; i < m; ++i) {
      auto itr   = U.col_ind_cbegin(i);
      auto last  = first + U_start[i];
      auto v_itr = U.val_cbegin(i);
      for (; itr != last; ++itr, ++v_itr) {
        row_ind[col_start[*itr]] = i;
        vals[col_start[*itr]++]  = *v_itr;
      }
    }
    if (m) {
      ASSERT_EQ(col_start[m], col_start[m - 1]);
    }
    int temp = 0;
    for (int i = 0; i < m; ++i) std::swap(col_start[i], temp);
  }
  const ccs_t AA(A);
  ccs_t       t_e, t_f;
  compute_Schur_H(L, L_start, L_B, s, AA, t, p, q, d, U_B, U, HC, t_e, t_f);

  auto L_E = create_mat<double>(N, m);
  {
    auto first   = L.row_ind().cbegin();
    auto v_first = L.vals().cbegin();
    for (int i = 0; i < m; ++i) {
      auto v_itr = v_first + L_start[i];
      for (auto itr = first + L_start[i]; itr != L.row_ind_cend(i);
           ++itr, ++v_itr)
        L_E[*itr - m][i] = *v_itr;
    }
  }
  auto U_F = create_mat<double>(m, N);
  {
    auto first   = U.col_ind().cbegin();
    auto v_first = U.vals().cbegin();
    for (int j = 0; j < m; ++j) {
      auto v_itr = v_first + U_start[j];
      for (auto itr = first + U_start[j]; itr != U.col_ind_cend(j);
           ++itr, ++v_itr)
        U_F[j][*itr - m] = *v_itr;
    }
  }

  const auto HC_d = compute_dense_Schur_h(Ad, m, convert2dense(L_B),
                                          convert2dense(U_B), d, L_E, U_F);

  const auto HC_dd = from_gen_dense(HC);

  COMPARE_MATS_TOL(HC_d, HC_dd, tol);

  // std::cout << std::setprecision(6) << std::fixed;
  // const auto print_m = [](const decltype(HC_d)& m) {
  //   for (int i = 0; i < m.size(); ++i) {
  //     for (int j = 0; j < m.front().size(); ++j) std::cout << m[i][j] << ' ';
  //     std::cout << '\n';
  //   }
  // };
  // std::cout << "HC_d=\n";
  // print_m(HC_d);
  // std::cout << "HC_dd=\n";
  // print_m(HC_dd);
  // const auto t_e1 = convert2dense(t_e), t_f1 = convert2dense(t_f);
  // const auto t_e2 = compute_dense_Schur_h_t_e(Ad, m, convert2dense(L_B),
  //                                             convert2dense(U_B), d, L_E),
  //            t_f2 = compute_dense_Schur_h_t_f(convert2dense(U_B), U_F);
  // std::cout << "t_e1=\n";
  // print_m(t_e1);
  // std::cout << "t_e2=\n";
  // print_m(t_e2);
  // std::cout << "t_f1=\n";
  // print_m(t_f1);
  // std::cout << "t_f2=\n";
  // print_m(t_f2);
}
