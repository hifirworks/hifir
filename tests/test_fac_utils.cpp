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
#include "psmilu_fac.hpp"

#include <gtest/gtest.h>

using namespace psmilu;

TEST(FAC_U, diag) {
  {
    using ccs_t         = CCS<double, int>;
    const auto        A = gen_rand_sparse<ccs_t>(100, 100);
    const int         m = RandIntGen(1, 99)();
    BiPermMatrix<int> p(100), q(100);
    p.make_eye();
    q.make_eye();
    std::shuffle(p().begin(), p().end(), std::mt19937_64(std::time(0)));
    std::shuffle(q().begin(), q().end(), std::mt19937_64(std::time(0)));
    p.build_inv();
    q.build_inv();
    const auto d   = internal::extract_perm_diag(A, m, p, q);
    const auto A_d = convert2dense(A);
    for (int i = 0; i < m; ++i) EXPECT_EQ(d[i], A_d[p[i]][q[i]]);
  }
  {
    using ccs_t         = CCS<double, int, true>;
    const auto        A = gen_rand_sparse<ccs_t>(100, 100);
    const int         m = RandIntGen(1, 99)();
    BiPermMatrix<int> p(100), q(100);
    p.make_eye();
    q.make_eye();
    std::shuffle(p().begin(), p().end(), std::mt19937_64(std::time(0)));
    std::shuffle(q().begin(), q().end(), std::mt19937_64(std::time(0)));
    p.build_inv();
    q.build_inv();
    const auto d   = internal::extract_perm_diag(A, m, p, q);
    const auto A_d = convert2dense(A);
    for (int i = 0; i < m; ++i) EXPECT_EQ(d[i], A_d[p[i]][q[i]]);
  }
}

TEST(FAC_U, E) {
  {
    using crs_t         = CRS<double, int>;
    const auto        A = gen_rand_sparse<crs_t>(100, 100);
    const int         m = RandIntGen(1, 99)();
    BiPermMatrix<int> p(100), q(100);
    p.make_eye();
    q.make_eye();
    std::shuffle(p().begin(), p().end(), std::mt19937_64(std::time(0)));
    std::shuffle(q().begin(), q().end(), std::mt19937_64(std::time(0)));
    p.build_inv();
    q.build_inv();
    const auto s = gen_ran_vec<Array<double>>(100);
    const auto t = gen_ran_vec<Array<double>>(100);
    const auto E = internal::extract_E(s, A, t, m, p, q);
    ASSERT_EQ((int)E.nrows(), 100 - m);
    ASSERT_EQ((int)E.ncols(), m);
    const auto E_d = convert2dense(E);
    const auto A_d = convert2dense(A);
    for (int i = 0; i < 100 - m; ++i)
      for (int j = 0; j < m; ++j)
        EXPECT_NEAR(E_d[i][j], s[p[i + m]] * A_d[p[i + m]][q[j]] * t[q[j]],
                    1e-12);
  }

  {
    using crs_t         = CRS<double, int, true>;
    const auto        A = gen_rand_sparse<crs_t>(100, 100);
    const int         m = RandIntGen(1, 99)();
    BiPermMatrix<int> p(100), q(100);
    p.make_eye();
    q.make_eye();
    std::shuffle(p().begin(), p().end(), std::mt19937_64(std::time(0)));
    std::shuffle(q().begin(), q().end(), std::mt19937_64(std::time(0)));
    p.build_inv();
    q.build_inv();
    const auto s = gen_ran_vec<Array<double>>(100);
    const auto t = gen_ran_vec<Array<double>>(100);
    const auto E = internal::extract_E(s, A, t, m, p, q);
    ASSERT_EQ((int)E.nrows(), 100 - m);
    ASSERT_EQ((int)E.ncols(), m);
    const auto E_d = convert2dense(E);
    const auto A_d = convert2dense(A);
    for (int i = 0; i < 100 - m; ++i)
      for (int j = 0; j < m; ++j)
        EXPECT_NEAR(E_d[i][j], s[p[i + m]] * A_d[p[i + m]][q[j]] * t[q[j]],
                    1e-12);
  }
}

TEST(FAC_U, F) {
  {
    using ccs_t         = CCS<double, int>;
    const auto        A = gen_rand_sparse<ccs_t>(100, 100);
    const int         m = RandIntGen(1, 99)();
    BiPermMatrix<int> p(100), q(100);
    p.make_eye();
    q.make_eye();
    std::shuffle(p().begin(), p().end(), std::mt19937_64(std::time(0)));
    std::shuffle(q().begin(), q().end(), std::mt19937_64(std::time(0)));
    p.build_inv();
    q.build_inv();
    const auto          s = gen_ran_vec<Array<double>>(100);
    const auto          t = gen_ran_vec<Array<double>>(100);
    std::vector<double> buf(100);
    const auto          F = internal::extract_F(s, A, t, m, p, q, buf);
    ASSERT_EQ((int)F.nrows(), m);
    ASSERT_EQ((int)F.ncols(), 100 - m);
    const auto F_d = convert2dense(F);
    const auto A_d = convert2dense(A);
    for (int i = 0; i < m; ++i)
      for (int j = 0; j < 100 - m; ++j)
        EXPECT_NEAR(F_d[i][j], s[p[i]] * A_d[p[i]][q[j + m]] * t[q[j + m]],
                    1e-12);
  }

  {
    using ccs_t         = CCS<double, int, true>;
    const auto        A = gen_rand_sparse<ccs_t>(100, 100);
    const int         m = RandIntGen(1, 99)();
    BiPermMatrix<int> p(100), q(100);
    p.make_eye();
    q.make_eye();
    std::shuffle(p().begin(), p().end(), std::mt19937_64(std::time(0)));
    std::shuffle(q().begin(), q().end(), std::mt19937_64(std::time(0)));
    p.build_inv();
    q.build_inv();
    const auto          s = gen_ran_vec<Array<double>>(100);
    const auto          t = gen_ran_vec<Array<double>>(100);
    std::vector<double> buf(100);
    const auto          F = internal::extract_F(s, A, t, m, p, q, buf);
    ASSERT_EQ((int)F.nrows(), m);
    ASSERT_EQ((int)F.ncols(), 100 - m);
    const auto F_d = convert2dense(F);
    const auto A_d = convert2dense(A);
    for (int i = 0; i < m; ++i)
      for (int j = 0; j < 100 - m; ++j)
        EXPECT_NEAR(F_d[i][j], s[p[i]] * A_d[p[i]][q[j + m]] * t[q[j + m]],
                    1e-12);
  }
}

TEST(FAC_U, L_B) {
  {
    using ccs_t    = CCS<double, int>;
    const auto A   = gen_rand_strict_lower_sparse<ccs_t>(100);
    const int  m   = RandIntGen(1, 99)();
    const auto A_d = convert2dense(A);
    for (int i = 0; i < 100; ++i)
      for (int j = i; j < 100; ++j) EXPECT_EQ(A_d[i][j], 0);

    Array<int> L_start(100);
    auto       first = A.row_ind().cbegin();
    for (int i = 0; i < m; ++i)
      L_start[i] =
          std::lower_bound(A.row_ind_cbegin(i), A.row_ind_cend(i), m) - first;

    const AugCCS<ccs_t> &L = static_cast<const AugCCS<ccs_t> &>(A);

    const auto L_B   = internal::extract_L_B(L, m, L_start);
    const auto L_B_d = convert2dense(L_B);
    for (int i = 0; i < m; ++i)
      for (int j = 0; j < m; ++j)
        EXPECT_EQ(A_d[i][j], L_B_d[i][j])
            << i << ',' << j << " entry should be equal to each other\n";
  }
  {
    using ccs_t    = CCS<double, int, true>;
    const auto A   = gen_rand_strict_lower_sparse<ccs_t>(100);
    const int  m   = RandIntGen(1, 99)();
    const auto A_d = convert2dense(A);
    for (int i = 0; i < 100; ++i)
      for (int j = i; j < 100; ++j) EXPECT_EQ(A_d[i][j], 0);

    Array<int> L_start(100);
    auto       first = A.row_ind().cbegin();
    for (int i = 0; i < m; ++i)
      L_start[i] =
          std::lower_bound(A.row_ind_cbegin(i), A.row_ind_cend(i), m + 1) -
          first;

    const AugCCS<ccs_t> &L = static_cast<const AugCCS<ccs_t> &>(A);

    const auto L_B = internal::extract_L_B(L, m, L_start);
    for (const auto v : L_B.row_ind()) std::cout << v << std::endl;
    const auto L_B_d = convert2dense(L_B);
    for (int i = 0; i < m; ++i)
      for (int j = 0; j < m; ++j)
        EXPECT_EQ(A_d[i][j], L_B_d[i][j])
            << i << ',' << j << " entry should be equal to each other\n";
  }
}

TEST(FAC_U, U_B) {
  {
    using crs_t    = CRS<double, int>;
    const auto A   = gen_rand_strict_upper_sparse<crs_t>(100);
    const int  m   = RandIntGen(1, 99)();
    const auto A_d = convert2dense(A);
    for (int i = 0; i < 100; ++i)
      for (int j = 0; j <= i; ++j) EXPECT_EQ(A_d[i][j], 0);

    Array<int> U_start(100);
    auto       first = A.col_ind().cbegin();
    for (int i = 0; i < m; ++i)
      U_start[i] =
          std::lower_bound(A.col_ind_cbegin(i), A.col_ind_cend(i), m) - first;

    const AugCRS<crs_t> &U = static_cast<const AugCRS<crs_t> &>(A);

    const auto U_B   = internal::extract_U_B(U, m, U_start);
    const auto U_B_d = convert2dense(U_B);
    for (int i = 0; i < m; ++i)
      for (int j = 0; j < m; ++j)
        EXPECT_EQ(A_d[i][j], U_B_d[i][j])
            << i << ',' << j << " entry should be equal to each other\n";
  }

  {
    using crs_t    = CRS<double, int, true>;
    const auto A   = gen_rand_strict_upper_sparse<crs_t>(100);
    const int  m   = RandIntGen(1, 99)();
    const auto A_d = convert2dense(A);
    for (int i = 0; i < 100; ++i)
      for (int j = 0; j <= i; ++j) EXPECT_EQ(A_d[i][j], 0);

    Array<int> U_start(100);
    auto       first = A.col_ind().cbegin();
    for (int i = 0; i < m; ++i)
      U_start[i] =
          std::lower_bound(A.col_ind_cbegin(i), A.col_ind_cend(i), m + 1) -
          first;

    const AugCRS<crs_t> &U = static_cast<const AugCRS<crs_t> &>(A);

    const auto U_B   = internal::extract_U_B(U, m, U_start);
    const auto U_B_d = convert2dense(U_B);
    for (int i = 0; i < m; ++i)
      for (int j = 0; j < m; ++j)
        EXPECT_EQ(A_d[i][j], U_B_d[i][j])
            << i << ',' << j << " entry should be equal to each other\n";
  }
}

TEST(FAC_U, select) {
  using crs_t = CRS<double, int>;
  using ccs_t = CCS<double, int>;
  crs_t A;
  ccs_t B;

  using s1_t = internal::CompressedTypeTrait<crs_t, ccs_t>;
  static_assert(std::is_same<s1_t::crs_type, crs_t>::value, "must be same");
  static_assert(std::is_same<s1_t::ccs_type, ccs_t>::value, "must be same");
  const auto &crs1 = s1_t::select_crs(A, B);
  ASSERT_EQ(&crs1, &A);
  const auto &ccs1 = s1_t::select_ccs(A, B);
  ASSERT_EQ(&ccs1, &B);

  using s2_t = internal::CompressedTypeTrait<ccs_t, crs_t>;
  static_assert(std::is_same<s2_t::crs_type, crs_t>::value, "must be same");
  static_assert(std::is_same<s2_t::ccs_type, ccs_t>::value, "must be same");
  const auto &crs2 = s2_t::select_crs(B, A);
  ASSERT_EQ(&crs2, &A);
  const auto &ccs2 = s2_t::select_ccs(B, A);
  ASSERT_EQ(&ccs2, &B);
}
