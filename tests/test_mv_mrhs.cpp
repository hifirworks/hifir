///////////////////////////////////////////////////////////////////////////////
//                  This file is part of HIF project                         //
///////////////////////////////////////////////////////////////////////////////

#include "common.hpp"
// line break to avoid sorting
#include "hif/ds/Array.hpp"
#include "hif/ds/CompressedStorage.hpp"

#include <gtest/gtest.h>

using namespace hif;

constexpr static int N = 100;

const static RandIntGen  i_rand(1, N);
const static RandRealGen r_rand;

TEST(MV, crs) {
  do {
    const int m(i_rand()), n(i_rand());
    using crs_t     = CRS<double, int>;
    const crs_t A   = gen_rand_sparse<crs_t>(m, n);
    const auto  A_d = convert2dense(A);
    if (1) {
      const auto                   x1 = gen_ran_vec<Array<double>>(n * 2);
      Array<std::array<double, 2>> x(n);
      for (int i = 0; i < n; ++i) {
        x[i][0] = x1[i];
        x[i][1] = x1[i + n];
      }
      Array<std::array<double, 2>> y1(m);
      A.mv_mrhs_nt(x, y1);
      std::vector<double> x2(x1.cbegin(), x1.cbegin() + n);
      const auto          y2 = dense_mv(A_d, x2);
      std::vector<double> x3(x1.cbegin() + n, x1.cend());
      const auto          y3 = dense_mv(A_d, x3);
      for (int i = 0; i < m; ++i) {
        EXPECT_NEAR(y1[i][0], y2[i], 1e-12);
        EXPECT_NEAR(y1[i][1], y3[i], 1e-12);
      }
    }
    if (1) {
      const auto                   x1 = gen_ran_vec<Array<double>>(m * 2);
      Array<std::array<double, 2>> x(m);
      for (int i = 0; i < m; ++i) {
        x[i][0] = x1[i];
        x[i][1] = x1[i + m];
      }
      Array<std::array<double, 2>> y1(n);
      A.mv_mrhs(x, y1, true);
      std::vector<double> x2(x1.cbegin(), x1.cbegin() + m);
      const auto          y2 = dense_mv(A_d, x2, true);
      std::vector<double> x3(x1.cbegin() + m, x1.cend());
      const auto          y3 = dense_mv(A_d, x3, true);
      for (int i = 0; i < n; ++i) {
        EXPECT_NEAR(y1[i][0], y2[i], 1e-12);
        EXPECT_NEAR(y1[i][1], y3[i], 1e-12);
      }
    }
  } while (0);
}

TEST(MV, ccs) {
  do {
    const int m(i_rand()), n(i_rand());
    using ccs_t     = CCS<double, int>;
    const ccs_t A   = gen_rand_sparse<ccs_t>(m, n);
    const auto  A_d = convert2dense(A);
    if (1) {
      const auto                   x1 = gen_ran_vec<Array<double>>(n * 2);
      Array<std::array<double, 2>> x(n);
      for (int i = 0; i < n; ++i) {
        x[i][0] = x1[i];
        x[i][1] = x1[i + n];
      }
      Array<std::array<double, 2>> y1(m);
      A.mv_mrhs_nt(x, y1);
      std::vector<double> x2(x1.cbegin(), x1.cbegin() + n);
      const auto          y2 = dense_mv(A_d, x2);
      std::vector<double> x3(x1.cbegin() + n, x1.cend());
      const auto          y3 = dense_mv(A_d, x3);
      for (int i = 0; i < m; ++i) {
        EXPECT_NEAR(y1[i][0], y2[i], 1e-12);
        EXPECT_NEAR(y1[i][1], y3[i], 1e-12);
      }
    }
    if (2) {
      const auto                   x1 = gen_ran_vec<Array<double>>(m * 2);
      Array<std::array<double, 2>> x(m);
      for (int i = 0; i < m; ++i) {
        x[i][0] = x1[i];
        x[i][1] = x1[i + m];
      }
      Array<std::array<double, 2>> y1(n);
      A.mv_mrhs(x, y1, true);
      std::vector<double> x2(x1.cbegin(), x1.cbegin() + m);
      const auto          y2 = dense_mv(A_d, x2, true);
      std::vector<double> x3(x1.cbegin() + m, x1.cend());
      const auto          y3 = dense_mv(A_d, x3, true);
      for (int i = 0; i < n; ++i) {
        EXPECT_NEAR(y1[i][0], y2[i], 1e-12);
        EXPECT_NEAR(y1[i][1], y3[i], 1e-12);
      }
    }
  } while (0);
}