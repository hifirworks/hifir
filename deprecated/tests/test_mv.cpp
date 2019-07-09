//@HEADER
//----------------------------------------------------------------------------
//                Copyright (C) 2019 The PSMILU AUTHORS
//----------------------------------------------------------------------------
//@HEADER

#include "common.hpp"
// line break to avoid sorting
#include "psmilu_Array.hpp"
#include "psmilu_CompressedStorage.hpp"

#include <gtest/gtest.h>

using namespace psmilu;

constexpr static int N = 100;

const static RandIntGen  i_rand(1, N);
const static RandRealGen r_rand;

TEST(MV, crs) {
  {
    const int m(i_rand()), n(i_rand());
    using crs_t     = CRS<double, int>;
    const crs_t A   = gen_rand_sparse<crs_t>(m, n);
    const auto  A_d = convert2dense(A);
    {
      const auto    x = gen_ran_vec<Array<double>>(n);
      Array<double> y1(m);
      A.mv(x, y1);
      const auto y2 = dense_mv(A_d, x);
      for (int i = 0; i < m; ++i) EXPECT_NEAR(y1[i], y2[i], 1e-12);
    }
    {
      const auto    x = gen_ran_vec<Array<double>>(m);
      Array<double> y1(n);
      A.mv(x, y1, true);
      const auto y2 = dense_mv(A_d, x, true);
      for (int i = 0; i < n; ++i) EXPECT_NEAR(y1[i], y2[i], 1e-12);
    }
  }
  {
    const int m(i_rand()), n(i_rand());
    using crs_t     = CRS<double, int, true>;
    const crs_t A   = gen_rand_sparse<crs_t>(m, n);
    const auto  A_d = convert2dense(A);
    {
      const auto    x = gen_ran_vec<Array<double>>(n);
      Array<double> y1(m);
      A.mv(x, y1);
      const auto y2 = dense_mv(A_d, x);
      for (int i = 0; i < m; ++i) EXPECT_NEAR(y1[i], y2[i], 1e-12);
    }
    {
      const auto    x = gen_ran_vec<Array<double>>(m);
      Array<double> y1(n);
      A.mv(x, y1, true);
      const auto y2 = dense_mv(A_d, x, true);
      for (int i = 0; i < n; ++i) EXPECT_NEAR(y1[i], y2[i], 1e-12);
    }
  }
}

TEST(MV, ccs) {
  {
    const int m(i_rand()), n(i_rand());
    using ccs_t     = CCS<double, int>;
    const ccs_t A   = gen_rand_sparse<ccs_t>(m, n);
    const auto  A_d = convert2dense(A);
    {
      const auto    x = gen_ran_vec<Array<double>>(n);
      Array<double> y1(m);
      A.mv(x, y1);
      const auto y2 = dense_mv(A_d, x);
      for (int i = 0; i < m; ++i) EXPECT_NEAR(y1[i], y2[i], 1e-12);
    }
    {
      const auto    x = gen_ran_vec<Array<double>>(m);
      Array<double> y1(n);
      A.mv(x, y1, true);
      const auto y2 = dense_mv(A_d, x, true);
      for (int i = 0; i < n; ++i) EXPECT_NEAR(y1[i], y2[i], 1e-12);
    }
  }
  {
    const int m(i_rand()), n(i_rand());
    using ccs_t     = CCS<double, int, true>;
    const ccs_t A   = gen_rand_sparse<ccs_t>(m, n);
    const auto  A_d = convert2dense(A);
    {
      const auto    x = gen_ran_vec<Array<double>>(n);
      Array<double> y1(m);
      A.mv(x, y1);
      const auto y2 = dense_mv(A_d, x);
      for (int i = 0; i < m; ++i) EXPECT_NEAR(y1[i], y2[i], 1e-12);
    }
    {
      const auto    x = gen_ran_vec<Array<double>>(m);
      Array<double> y1(n);
      A.mv(x, y1, true);
      const auto y2 = dense_mv(A_d, x, true);
      for (int i = 0; i < n; ++i) EXPECT_NEAR(y1[i], y2[i], 1e-12);
    }
  }
}
