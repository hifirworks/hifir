///////////////////////////////////////////////////////////////////////////////
//                  This file is part of the HIFIR library                         //
///////////////////////////////////////////////////////////////////////////////

#include "common.hpp"
// line break to avoid sorting
#include "hif/ds/Array.hpp"
#include "hif/ds/IntervalCompressedStorage.hpp"

#include <gtest/gtest.h>

using namespace hif;

constexpr static int N = 300;

const static RandIntGen  i_rand(1, N);
const static RandRealGen r_rand;

TEST(MV, crs) {
  do {
    const int m(i_rand()), n(i_rand());
    using crs_t     = IntervalCRS<double, int>;
    auto        A_  = gen_rand_sparse<crs_t::crs_type>(m, n);
    const auto  A_d = convert2dense(A_);
    const crs_t A(std::move(A_), false);
    std::cout << A.storage_cost_ratio() << std::endl;
    if (1) {
      const auto    x = gen_ran_vec<Array<double>>(n);
      Array<double> y1(m);
      A.mv(x, y1);
      const auto y2 = dense_mv(A_d, x);
      for (int i = 0; i < m; ++i) EXPECT_NEAR(y1[i], y2[i], 1e-12) << i;
    }
    if (1) {
      const auto    x = gen_ran_vec<Array<double>>(m);
      Array<double> y1(n);
      A.mv(x, y1, true);
      const auto y2 = dense_mv(A_d, x, true);
      for (int i = 0; i < n; ++i) EXPECT_NEAR(y1[i], y2[i], 1e-12) << i;
    }
  } while (0);
}

TEST(MV, ccs) {
  do {
    const int m(i_rand()), n(i_rand());
    using ccs_t     = IntervalCCS<double, int>;
    auto        A_  = gen_rand_sparse<ccs_t::ccs_type>(m, n);
    const auto  A_d = convert2dense(A_);
    const ccs_t A(std::move(A_), false);
    std::cout << A.storage_cost_ratio() << std::endl;
    if (1) {
      const auto    x = gen_ran_vec<Array<double>>(n);
      Array<double> y1(m);
      A.mv(x, y1);
      const auto y2 = dense_mv(A_d, x);
      for (int i = 0; i < m; ++i) EXPECT_NEAR(y1[i], y2[i], 1e-12) << i;
    }
    if (1) {
      const auto    x = gen_ran_vec<Array<double>>(m);
      Array<double> y1(n);
      A.mv(x, y1, true);
      const auto y2 = dense_mv(A_d, x, true);
      for (int i = 0; i < n; ++i) EXPECT_NEAR(y1[i], y2[i], 1e-12) << i;
    }
  } while (0);
}
