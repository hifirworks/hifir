///////////////////////////////////////////////////////////////////////////////
//                  This file is part of HIF project                         //
///////////////////////////////////////////////////////////////////////////////

#include "common.hpp"
// line break to avoid sorting
#include "hif/ds/Array.hpp"
#include "hif/ds/IntervalCompressedStorage.hpp"

#include <gtest/gtest.h>

using namespace hif;

constexpr static int N = 300;

const static RandIntGen  i_rand(5, N);
const static RandRealGen r_rand;

TEST(L, crs) {
  const int n(i_rand());
  using crs_t    = IntervalCRS<double, int>;
  auto       A_  = gen_rand_strict_lower_sparse<crs_t::ccs_type>(n);
  const auto A_d = convert2dense(A_);
  auto       y1  = gen_ran_vec<crs_t::array_type>(n);
  auto       y2  = create_mat<double>(y1.size(), 1);
  for (int i = 0; i < n; ++i) y2[i][0] = y1[i];
  forward_sub_unit_diag(A_d, y2);
  crs_t A(A_, false);
  A.solve_as_strict_lower(y1);
  for (int i = 0; i < n; ++i)
    EXPECT_NEAR(y1[i], y2[i][0], 1e-12 * (std::abs(y2[i][0] ? y2[i][0] : 1.0)))
        << i;
}

TEST(U, crs) {
  const int n(i_rand());
  using crs_t    = IntervalCRS<double, int>;
  auto       A_  = gen_rand_strict_upper_sparse<crs_t::crs_type>(n);
  const auto A_d = convert2dense(A_);
  auto       y1  = gen_ran_vec<crs_t::array_type>(n);
  auto       y2  = create_mat<double>(y1.size(), 1);
  for (int i = 0; i < n; ++i) y2[i][0] = y1[i];
  backward_sub_unit_diag(A_d, y2);
  crs_t A(std::move(A_), false);
  A.solve_as_strict_upper(y1);
  for (int i = 0; i < n; ++i)
    EXPECT_NEAR(y1[i], y2[i][0], 1e-12 * (std::abs(y2[i][0] ? y2[i][0] : 1.0)))
        << i;
}

TEST(L, ccs) {
  const int n(i_rand());
  using ccs_t    = IntervalCCS<double, int>;
  auto       A_  = gen_rand_strict_lower_sparse<ccs_t::ccs_type>(n);
  const auto A_d = convert2dense(A_);
  auto       y1  = gen_ran_vec<ccs_t::array_type>(n);
  auto       y2  = create_mat<double>(y1.size(), 1);
  for (int i = 0; i < n; ++i) y2[i][0] = y1[i];
  forward_sub_unit_diag(A_d, y2);
  ccs_t A(std::move(A_), false);
  A.solve_as_strict_lower(y1);
  for (int i = 0; i < n; ++i)
    EXPECT_NEAR(y1[i], y2[i][0], 1e-12 * (std::abs(y2[i][0] ? y2[i][0] : 1.0)))
        << i;
}

TEST(U, ccs) {
  const int n(i_rand());
  using ccs_t    = IntervalCCS<double, int>;
  auto       A_  = gen_rand_strict_upper_sparse<ccs_t::crs_type>(n);
  const auto A_d = convert2dense(A_);
  auto       y1  = gen_ran_vec<ccs_t::array_type>(n);
  auto       y2  = create_mat<double>(y1.size(), 1);
  for (int i = 0; i < n; ++i) y2[i][0] = y1[i];
  backward_sub_unit_diag(A_d, y2);
  ccs_t A(A_, false);
  A.solve_as_strict_upper(y1);
  for (int i = 0; i < n; ++i)
    EXPECT_NEAR(y1[i], y2[i][0], 1e-12 * (std::abs(y2[i][0] ? y2[i][0] : 1.0)))
        << i;
}
