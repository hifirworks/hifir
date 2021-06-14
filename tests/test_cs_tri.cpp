///////////////////////////////////////////////////////////////////////////////
//                  This file is part of the HIFIR library                   //
///////////////////////////////////////////////////////////////////////////////

#include "common.hpp"
// line break to avoid sorting
#include "hif/ds/Array.hpp"
#include "hif/ds/CompressedStorage.hpp"

#include <gtest/gtest.h>

using namespace hif;

constexpr static int N = 300;

const static RandIntGen  i_rand(5, N);
const static RandRealGen r_rand;

TEST(L, crs) {
  const int n(i_rand());
  using crs_t    = CRS<double, int>;
  auto       A_  = gen_rand_strict_lower_sparse<crs_t::other_type>(n);
  auto       A   = crs_t(A_);
  const auto A_d = convert2dense(A);
  auto       y1  = gen_ran_vec<crs_t::array_type>(n);
  auto       y2  = create_mat<double>(y1.size(), 1);
  for (int i = 0; i < n; ++i) y2[i][0] = y1[i];
  forward_sub_unit_diag(A_d, y2);
  A.solve_as_strict_lower(y1);
  for (int i = 0; i < n; ++i)
    EXPECT_NEAR(y1[i], y2[i][0], 1e-12 * (std::abs(y2[i][0] ? y2[i][0] : 1.0)))
        << i;
}

TEST(L_tran, crs) {
  const int n(i_rand());
  using crs_t = CRS<double, int>;
  auto A_     = gen_rand_strict_lower_sparse<crs_t::other_type>(n);
  auto A      = crs_t(A_);
  auto A_d    = convert2dense(A);
  // make transpose
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < i; ++j) {
      A_d[j][i] = A_d[i][j];
      A_d[i][j] = 0.0;
    }
  auto y1 = gen_ran_vec<crs_t::array_type>(n);
  auto y2 = create_mat<double>(y1.size(), 1);
  for (int i = 0; i < n; ++i) y2[i][0] = y1[i];
  backward_sub_unit_diag(A_d, y2);
  A.solve_as_strict_lower_tran(y1);
  for (int i = 0; i < n; ++i)
    EXPECT_NEAR(y1[i], y2[i][0], 1e-12 * (std::abs(y2[i][0] ? y2[i][0] : 1.0)))
        << i;
}

TEST(U, crs) {
  const int n(i_rand());
  using crs_t    = CRS<double, int>;
  auto       A   = gen_rand_strict_upper_sparse<crs_t>(n);
  const auto A_d = convert2dense(A);
  auto       y1  = gen_ran_vec<crs_t::array_type>(n);
  auto       y2  = create_mat<double>(y1.size(), 1);
  for (int i = 0; i < n; ++i) y2[i][0] = y1[i];
  backward_sub_unit_diag(A_d, y2);
  A.solve_as_strict_upper(y1);
  for (int i = 0; i < n; ++i)
    EXPECT_NEAR(y1[i], y2[i][0], 1e-12 * (std::abs(y2[i][0] ? y2[i][0] : 1.0)))
        << i;
}

TEST(U_tran, crs) {
  const int n(i_rand());
  using crs_t = CRS<double, int>;
  auto A      = gen_rand_strict_upper_sparse<crs_t>(n);
  auto A_d    = convert2dense(A);
  // make transpose
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < i; ++j) {
      A_d[i][j] = A_d[j][i];
      A_d[j][i] = 0.0;
    }
  auto y1 = gen_ran_vec<crs_t::array_type>(n);
  auto y2 = create_mat<double>(y1.size(), 1);
  for (int i = 0; i < n; ++i) y2[i][0] = y1[i];
  forward_sub_unit_diag(A_d, y2);
  A.solve_as_strict_upper_tran(y1);
  for (int i = 0; i < n; ++i)
    EXPECT_NEAR(y1[i], y2[i][0], 1e-10 * (std::abs(y2[i][0] ? y2[i][0] : 1.0)))
        << i;
}

TEST(L, ccs) {
  const int n(i_rand());
  using ccs_t    = CCS<double, int>;
  auto       A   = gen_rand_strict_lower_sparse<ccs_t>(n);
  const auto A_d = convert2dense(A);
  auto       y1  = gen_ran_vec<ccs_t::array_type>(n);
  auto       y2  = create_mat<double>(y1.size(), 1);
  for (int i = 0; i < n; ++i) y2[i][0] = y1[i];
  forward_sub_unit_diag(A_d, y2);
  A.solve_as_strict_lower(y1);
  for (int i = 0; i < n; ++i)
    EXPECT_NEAR(y1[i], y2[i][0], 1e-12 * (std::abs(y2[i][0] ? y2[i][0] : 1.0)))
        << i;
}

TEST(L_tran, ccs) {
  const int n(i_rand());
  using ccs_t = CCS<double, int>;
  auto A      = gen_rand_strict_lower_sparse<ccs_t>(n);
  auto A_d    = convert2dense(A);
  // make transpose
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < i; ++j) {
      A_d[j][i] = A_d[i][j];
      A_d[i][j] = 0.0;
    }
  auto y1 = gen_ran_vec<ccs_t::array_type>(n);
  auto y2 = create_mat<double>(y1.size(), 1);
  for (int i = 0; i < n; ++i) y2[i][0] = y1[i];
  backward_sub_unit_diag(A_d, y2);
  A.solve_as_strict_lower_tran(y1);
  for (int i = 0; i < n; ++i)
    EXPECT_NEAR(y1[i], y2[i][0], 1e-12 * (std::abs(y2[i][0] ? y2[i][0] : 1.0)))
        << i;
}

TEST(U, ccs) {
  const int n(i_rand());
  using ccs_t    = CCS<double, int>;
  auto       A_  = gen_rand_strict_upper_sparse<ccs_t::other_type>(n);
  auto       A   = ccs_t(A_);
  const auto A_d = convert2dense(A);
  auto       y1  = gen_ran_vec<ccs_t::array_type>(n);
  auto       y2  = create_mat<double>(y1.size(), 1);
  for (int i = 0; i < n; ++i) y2[i][0] = y1[i];
  backward_sub_unit_diag(A_d, y2);
  A.solve_as_strict_upper(y1);
  for (int i = 0; i < n; ++i)
    EXPECT_NEAR(y1[i], y2[i][0], 1e-12 * (std::abs(y2[i][0] ? y2[i][0] : 1.0)))
        << i;
}

TEST(U_tran, ccs) {
  const int n(i_rand());
  using ccs_t = CCS<double, int>;
  auto A_     = gen_rand_strict_upper_sparse<ccs_t::other_type>(n);
  auto A      = ccs_t(A_);
  auto A_d    = convert2dense(A);
  // make transpose
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < i; ++j) {
      A_d[i][j] = A_d[j][i];
      A_d[j][i] = 0.0;
    }
  auto y1 = gen_ran_vec<ccs_t::array_type>(n);
  auto y2 = create_mat<double>(y1.size(), 1);
  for (int i = 0; i < n; ++i) y2[i][0] = y1[i];
  forward_sub_unit_diag(A_d, y2);
  A.solve_as_strict_upper_tran(y1);
  for (int i = 0; i < n; ++i)
    EXPECT_NEAR(y1[i], y2[i][0], 1e-12 * (std::abs(y2[i][0] ? y2[i][0] : 1.0)))
        << i;
}
