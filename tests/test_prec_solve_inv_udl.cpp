//@HEADER
//----------------------------------------------------------------------------
//                Copyright (C) 2019 The PSMILU AUTHORS
//----------------------------------------------------------------------------
//@HEADER

#include "common.hpp"
// line break to avoid sorting
#include "psmilu_Array.hpp"
#include "psmilu_CompressedStorage.hpp"
#include "psmilu_prec_solve.hpp"

#include <gtest/gtest.h>

using namespace psmilu;

TEST(INV_UDL, c) {
  using ccs_t   = CCS<double, int>;
  const auto L  = gen_rand_strict_lower_sparse<ccs_t>(100);
  const auto U  = ccs_t(gen_rand_strict_upper_sparse<ccs_t::other_type>(100));
  using array_t = Array<double>;
  const auto D  = gen_ran_vec<array_t>(100);
  auto       y1 = gen_ran_vec<array_t>(100);
  auto       y2 = create_mat<double>(100, 1);
  for (int i = 0; i < 100; ++i) y2[i][0] = y1[i];
  internal::prec_solve_udl_inv(U, D, L, y1);
  const auto L1 = convert2dense(L), U2 = convert2dense(U);
  forward_sub_unit_diag(L1, y2);
  for (int i = 0; i < 100; ++i) y2[i][0] /= D[i];
  backward_sub_unit_diag(U2, y2);
  for (int i = 0; i < 100; ++i)
    EXPECT_NEAR(y1[i], y2[i][0], 1e-10) << i << " failed\n";
}

TEST(INV_UDL, fortran) {
  using ccs_t   = CCS<double, int, true>;
  const auto L  = gen_rand_strict_lower_sparse<ccs_t>(100);
  const auto U  = ccs_t(gen_rand_strict_upper_sparse<ccs_t::other_type>(100));
  using array_t = Array<double>;
  const auto D  = gen_ran_vec<array_t>(100);
  auto       y1 = gen_ran_vec<array_t>(100);
  auto       y2 = create_mat<double>(100, 1);
  for (int i = 0; i < 100; ++i) y2[i][0] = y1[i];
  internal::prec_solve_udl_inv(U, D, L, y1);
  const auto L1 = convert2dense(L), U2 = convert2dense(U);
  forward_sub_unit_diag(L1, y2);
  for (int i = 0; i < 100; ++i) y2[i][0] /= D[i];
  backward_sub_unit_diag(U2, y2);
  for (int i = 0; i < 100; ++i)
    EXPECT_NEAR(y1[i], y2[i][0], 1e-10) << i << " failed\n";
}
