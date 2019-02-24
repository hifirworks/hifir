//@HEADER
//----------------------------------------------------------------------------
//                Copyright (C) 2019 The PSMILU AUTHORS
//----------------------------------------------------------------------------
//@HEADER

#include "common.hpp"
// line break to avoid sorting
#include "psmilu_Options.h"

#include <gtest/gtest.h>

using namespace psmilu;

const RandIntGen  i_rand(0, 100);
const RandRealGen r_rand(0.0, 100.0);

#define MUST_SUCCESS ASSERT_FALSE
#define MUST_FAIL ASSERT_TRUE

TEST(OPT, set) {
  Options      opt;
  const double tau_L = r_rand(), tau_U = r_rand(), tau_d = r_rand(),
               tau_kappa = r_rand();
  const int    alpha_L = i_rand(), alpha_U = i_rand();
  const double rho = r_rand(), c_d = r_rand(), c_h = r_rand();
  const int    N = i_rand(), verbose = i_rand();
  MUST_SUCCESS(set_option_attr("tau_L", tau_L, opt));
  MUST_SUCCESS(set_option_attr("tau_U", tau_U, opt));
  MUST_SUCCESS(set_option_attr("tau_d", tau_d, opt));
  MUST_SUCCESS(set_option_attr("tau_kappa", tau_kappa, opt));
  MUST_SUCCESS(set_option_attr("alpha_L", alpha_L, opt));
  MUST_SUCCESS(set_option_attr("alpha_U", alpha_U, opt));
  MUST_SUCCESS(set_option_attr("rho", rho, opt));
  MUST_SUCCESS(set_option_attr("c_d", c_d, opt));
  MUST_SUCCESS(set_option_attr("c_h", c_h, opt));
  MUST_SUCCESS(set_option_attr("N", N, opt));
  MUST_SUCCESS(set_option_attr("verbose", verbose, opt));

  ASSERT_EQ(opt.tau_L, tau_L);
  ASSERT_EQ(opt.tau_U, tau_U);
  ASSERT_EQ(opt.tau_d, tau_d);
  ASSERT_EQ(opt.tau_kappa, tau_kappa);
  ASSERT_EQ(opt.alpha_L, alpha_L);
  ASSERT_EQ(opt.alpha_U, alpha_U);
  ASSERT_EQ(opt.rho, rho);
  ASSERT_EQ(opt.c_d, c_d);
  ASSERT_EQ(opt.c_h, c_h);
  ASSERT_EQ(opt.N, N);
  ASSERT_EQ(opt.verbose, verbose);

  MUST_FAIL(set_option_attr("foobar", 1, opt));
}
