///////////////////////////////////////////////////////////////////////////////
//                  This file is part of HIF project                         //
///////////////////////////////////////////////////////////////////////////////

#include "common.hpp"
// line break to avoid sorting
#include "hif/ds/IntervalCompressedStorage.hpp"

#include <gtest/gtest.h>

using namespace hif;

// NOTE: This test is actually for compilation test

TEST(IS_INTV, test) {
  using mat1_t = CRS<double, int>;
  using mat2_t = IntervalCCS<double, int>;
  EXPECT_TRUE(is_interval_cs<mat2_t>::value);
  EXPECT_FALSE(is_interval_cs<mat1_t>::value);
}
