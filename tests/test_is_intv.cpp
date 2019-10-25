///////////////////////////////////////////////////////////////////////////////
//                This file is part of HILUCSI project                       //
///////////////////////////////////////////////////////////////////////////////

#include "common.hpp"
// line break to avoid sorting
#include "hilucsi/ds/IntervalCompressedStorage.hpp"

#include <gtest/gtest.h>

using namespace hilucsi;

// NOTE: This test is actually for compilation test

TEST(IS_INTV, test) {
  using mat1_t = CRS<double, int>;
  using mat2_t = IntervalCCS<double, int>;
  EXPECT_TRUE(IsIntervalCS<mat2_t>::value);
  EXPECT_FALSE(IsIntervalCS<mat1_t>::value);
}
