///////////////////////////////////////////////////////////////////////////////
//                This file is part of HILUCSI project                       //
///////////////////////////////////////////////////////////////////////////////

#include "common.hpp"
// line break to avoid sorting
#include "hilucsi/ds/Array.hpp"
#include "hilucsi/utils/math.hpp"

#include <gtest/gtest.h>

using namespace hilucsi;
using array_t = Array<double>;
#define N 10

TEST(NRM, nrm) {
  array_t w(N);
  do {
    array_t v = gen_ran_vec<array_t>(N, 0.0, 20.0);
    normalize(v);
    EXPECT_NEAR(norm2(v), 1.0, 1e-15);
  } while (false);
  const auto v = gen_ran_vec<array_t>(N, 0.0, 20.0);
  normalize2(v, w);
  EXPECT_NEAR(norm2(w), 1.0, 1e-15);
}
