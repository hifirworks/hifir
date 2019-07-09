//@HEADER
//----------------------------------------------------------------------------
//                Copyright (C) 2019 The PSMILU AUTHORS
//----------------------------------------------------------------------------
//@HEADER

#include "common.hpp"
// line break to avoid sorting
#include "psmilu_Array.hpp"
#include "psmilu_PermMatrix.hpp"

#include <gtest/gtest.h>

using namespace psmilu;

const static RandIntGen i_rand(0, 1000);

TEST(PERM_MAT, core) {
  using perm_t = PermMatrix<int>;
  const int n  = i_rand() + 1;
  perm_t    p(n);
  ASSERT_EQ(p.size(), n);
  p.make_eye();
  using bi_perm_t = BiPermMatrix<int>;
  bi_perm_t bp(p);           // shallow p and allocate the inverse storage
  ASSERT_EQ(&bp[0], &p[0]);  // should equal
  bp.build_inv();
  for (int i = 0; i < n; ++i) ASSERT_EQ(bp.inv(i), i);
  // test swap
  for (int _ = 0; _ < std::min(20, n); ++_) {
    const int i = i_rand() % n, j = i_rand() % n;
    bp.interchange(i, j);
  }
  for (int i = 0; i < n; ++i) ASSERT_EQ(bp[bp.inv(i)], i);
}
