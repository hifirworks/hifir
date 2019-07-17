//@HEADER
//----------------------------------------------------------------------------
//                Copyright (C) 2019 The HILUCSI AUTHORS
//----------------------------------------------------------------------------
//@HEADER

#include "common.hpp"
// line break to avoid sorting
#include "hilucsi/ds/Array.hpp"
#include "hilucsi/ds/CompressedStorage.hpp"
#include "hilucsi/utils/io.hpp"

#include <gtest/gtest.h>

using namespace hilucsi;

TEST(IO, crs) {
  const int n  = RandIntGen(100, 300)();
  using crs_t  = CRS<double, int>;
  const auto A = gen_rand_sparse<crs_t>(n, n);
  A.write_ascii("foo1.hilucsi");
  crs_t B;
  B.read_ascii("foo1.hilucsi");
  const auto Ad = convert2dense(A);
  const auto Bd = convert2dense(B);
  COMPARE_MATS(Ad, Bd);
  crs_t::other_type D;
  D.read_ascii("foo1.hilucsi");
  const auto Dd = convert2dense(D);
  COMPARE_MATS(Ad, Dd);
}

TEST(IO, ccs) {
  const int n  = RandIntGen(100, 300)();
  using ccs_t  = CCS<double, int>;
  const auto A = gen_rand_sparse<ccs_t>(n, n);
  A.write_ascii("foo2.hilucsi");
  ccs_t B;
  B.read_ascii("foo2.hilucsi");

  const auto Ad = convert2dense(A);
  const auto Bd = convert2dense(B);
  COMPARE_MATS(Ad, Bd);
  ccs_t::other_type D;

  D.read_ascii("foo2.hilucsi");
  const auto Dd = convert2dense(D);
  COMPARE_MATS(Ad, Dd);
}
