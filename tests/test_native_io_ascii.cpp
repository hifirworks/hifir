//@HEADER
//----------------------------------------------------------------------------
//                Copyright (C) 2019 The PSMILU AUTHORS
//----------------------------------------------------------------------------
//@HEADER

#include "common.hpp"
// line break to avoid sorting
#include "psmilu_Array.hpp"
#include "psmilu_CompressedStorage.hpp"
#include "psmilu_native_io.hpp"

#include <gtest/gtest.h>

using namespace psmilu;

TEST(IO, crs) {
  const int n = RandIntGen(100, 300)();
  {
    using crs_t  = CRS<double, int>;
    const auto A = gen_rand_sparse<crs_t>(n, n);
    A.write_native_ascii("foo1.psmilu");
    crs_t B;
    B.read_native_ascii("foo1.psmilu");
    const auto Ad = convert2dense(A);
    const auto Bd = convert2dense(B);
    COMPARE_MATS(Ad, Bd);
    CRS<double, int, true> C;
    C.read_native_ascii("foo1.psmilu");
    const auto Cd = convert2dense(C);
    COMPARE_MATS(Ad, Cd);
    crs_t::other_type D;
    D.read_native_ascii("foo1.psmilu");
    const auto Dd = convert2dense(D);
    COMPARE_MATS(Ad, Dd);
  }
}

TEST(IO, ccs) {
  const int n = RandIntGen(100, 300)();
  {
    using ccs_t  = CCS<double, int>;
    const auto A = gen_rand_sparse<ccs_t>(n, n);
    A.write_native_ascii("foo2.psmilu");
    ccs_t B;
    B.read_native_ascii("foo2.psmilu");

    const auto Ad = convert2dense(A);
    const auto Bd = convert2dense(B);
    COMPARE_MATS(Ad, Bd);
    CCS<double, int, true> C;
    C.read_native_ascii("foo2.psmilu");
    const auto Cd = convert2dense(C);
    COMPARE_MATS(Ad, Cd);
    ccs_t::other_type D;

    D.read_native_ascii("foo2.psmilu");
    const auto Dd = convert2dense(D);
    COMPARE_MATS(Ad, Dd);
  }
}
