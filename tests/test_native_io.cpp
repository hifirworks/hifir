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

TEST(IO, c) {
  {
    const int n  = RandIntGen(100, 300)();
    using crs_t  = CRS<double, int>;
    const auto A = gen_rand_sparse<crs_t>(n, n);
    write_native_bin("foo.bin", A, 0u);
    crs_t B;
    auto  m = read_native_bin("foo.bin", B);
    ASSERT_EQ(m, 0u);
    const auto Ad = convert2dense(A), Bd = convert2dense(B);
    COMPARE_MATS(Ad, Bd);
    using crs2_t = CRS<double, long>;
    crs2_t C;
    m             = read_native_bin("foo.bin", C);
    const auto Cd = convert2dense(C);
    COMPARE_MATS(Cd, Ad);
    using crs3_t = CRS<double, int, true>;
    crs3_t D;
    read_native_bin("foo.bin", D);
    const auto Dd = convert2dense(D);
    COMPARE_MATS(Dd, Ad);
    using ccs_t = CCS<double, int>;
    ccs_t E;
    read_native_bin("foo.bin", E);
    const auto Ed = convert2dense(E);
    COMPARE_MATS(Ed, Ad);
    using ccs2_t = CCS<float, int, true>;
    ccs2_t F;
    read_native_bin("foo.bin", F);
    const auto Fd = convert2dense(F);
    COMPARE_MATS_TOL(Fd, Ad, 1e-6);
  }
}