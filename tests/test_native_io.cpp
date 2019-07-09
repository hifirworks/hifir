//@HEADER
//----------------------------------------------------------------------------
//                Copyright (C) 2019 The PSMILU AUTHORS
//----------------------------------------------------------------------------
//@HEADER

#include "common.hpp"
// line break to avoid sorting
#include "hilucsi/ds/Array.hpp"
#include "hilucsi/ds/CompressedStorage.hpp"
#include "hilucsi/utils/io.hpp"

#include <gtest/gtest.h>

using namespace hilucsi;

TEST(IO, c) {
  {
    const int n  = RandIntGen(100, 300)();
    using crs_t  = CRS<double, int>;
    const auto A = gen_rand_sparse<crs_t>(n, n);
    write_bin("foo.bin", A, 0u);
    crs_t B;
    auto  m = read_bin("foo.bin", B);
    ASSERT_EQ(m, 0u);
    const auto Ad = convert2dense(A), Bd = convert2dense(B);
    COMPARE_MATS(Ad, Bd);
    using crs2_t = CRS<double, long>;
    crs2_t C;
    m             = read_bin("foo.bin", C);
    const auto Cd = convert2dense(C);
    COMPARE_MATS(Cd, Ad);
    using ccs_t = CCS<double, int>;
    ccs_t E;
    read_bin("foo.bin", E);
    const auto Ed = convert2dense(E);
    COMPARE_MATS(Ed, Ad);
    using ccs2_t = CCS<float, int>;
    ccs2_t F;
    read_bin("foo.bin", F);
    const auto Fd = convert2dense(F);
    COMPARE_MATS_TOL(Fd, Ad, 1e-6);
  }
}
