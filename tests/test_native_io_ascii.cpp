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
    const int n  = RandIntGen(100, 300)();
  {
    using crs_t  = CRS<double, int>;
    const auto A = gen_rand_sparse<crs_t>(n, n);
    A.write_native_ascii("foo.psmilu");
  }
}