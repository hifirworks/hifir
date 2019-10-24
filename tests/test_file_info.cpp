///////////////////////////////////////////////////////////////////////////////
//                This file is part of HILUCSI project                       //
///////////////////////////////////////////////////////////////////////////////

#include "common.hpp"
// line break to avoid sorting
#include "hilucsi/ds/Array.hpp"
#include "hilucsi/ds/CompressedStorage.hpp"
#include "hilucsi/utils/io.hpp"

#include <gtest/gtest.h>

using namespace hilucsi;

TEST(IO, bin) {
  const int n  = RandIntGen(100, 300)();
  using crs_t  = CRS<double, int>;
  const auto A = gen_rand_sparse<crs_t>(n, n);
  write_bin("foo.bin", A, 0u);
  bool          is_row, is_c, is_double, is_real;
  std::uint64_t nrows, ncols, nnz, m;
  std::tie(is_row, is_c, is_double, is_real, nrows, ncols, nnz, m) =
      query_info_bin("foo.bin");
  EXPECT_TRUE(is_row);
  EXPECT_TRUE(is_c);
  EXPECT_TRUE(is_double);
  EXPECT_TRUE(is_real);
  EXPECT_EQ(nrows, n);
  EXPECT_EQ(ncols, n);
  EXPECT_EQ(nnz, A.nnz());
  EXPECT_EQ(m, 0u);
}

TEST(IO, ascii) {
  const int n  = RandIntGen(100, 300)();
  using crs_t  = CRS<float, int>;
  const auto A = gen_rand_sparse<crs_t>(n, n);
  A.write_ascii("foo1.hilucsi");
  bool          is_row, is_c, is_double, is_real;
  std::uint64_t nrows, ncols, nnz, m;
  std::tie(is_row, is_c, is_double, is_real, nrows, ncols, nnz, m) =
      query_info_ascii("foo1.hilucsi");
  EXPECT_TRUE(is_row);
  EXPECT_TRUE(is_c);
  EXPECT_FALSE(is_double);
  EXPECT_TRUE(is_real);
  EXPECT_EQ(nrows, n);
  EXPECT_EQ(ncols, n);
  EXPECT_EQ(nnz, A.nnz());
  EXPECT_EQ(m, 0u);
}