///////////////////////////////////////////////////////////////////////////////
//                  This file is part of the HIFIR library                   //
///////////////////////////////////////////////////////////////////////////////

#include "common.hpp"
// line break to avoid sorting
#include "hif/ds/Array.hpp"
#include "hif/ds/CompressedStorage.hpp"

#include <gtest/gtest.h>

using namespace hif;

static const RandIntGen i_rand(1, 100);

TEST(crs, core) {
  using mat_t   = CRS<double, int>;
  const auto A  = gen_rand_sparse<mat_t>(100, 100);
  const auto AD = convert2dense(A);
  mat_t      F, S;
  const int  m   = i_rand();
  std::tie(F, S) = A.split(m);
  const auto FD  = convert2dense(F);
  const auto SD  = convert2dense(S);
  for (int i = 0; i < 100; ++i) {
    int j = 0;
    for (; j < m; ++j) EXPECT_EQ(AD[i][j], FD[i][j]);
    for (; j < 100; ++j) EXPECT_EQ(AD[i][j], SD[i][j - m]);
  }
}
TEST(ccs, core) {
  using mat_t   = CCS<double, int>;
  const auto A  = gen_rand_sparse<mat_t>(100, 100);
  const auto AD = convert2dense(A);
  mat_t      F, S;
  const int  m   = i_rand();
  std::tie(F, S) = A.split(m);
  const auto FD  = convert2dense(F);
  const auto SD  = convert2dense(S);
  for (int j = 0; j < 100; ++j) {
    int i = 0;
    for (; i < m; ++i) EXPECT_EQ(AD[i][j], FD[i][j]);
    for (; i < 100; ++i) EXPECT_EQ(AD[i][j], SD[i - m][j]);
  }
}