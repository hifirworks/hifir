//@HEADER
//----------------------------------------------------------------------------
//                Copyright (C) 2019 The PSMILU AUTHORS
//----------------------------------------------------------------------------
//@HEADER

#include "common.hpp"
// line break to avoid sorting
#include "psmilu_Array.hpp"
#include "psmilu_CompressedStorage.hpp"
#include "psmilu_matching/HSL_MC64.hpp"

#include <gtest/gtest.h>

using namespace psmilu;

constexpr static int N = 100;

const static RandIntGen  i_rand(0, N);
const static RandRealGen r_rand(1.0, 2.0);

TEST(MC64_EXTRACT, asymm) {
  {
    using ccs_t      = CCS<double, int>;
    const ccs_t A    = gen_rand_sparse<ccs_t>(N, N);
    const int   m    = (i_rand() % N) + 1;
    const auto  mat1 = convert2dense(A);
    const auto  B    = internal::extract_leading_block4matching<false>(A, m);
    const auto  mat2 = convert2dense(B);
    COMPARE_MATS_BLOCK(mat1, mat2, m);
  }
  {
    using ccs_t      = CCS<double, int, true>;
    const ccs_t A    = gen_rand_sparse<ccs_t>(N, N);
    const int   m    = (i_rand() % N) + 1;
    const auto  mat1 = convert2dense(A);
    const auto  B    = internal::extract_leading_block4matching<false>(A, m);
    const auto  mat2 = convert2dense(B);
    COMPARE_MATS_BLOCK(mat1, mat2, m);
  }
}

TEST(MC64_EXTRACT, symm) {
  {
    using ccs_t      = CCS<double, int>;
    const ccs_t A    = gen_rand_sparse<ccs_t>(N, N);
    const int   m    = (i_rand() % N) + 1;
    const auto  mat1 = convert2dense(A);
    const auto  B    = internal::extract_leading_block4matching<true>(A, m);
    // std::cout << "nnz=" << B.nnz() << '\n';
    const auto mat2 = convert2dense(B);
    for (int i = 0; i < m; ++i) {
      int j(0);
      for (; j <= i; ++j)
        EXPECT_EQ(mat1[i][j], mat2[i][j]) << i << ',' << j << " failed\n";
      for (; j < m; ++j) EXPECT_EQ(mat2[i][j], 0);
    }
  }
  {
    using ccs_t      = CCS<double, int, true>;
    const ccs_t A    = gen_rand_sparse<ccs_t>(N, N);
    const int   m    = (i_rand() % N) + 1;
    const auto  mat1 = convert2dense(A);
    const auto  B    = internal::extract_leading_block4matching<true>(A, m);
    // std::cout << "nnz=" << B.nnz() << '\n';
    const auto mat2 = convert2dense(B);
    for (int i = 0; i < m; ++i) {
      int j(0);
      for (; j <= i; ++j)
        EXPECT_EQ(mat1[i][j], mat2[i][j]) << i << ',' << j << " failed\n";
      for (; j < m; ++j) EXPECT_EQ(mat2[i][j], 0);
    }
  }
}
