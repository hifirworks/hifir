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

using ccs_t = CCS<double, int>;

TEST(MC64_EXTRACT, asymm) {
  ccs_t A(N, N);

  // construct matrix
  {
    std::vector<std::vector<int>> row_inds(N);
    std::vector<bool>             tags(N, false);
    for (int i = 0u; i < N; ++i) {
      const int nnz    = std::max(5, i_rand() % N);
      int       counts = 0, guard = 0;
      auto&     row_ind = row_inds[i];
      for (;;) {
        if (counts == nnz || guard > 2 * N) break;
        const int idx = i_rand() % N;
        ++guard;
        if (tags[idx]) continue;
        ++counts;
        row_ind.push_back(idx);
        tags[idx] = true;
      }
      if (std::find(row_ind.cbegin(), row_ind.cend(), i) == row_ind.cend())
        row_ind.push_back(i);
      std::sort(row_ind.begin(), row_ind.end());
      for (const auto j : row_ind) tags[j] = false;
    }
    std::vector<double> buf(N);
    // construct sparse matrix
    A.begin_assemble_cols();
    for (int i = 0; i < N; ++i) {
      const auto& row_ind = row_inds[i];
      for (const auto col : row_ind) buf[col] = r_rand();
      A.push_back_col(i, row_ind.cbegin(), row_ind.cend(), buf);
      ASSERT_EQ(A.nnz_in_col(i), row_ind.size());
    }
    A.end_assemble_cols();
  }
  const int  m    = (i_rand() % N) + 1;
  const auto mat1 = convert2dense(A);
  const auto B    = internal::extract_leading_block4matching<false>(A, m);
  const auto mat2 = convert2dense(B);
  COMPARE_MATS_BLOCK(mat1, mat2, m);
}

TEST(MC64_EXTRACT, symm) {
  ccs_t A(N, N);

  // construct matrix
  {
    std::vector<std::vector<int>> row_inds(N);
    std::vector<bool>             tags(N, false);
    for (int i = 0u; i < N; ++i) {
      const int nnz    = std::max(5, i_rand() % N);
      int       counts = 0, guard = 0;
      auto&     row_ind = row_inds[i];
      for (;;) {
        if (counts == nnz || guard > 2 * N) break;
        const int idx = i_rand() % N;
        ++guard;
        if (tags[idx]) continue;
        ++counts;
        row_ind.push_back(idx);
        tags[idx] = true;
      }
      if (std::find(row_ind.cbegin(), row_ind.cend(), i) == row_ind.cend())
        row_ind.push_back(i);
      std::sort(row_ind.begin(), row_ind.end());
      for (const auto j : row_ind) tags[j] = false;
    }
    std::vector<double> buf(N);
    // construct sparse matrix
    A.begin_assemble_cols();
    for (int i = 0; i < N; ++i) {
      const auto& row_ind = row_inds[i];
      for (const auto col : row_ind) buf[col] = r_rand();
      A.push_back_col(i, row_ind.cbegin(), row_ind.cend(), buf);
      ASSERT_EQ(A.nnz_in_col(i), row_ind.size());
    }
    A.end_assemble_cols();
  }
  const int  m    = (i_rand() % N) + 1;
  const auto mat1 = convert2dense(A);
  const auto B    = internal::extract_leading_block4matching<true>(A, m);
  // std::cout << "nnz=" << B.nnz() << '\n';
  const auto mat2 = convert2dense(B);
  for (int i = 0; i < m; ++i) {
    int j(0);
    for (; j <= i; ++j)
      EXPECT_EQ(mat1[i][j], mat2[i][j]) << i << ',' << j << " failed\n";
    for (; j < m; ++j) EXPECT_EQ(mat2[i][j], 0);
  }

  //   for (int i = 0; i < std::min(5, m); ++i) {
  //     for (int j = 0; j < std::min(5, m); ++j) std::cout << mat2[i][j] << '
  //     '; std::cout << std::endl;
}
