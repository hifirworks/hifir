//@HEADER
//----------------------------------------------------------------------------
//                Copyright (C) 2019 The PSMILU AUTHORS
//----------------------------------------------------------------------------
//@HEADER

#include "common.hpp"
// line break to avoid sorting
#include "psmilu_Array.hpp"
#include "psmilu_CompressedStorage.hpp"

#include "psmilu_AugmentedStorage.hpp"

#include <gtest/gtest.h>

using namespace psmilu;

TEST(crs, c) {
  using mat_t   = CRS<double, int>;
  const auto A  = gen_rand_sparse<mat_t>(100, 100);
  const auto AD = convert2dense(A);
  using aug_t   = AugCRS<mat_t>;
  aug_t B;
  B.resize(100, 200);
  B.begin_assemble_rows();
  std::vector<double> buf(100);
  for (int i = 0; i < 100; ++i) {
    auto v_itr = A.val_cbegin(i);
    for (auto itr = A.col_ind_cbegin(i); itr != A.col_ind_cend(i);
         ++itr, ++v_itr)
      buf[*itr] = *v_itr;
    B.push_back_row(i, A.col_ind_cbegin(i), A.col_ind_cend(i), buf);
  }
  // defer
  for (int i = 100; i < 200; ++i) B.defer_col(i - 100, i);
  B.end_assemble_rows();
  for (auto &v : B.col_ind()) v -= 100;
  B.resize_ncols(100);
  const auto BD = convert2dense(B);
  COMPARE_MATS(AD, BD);
}

TEST(crs, fortran) {
  using mat_t   = CRS<double, int, true>;
  const auto A  = gen_rand_sparse<mat_t>(100, 100);
  const auto AD = convert2dense(A);
  using aug_t   = AugCRS<mat_t>;
  aug_t B;
  B.resize(100, 200);
  B.begin_assemble_rows();
  std::vector<double> buf(100);
  for (int i = 0; i < 100; ++i) {
    auto v_itr = A.val_cbegin(i);
    for (auto itr = A.col_ind_cbegin(i); itr != A.col_ind_cend(i);
         ++itr, ++v_itr)
      buf[*itr - 1] = *v_itr;
    B.push_back_row(i, A.col_ind_cbegin(i), A.col_ind_cend(i), buf);
  }
  // defer
  for (int i = 100; i < 200; ++i) B.defer_col(i - 100, i);
  B.end_assemble_rows();
  for (auto &v : B.col_ind()) v -= 100;
  B.resize_ncols(100);
  const auto BD = convert2dense(B);
  COMPARE_MATS(AD, BD);
}

TEST(ccs, c) {
  using mat_t   = CCS<double, int>;
  const auto A  = gen_rand_sparse<mat_t>(100, 100);
  const auto AD = convert2dense(A);
  using aug_t   = AugCCS<mat_t>;
  aug_t B;
  B.resize(200, 100);
  B.begin_assemble_cols();
  std::vector<double> buf(100);
  for (int i = 0; i < 100; ++i) {
    auto v_itr = A.val_cbegin(i);
    for (auto itr = A.row_ind_cbegin(i); itr != A.row_ind_cend(i);
         ++itr, ++v_itr)
      buf[*itr] = *v_itr;
    B.push_back_col(i, A.row_ind_cbegin(i), A.row_ind_cend(i), buf);
  }
  // defer
  for (int i = 100; i < 200; ++i) B.defer_row(i - 100, i);
  B.end_assemble_cols();
  for (auto &v : B.row_ind()) v -= 100;
  B.resize_nrows(100);
  const auto BD = convert2dense(B);
  COMPARE_MATS(AD, BD);
}

TEST(ccs, fortran) {
  using mat_t   = CCS<double, int, true>;
  const auto A  = gen_rand_sparse<mat_t>(100, 100);
  const auto AD = convert2dense(A);
  using aug_t   = AugCCS<mat_t>;
  aug_t B;
  B.resize(200, 100);
  B.begin_assemble_cols();
  std::vector<double> buf(100);
  for (int i = 0; i < 100; ++i) {
    auto v_itr = A.val_cbegin(i);
    for (auto itr = A.row_ind_cbegin(i); itr != A.row_ind_cend(i);
         ++itr, ++v_itr)
      buf[*itr - 1] = *v_itr;
    B.push_back_col(i, A.row_ind_cbegin(i), A.row_ind_cend(i), buf);
  }
  // defer
  for (int i = 100; i < 200; ++i) B.defer_row(i - 100, i);
  B.end_assemble_cols();
  for (auto &v : B.row_ind()) v -= 100;
  B.resize_nrows(100);
  const auto BD = convert2dense(B);
  COMPARE_MATS(AD, BD);
}