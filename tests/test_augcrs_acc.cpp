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

typedef AugCRS<CRS<double, int>> aug_crs_c_t;

const static RandIntGen  i_rand(0, 100);
const static RandRealGen r_rand(1.0, 2.0);

TEST(AUG_CRS, c) {
  aug_crs_c_t aug_crs;
  int         nrows = i_rand(), ncols = i_rand();
  if (nrows < 5) nrows = 5;
  if (ncols < 5) ncols = 5;
  aug_crs.resize(nrows, ncols);
  ASSERT_EQ(aug_crs.nrows(), nrows);
  ASSERT_EQ(aug_crs.ncols(), ncols);
  auto                          mat1 = create_mat<double>(nrows, ncols);
  std::vector<std::vector<int>> col_inds(nrows);
  std::vector<bool>             tags(ncols, false);
  for (int i = 0; i < nrows; ++i) {
    int nnz = i_rand() % ncols;
    if (!nnz) nnz = (i_rand() % 2) ? 0 : ncols;
    int counts = 0, guard = 0;
    for (;;) {
      if (counts == nnz || guard > 2 * nnz) break;
      const int idx = i_rand() % ncols;
      ++guard;
      if (tags[idx]) continue;
      ++counts;
      tags[idx] = true;
      col_inds[i].push_back(idx);
    }
    std::sort(col_inds[i].begin(), col_inds[i].end());
    for (const auto j : col_inds[i]) tags[j] = false;
  }
  aug_crs.begin_assemble_rows();
  for (int i = 0; i < nrows; ++i) {
    const auto &col_ind = col_inds[i];
    for (const auto col : col_ind) mat1[i][col] = r_rand();
    aug_crs.push_back_row(i, col_ind.cbegin(), col_ind.cend(), mat1[i]);
    ASSERT_EQ(aug_crs.nnz_in_row(i), col_ind.size());
  }
  aug_crs.end_assemble_rows();

  std::vector<double> buf1(nrows), buf2(nrows);

  for (int i = 0; i < std::min(nrows, 20); ++i) {
    const int col = i_rand() % ncols;
    load_dense_col(col, mat1, buf1);       // load dense
    load_aug_crs_col(col, aug_crs, buf2);  // load sparse
    for (int j = 0; j < nrows; ++j)
      ASSERT_EQ(buf1[j], buf2[j]) << "col " << col << " did not match!\n";
  }
}

typedef AugCRS<CRS<double, int, true>> aug_crs_fortran_t;

TEST(AUG_CRS, fortran) {
  aug_crs_fortran_t aug_crs;
  int               nrows = i_rand(), ncols = i_rand();
  if (nrows < 5) nrows = 5;
  if (ncols < 5) ncols = 5;
  aug_crs.resize(nrows, ncols);
  ASSERT_EQ(aug_crs.nrows(), nrows);
  ASSERT_EQ(aug_crs.ncols(), ncols);
  auto                          mat1 = create_mat<double>(nrows, ncols);
  std::vector<std::vector<int>> col_inds(nrows);
  std::vector<bool>             tags(ncols, false);
  for (int i = 0; i < nrows; ++i) {
    int nnz = i_rand() % ncols;
    if (!nnz) nnz = (i_rand() % 2) ? 0 : ncols;
    int counts = 0, guard = 0;
    for (;;) {
      if (counts == nnz || guard > 2 * nnz) break;
      const int idx = i_rand() % ncols;
      ++guard;
      if (tags[idx]) continue;
      ++counts;
      tags[idx] = true;
      col_inds[i].push_back(idx + 1);
    }
    std::sort(col_inds[i].begin(), col_inds[i].end());
    for (const auto j : col_inds[i]) tags[j - 1] = false;
  }
  aug_crs.begin_assemble_rows();
  for (int i = 0; i < nrows; ++i) {
    const auto &col_ind = col_inds[i];
    for (const auto col : col_ind) mat1[i][col - 1] = r_rand();
    aug_crs.push_back_row(i, col_ind.cbegin(), col_ind.cend(), mat1[i]);
    ASSERT_EQ(aug_crs.nnz_in_row(i), col_ind.size());
  }
  aug_crs.end_assemble_rows();

  const auto load_col_aug = [&](std::vector<double> &buf, const int col) {
    std::fill(buf.begin(), buf.end(), 0);  // fill all zeros
    auto id = aug_crs.start_col_id(col);
    for (;;) {
      if (aug_crs.is_nil(id)) break;
      buf.at(aug_crs.row_idx(id)) = aug_crs.vals()[aug_crs.val_pos_idx(id)];
      id                          = aug_crs.next_col_id(id);
    }
  };

  std::vector<double> buf1(nrows), buf2(nrows);

  for (int i = 0; i < std::min(nrows, 20); ++i) {
    const int col = i_rand() % ncols;
    load_dense_col(col, mat1, buf1);  // load dense
    load_col_aug(buf2, col);          // load sparse
    for (int j = 0; j < nrows; ++j)
      ASSERT_EQ(buf1[j], buf2[j]) << "col " << col << " did not match!\n";
  }
}
