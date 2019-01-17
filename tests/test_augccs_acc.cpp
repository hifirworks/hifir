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

typedef AugCCS<CCS<double, int>> aug_ccs_c_t;

const static RandIntGen  i_rand(0, 200);
const static RandRealGen r_rand(1.0, 2.0);

TEST(AUG_CCS, c) {
  aug_ccs_c_t aug_ccs;
  int         nrows = i_rand(), ncols = i_rand();
  if (nrows < 5) nrows = 5;
  if (ncols < 5) ncols = 5;
  aug_ccs.resize(nrows, ncols);
  ASSERT_EQ(aug_ccs.nrows(), nrows);
  ASSERT_EQ(aug_ccs.ncols(), ncols);
  auto                          mat1 = create_mat<double>(nrows, ncols);
  std::vector<std::vector<int>> row_inds(ncols);
  std::vector<bool>             tags(nrows, false);
  for (int i = 0; i < ncols; ++i) {
    int nnz = i_rand() % nrows;
    if (!nnz) nnz = (i_rand() % 2) ? 0 : nrows;
    int counts = 0, guard = 0;
    for (;;) {
      if (counts == nnz || guard > 2 * nnz) break;
      const int idx = i_rand() % nrows;
      ++guard;
      if (tags[idx]) continue;
      ++counts;
      tags[idx] = true;
      row_inds[i].push_back(idx);
    }
    std::sort(row_inds[i].begin(), row_inds[i].end());
    for (const auto j : row_inds[i]) tags[j] = false;
  }
  std::vector<double> buf(nrows);
  aug_ccs.begin_assemble_cols();
  for (int i = 0; i < ncols; ++i) {
    const auto &row_ind = row_inds[i];
    for (const auto row : row_ind) mat1[row][i] = r_rand();
    for (int j = 0u; j < nrows; ++j) buf[j] = mat1[j][i];
    aug_ccs.push_back_col(i, row_ind.cbegin(), row_ind.cend(), buf);
    ASSERT_EQ(aug_ccs.nnz_in_col(i), row_ind.size());
  }
  aug_ccs.end_assemble_cols();

  std::vector<double> buf1(ncols), buf2(ncols);

  for (int i = 0; i < std::min(ncols, 20); ++i) {
    const int row = i_rand() % nrows;
    load_dense_row(row, mat1, buf1);       // load dense
    load_aug_ccs_row(row, aug_ccs, buf2);  // load sparse
    for (int j = 0; j < ncols; ++j)
      ASSERT_EQ(buf1[j], buf2[j]) << "row " << row << " did not match!\n";
  }
}

typedef AugCCS<CCS<double, int, true>> aug_ccs_fortran_t;

TEST(AUG_CCS, fortran) {
  aug_ccs_fortran_t aug_ccs;
  int               nrows = i_rand(), ncols = i_rand();
  if (nrows < 5) nrows = 5;
  if (ncols < 5) ncols = 5;
  aug_ccs.resize(nrows, ncols);
  ASSERT_EQ(aug_ccs.nrows(), nrows);
  ASSERT_EQ(aug_ccs.ncols(), ncols);
  auto                          mat1 = create_mat<double>(nrows, ncols);
  std::vector<std::vector<int>> row_inds(ncols);
  std::vector<bool>             tags(nrows, false);
  for (int i = 0; i < ncols; ++i) {
    int nnz = i_rand() % nrows;
    if (!nnz) nnz = (i_rand() % 2) ? 0 : nrows;
    int counts = 0, guard = 0;
    for (;;) {
      if (counts == nnz || guard > 2 * nnz) break;
      const int idx = i_rand() % nrows;
      ++guard;
      if (tags[idx]) continue;
      ++counts;
      tags[idx] = true;
      row_inds[i].push_back(idx + 1);
    }
    std::sort(row_inds[i].begin(), row_inds[i].end());
    for (const auto j : row_inds[i]) tags[j - 1] = false;
  }
  std::vector<double> buf(nrows);
  aug_ccs.begin_assemble_cols();
  for (int i = 0; i < ncols; ++i) {
    const auto &row_ind = row_inds[i];
    for (const auto row : row_ind) mat1[row - 1][i] = r_rand();
    for (int j = 0u; j < nrows; ++j) buf[j] = mat1[j][i];
    aug_ccs.push_back_col(i, row_ind.cbegin(), row_ind.cend(), buf);
    ASSERT_EQ(aug_ccs.nnz_in_col(i), row_ind.size());
  }
  aug_ccs.end_assemble_cols();

  std::vector<double> buf1(ncols), buf2(ncols);

  for (int i = 0; i < std::min(ncols, 20); ++i) {
    const int row = i_rand() % nrows;
    load_dense_row(row, mat1, buf1);       // load dense
    load_aug_ccs_row(row, aug_ccs, buf2);  // load sparse
    for (int j = 0; j < ncols; ++j)
      ASSERT_EQ(buf1[j], buf2[j]) << "row " << row << " did not match!\n";
  }
}
