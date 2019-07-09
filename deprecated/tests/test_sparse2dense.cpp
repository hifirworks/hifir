//@HEADER
//----------------------------------------------------------------------------
//                Copyright (C) 2019 The PSMILU AUTHORS
//----------------------------------------------------------------------------
//@HEADER

#include "common.hpp"
// line break to avoid sorting
#include "psmilu_Array.hpp"
#include "psmilu_CompressedStorage.hpp"
#include "psmilu_DenseMatrix.hpp"

#include <gtest/gtest.h>

using namespace psmilu;

const static RandIntGen  i_rand(0, 100);
const static RandRealGen r_rand(1.0, 2.0);

using dense_t = DenseMatrix<double>;

TEST(S2D, c) {
  typedef CRS<double, int> crs_t;
  crs_t                    crs1;
  const int                nrows(i_rand() + 1), ncols(i_rand() + 1);
  std::cout << "c-test, (nrows,ncols)=(" << nrows << ',' << ncols << ")\n";
  crs1.resize(nrows, ncols);

  auto                          mat_ref = create_mat<double>(nrows, ncols);
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
  // construct sparse matrix
  crs1.begin_assemble_rows();
  for (int i = 0; i < nrows; ++i) {
    for (const auto col : col_inds[i]) mat_ref[i][col] = r_rand();
    crs1.push_back_row(i, col_inds[i].cbegin(), col_inds[i].cend(), mat_ref[i]);
    ASSERT_EQ(crs1.nnz_in_row(i), col_inds[i].size());
  }
  crs1.end_assemble_rows();

  auto mat(dense_t::from_sparse(crs1));
  for (int i = 0; i < nrows; ++i)
    for (int j = 0; j < ncols; ++j)
      ASSERT_EQ(mat(i, j), mat_ref[i][j])
          << "(i,j)=(" << i << ',' << j << ", failed!\n";

  typedef CCS<double, int> ccs_t;
  ccs_t                    ccs(nrows, ncols);
  std::vector<int>         inds;
  std::vector<double>      vals(nrows);
  ccs.begin_assemble_cols();
  for (int j = 0; j < ncols; ++j) {
    for (int i = 0; i < nrows; ++i)
      if (mat_ref[i][j] != 0.0) {
        inds.push_back(i);
        vals[i] = mat_ref[i][j];
      }
    ccs.push_back_col(j, inds.cbegin(), inds.cend(), vals);
    ASSERT_EQ(ccs.nnz_in_col(j), inds.size());
    inds.clear();
  }
  ccs.end_assemble_cols();

  mat.copy_sparse(ccs);
  for (int i = 0; i < nrows; ++i)
    for (int j = 0; j < ncols; ++j)
      ASSERT_EQ(mat(i, j), mat_ref[i][j])
          << "(i,j)=(" << i << ',' << j << ", failed!\n";
}

TEST(S2D, fortran) {
  typedef CRS<double, int, true> crs_t;
  crs_t                          crs1;
  const int                      nrows(i_rand() + 1), ncols(i_rand() + 1);
  std::cout << "c-test, (nrows,ncols)=(" << nrows << ',' << ncols << ")\n";
  crs1.resize(nrows, ncols);

  auto                          mat_ref = create_mat<double>(nrows, ncols);
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
  // construct sparse matrix
  crs1.begin_assemble_rows();
  for (int i = 0; i < nrows; ++i) {
    for (const auto col : col_inds[i]) mat_ref[i][col - 1] = r_rand();
    crs1.push_back_row(i, col_inds[i].cbegin(), col_inds[i].cend(), mat_ref[i]);
    ASSERT_EQ(crs1.nnz_in_row(i), col_inds[i].size());
  }
  crs1.end_assemble_rows();

  auto mat(dense_t::from_sparse(crs1));
  for (int i = 0; i < nrows; ++i)
    for (int j = 0; j < ncols; ++j)
      ASSERT_EQ(mat(i, j), mat_ref[i][j])
          << "(i,j)=(" << i << ',' << j << ", failed!\n";

  typedef CCS<double, int, true> ccs_t;
  ccs_t                          ccs(nrows, ncols);
  std::vector<int>               inds;
  std::vector<double>            vals(nrows);
  ccs.begin_assemble_cols();
  for (int j = 0; j < ncols; ++j) {
    for (int i = 0; i < nrows; ++i)
      if (mat_ref[i][j] != 0.0) {
        inds.push_back(i + 1);
        vals[i] = mat_ref[i][j];
      }
    ccs.push_back_col(j, inds.cbegin(), inds.cend(), vals);
    ASSERT_EQ(ccs.nnz_in_col(j), inds.size());
    inds.clear();
  }
  ccs.end_assemble_cols();

  mat.copy_sparse(ccs);
  for (int i = 0; i < nrows; ++i)
    for (int j = 0; j < ncols; ++j)
      ASSERT_EQ(mat(i, j), mat_ref[i][j])
          << "(i,j)=(" << i << ',' << j << ", failed!\n";
}
