// psmilu_Array.hpp psmilu_CompressedStorage.hpp

//@HEADER
//----------------------------------------------------------------------------
//                Copyright (C) 2019 The PSMILU AUTHORS
//----------------------------------------------------------------------------
//@HEADER

#include "common.hpp"
// line break to avoid sorting
#include "psmilu_Array.hpp"
#include "psmilu_CompressedStorage.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <ctime>

#define get_a_rand (std::rand() % 100000) / 100000.0
#define get_a_positive_rand std::fabs(get_a_rand) + 1.0

using namespace psmilu;

TEST(CRS2CCS, test_c) {
  typedef CRS<double, int>  crs_t;
  typedef crs_t::other_type ccs_t;
  typedef ccs_t::size_type  s_t;
  std::srand(std::time(0));
  s_t nrows = std::rand() % 300, ncols = std::rand() % 300;
  if (nrows < 5u) nrows = 5u;
  if (ncols < 5u) ncols = 5u;
  auto                          mat1 = create_mat<double>(nrows, ncols);
  std::vector<std::vector<int>> col_inds(nrows);
  std::vector<bool>             tags(ncols, false);
  for (s_t i = 0u; i < nrows; ++i) {
    int nnz = std::rand() % ncols;
    if (!nnz) nnz = (std::rand() % 2) ? 0 : ncols;
    int counts = 0, guard = 0;
    for (;;) {
      if (counts == nnz || guard > 2 * nnz) break;
      const int idx = std::rand() % ncols;
      ++guard;
      if (tags[idx]) continue;
      ++counts;
      tags[idx] = true;
      col_inds[i].push_back(idx);
    }
    std::sort(col_inds[i].begin(), col_inds[i].end());
    for (const auto j : col_inds[i]) tags[j] = false;
  }
  crs_t crs(nrows, ncols);
  crs.begin_assemble_rows();
  for (s_t i = 0u; i < nrows; ++i) {
    const auto &col_ind = col_inds[i];
    for (const auto col : col_ind) mat1[i][col] = get_a_positive_rand;
    crs.push_back_row(i, col_ind.cbegin(), col_ind.cend(), mat1[i]);
    ASSERT_EQ(crs.nnz_in_row(i), col_ind.size());
  }
  crs.end_assemble_rows();
  ccs_t      ccs(crs);
  const auto mat2 = convert2dense(ccs);
  COMPARE_MATS(mat1, mat2);
}

TEST(CRS2CCS, test_fortran) {
  typedef CRS<double, int, true> crs_t;
  typedef crs_t::other_type      ccs_t;
  typedef ccs_t::size_type       s_t;
  std::srand(std::time(0));
  s_t nrows = std::rand() % 300, ncols = std::rand() % 300;
  if (nrows < 5u) nrows = 5u;
  if (ncols < 5u) ncols = 5u;
  auto                          mat1 = create_mat<double>(nrows, ncols);
  std::vector<std::vector<int>> col_inds(nrows);
  std::vector<bool>             tags(ncols, false);
  for (s_t i = 0u; i < nrows; ++i) {
    int nnz = std::rand() % ncols;
    if (!nnz) nnz = (std::rand() % 2) ? 0 : ncols;
    int counts = 0, guard = 0;
    for (;;) {
      if (counts == nnz || guard > 2 * nnz) break;
      const int idx = std::rand() % ncols;
      ++guard;
      if (tags[idx]) continue;
      ++counts;
      tags[idx] = true;
      col_inds[i].push_back(idx + 1);
    }
    std::sort(col_inds[i].begin(), col_inds[i].end());
    for (const auto j : col_inds[i]) tags[j - 1] = false;
  }
  crs_t crs(nrows, ncols);
  crs.begin_assemble_rows();
  for (s_t i = 0u; i < nrows; ++i) {
    const auto &col_ind = col_inds[i];
    for (const auto col : col_ind) mat1[i][col - 1] = get_a_positive_rand;
    crs.push_back_row(i, col_ind.cbegin(), col_ind.cend(), mat1[i]);
    ASSERT_EQ(crs.nnz_in_row(i), col_ind.size());
  }
  crs.end_assemble_rows();
  ccs_t      ccs(crs);
  const auto mat2 = convert2dense(ccs);
  COMPARE_MATS(mat1, mat2);
}

TEST(CCS2CRS, test_c) {
  typedef CRS<double, int>  crs_t;
  typedef crs_t::other_type ccs_t;
  typedef ccs_t::size_type  s_t;
  std::srand(std::time(0));
  s_t nrows = std::rand() % 300, ncols = std::rand() % 300;
  if (nrows < 5u) nrows = 5u;
  if (ncols < 5u) ncols = 5u;
  auto                          mat1 = create_mat<double>(nrows, ncols);
  std::vector<std::vector<int>> row_inds(ncols);
  std::vector<bool>             tags(nrows, false);
  for (s_t i = 0u; i < ncols; ++i) {
    int nnz = std::rand() % nrows;
    if (!nnz) nnz = (std::rand() % 2) ? 0 : nrows;
    int counts = 0, guard = 0;
    for (;;) {
      if (counts == nnz || guard > 2 * nnz) break;
      const int idx = std::rand() % nrows;
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
  ccs_t               ccs(nrows, ncols);
  ccs.begin_assemble_cols();
  for (s_t i = 0; i < ncols; ++i) {
    const auto &row_ind = row_inds[i];
    for (const auto row : row_ind) mat1[row][i] = get_a_positive_rand;
    for (s_t j = 0u; j < nrows; ++j) buf[j] = mat1[j][i];
    ccs.push_back_col(i, row_ind.cbegin(), row_ind.cend(), buf);
    ASSERT_EQ(ccs.nnz_in_col(i), row_ind.size());
  }
  ccs.end_assemble_cols();
  crs_t      crs(ccs);
  const auto mat2 = convert2dense(crs);
  COMPARE_MATS(mat1, mat2);
}

TEST(CCS2CRS, test_fortran) {
  typedef CRS<double, int, true> crs_t;
  typedef crs_t::other_type      ccs_t;
  typedef ccs_t::size_type       s_t;
  std::srand(std::time(0));
  s_t nrows = std::rand() % 300, ncols = std::rand() % 300;
  if (nrows < 5u) nrows = 5u;
  if (ncols < 5u) ncols = 5u;
  auto                          mat1 = create_mat<double>(nrows, ncols);
  std::vector<std::vector<int>> row_inds(ncols);
  std::vector<bool>             tags(nrows, false);
  for (s_t i = 0u; i < ncols; ++i) {
    int nnz = std::rand() % nrows;
    if (!nnz) nnz = (std::rand() % 2) ? 0 : nrows;
    int counts = 0, guard = 0;
    for (;;) {
      if (counts == nnz || guard > 2 * nnz) break;
      const int idx = std::rand() % nrows;
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
  ccs_t               ccs(nrows, ncols);
  ccs.begin_assemble_cols();
  for (s_t i = 0; i < ncols; ++i) {
    const auto &row_ind = row_inds[i];
    for (const auto row : row_ind) mat1[row - 1][i] = get_a_positive_rand;
    for (s_t j = 0u; j < nrows; ++j) buf[j] = mat1[j][i];
    ccs.push_back_col(i, row_ind.cbegin(), row_ind.cend(), buf);
    ASSERT_EQ(ccs.nnz_in_col(i), row_ind.size());
  }
  ccs.end_assemble_cols();
  crs_t      crs(ccs);
  const auto mat2 = convert2dense(crs);
  COMPARE_MATS(mat1, mat2);
}
