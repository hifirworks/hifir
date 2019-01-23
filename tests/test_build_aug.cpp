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

static const RandIntGen  i_rand(0, 50);
static const RandRealGen r_rand;

TEST(BUILD_AUG_CRS_EXP, crs) {
  typedef CRS<double, int> crs_t;
  typedef crs_t::size_type s_t;
  crs_t                    crs1;
  const s_t                nrows(i_rand() + 1), ncols(i_rand() + 1);
  std::cout << "c-test, (nrows,ncols)=(" << nrows << ',' << ncols << ")\n";
  crs1.resize(nrows, ncols);

  auto                          mat_ref = create_mat<double>(nrows, ncols);
  std::vector<std::vector<int>> nnz_list(nrows);
  std::vector<bool>             col_tags(ncols, false);
  for (s_t i = 0u; i < nrows; ++i) {
    const int nnz    = i_rand() % ncols;
    int       counts = 0;
    for (;;) {
      if (counts == nnz) break;
      const int idx = i_rand() % ncols;
      if (col_tags[idx]) continue;
      ++counts;
      nnz_list[i].push_back(idx);
      col_tags[idx] = true;
    }
    for (const auto j : nnz_list[i]) col_tags[j] = false;
  }
  // construct sparse matrix
  crs1.begin_assemble_rows();
  for (s_t i = 0u; i < nrows; ++i) {
    for (const auto col : nnz_list[i]) mat_ref[i][col] = r_rand();
    crs1.push_back_row(i, nnz_list[i].cbegin(), nnz_list[i].cend(), mat_ref[i]);
    ASSERT_EQ(crs1.nnz_in_row(i), nnz_list[i].size());
  }
  crs1.end_assemble_rows();

  // aug crs type
  using aug_crs_t = AugCRS<crs_t>;
  aug_crs_t aug_crs;
  aug_crs.build_aug(crs1);

  std::vector<double> buf1(nrows), buf2(nrows);

  for (s_t i = 0u; i < std::min(nrows, 20ul); ++i) {
    const int col = i_rand() % ncols;
    load_dense_col(col, mat_ref, buf1);    // load dense
    load_aug_crs_col(col, aug_crs, buf2);  // load sparse
    for (int j = 0; j < (int)nrows; ++j)
      ASSERT_EQ(buf1[j], buf2[j]) << "col " << col << " did not match!\n";
  }
}

TEST(BUILD_AUG_CCS_EXP, ccs) {
  using ccs_t = CCS<double, int>;
  ccs_t     ccs;
  const int nrows(i_rand() + 1), ncols(i_rand() + 1);
  ccs.resize(nrows, ncols);

  auto                          mat_ref = create_mat<double>(nrows, ncols);
  std::vector<std::vector<int>> row_inds(ncols);
  std::vector<bool>             tags(nrows, false);
  for (int i = 0; i < ncols; ++i) {
    int nnz = i_rand() % nrows;
    if (!nnz) nnz = i_rand() % 2 ? 0 : nrows;
    int counts = 0;
    for (;;) {
      if (counts == nnz) break;
      const int idx = i_rand() % nrows;
      if (tags[idx]) continue;
      ++counts;
      row_inds[i].push_back(idx);
      tags[idx] = true;
    }
    for (const auto j : row_inds[i]) tags[j] = false;
  }
  std::vector<double> buf(nrows);
  ccs.begin_assemble_cols();
  for (int i = 0; i < ncols; ++i) {
    for (const auto row : row_inds[i]) mat_ref[row][i] = r_rand();
    for (int j = 0; j < nrows; ++j) buf[j] = mat_ref[j][i];
    ccs.push_back_col(i, row_inds[i].cbegin(), row_inds[i].cend(), buf);
    ASSERT_EQ(ccs.nnz_in_col(i), row_inds[i].size());
  }

  // aug ccs type
  using aug_ccs_t = AugCCS<ccs_t>;
  aug_ccs_t aug_ccs;
  aug_ccs.build_aug(ccs);

  std::vector<double> buf1(ncols), buf2(ncols);

  for (int i = 0; i < std::min(ncols, 20); ++i) {
    const int row = i_rand() % nrows;
    load_dense_row(row, mat_ref, buf1);    // load dense
    load_aug_ccs_row(row, aug_ccs, buf2);  // load sparse
    for (int j = 0; j < ncols; ++j)
      ASSERT_EQ(buf1[j], buf2[j]) << "row " << row << " did not match!\n";
  }
}
