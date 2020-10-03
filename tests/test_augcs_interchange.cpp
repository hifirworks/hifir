///////////////////////////////////////////////////////////////////////////////
//                This file is part of HILUCSI project                       //
///////////////////////////////////////////////////////////////////////////////

#include "common.hpp"
// line break to avoid sorting
#include "hilucsi/ds/Array.hpp"
#include "hilucsi/ds/AugmentedStorage.hpp"
#include "hilucsi/ds/CompressedStorage.hpp"

#include <gtest/gtest.h>

using namespace hilucsi;

typedef AugCCS<CCS<double, int>> aug_ccs_t;
typedef AugCRS<CRS<double, int>> aug_crs_t;

const static RandIntGen  i_rand(0, 200);
const static RandRealGen r_rand(1.0, 2.0);

#define GET_RANDOM_K_M            \
  const int k = i_rand() % nrows; \
  int       m;                    \
  do {                            \
    m = i_rand() % nrows;         \
  } while (m == k)

TEST(AUG_CS_SWAP, ccs) {
  aug_ccs_t aug_ccs;
  int       nrows = i_rand(), ncols = i_rand();
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
  // test incremental swapping
  std::vector<double> buf(nrows);
  std::vector<double> buf1(ncols), buf2(ncols);
  aug_ccs.begin_assemble_cols();
  for (int i = 0; i < ncols; ++i) {
    const auto &row_ind = row_inds[i];
    for (const auto row : row_ind) mat1[row][i] = r_rand();
    for (int j = 0u; j < nrows; ++j) buf[j] = mat1[j][i];
    aug_ccs.push_back_col(i, row_ind.cbegin(), row_ind.cend(), buf);
    ASSERT_EQ(aug_ccs.nnz_in_col(i), row_ind.size());
    GET_RANDOM_K_M;
    std::cout << "interchange rows " << k << " and " << m << '\n';
    aug_ccs.interchange_rows(k, m);
    interchange_dense_rows(mat1, k, m);
    load_dense_row(k, mat1, buf1);       // load dense
    load_aug_ccs_row(k, aug_ccs, buf2);  // load sparse
    for (int j = 0; j < i; ++j)
      ASSERT_EQ(buf1[j], buf2[j]) << "row " << k << " did not match!\n";
    load_dense_row(m, mat1, buf1);       // load dense
    load_aug_ccs_row(m, aug_ccs, buf2);  // load sparse
    for (int j = 0; j < i; ++j)
      ASSERT_EQ(buf1[j], buf2[j]) << "row " << m << " did not match!\n";
  }
  aug_ccs.end_assemble_cols();

  // test static swapping
  for (int i = 0; i < std::min(ncols, 20); ++i) {
    GET_RANDOM_K_M;
    std::cout << "interchange rows " << k << " and " << m << '\n';
    aug_ccs.interchange_rows(k, m);
    interchange_dense_rows(mat1, k, m);
    load_dense_row(k, mat1, buf1);       // load dense
    load_aug_ccs_row(k, aug_ccs, buf2);  // load sparse
    for (int j = 0; j < ncols; ++j)
      ASSERT_EQ(buf1[j], buf2[j]) << "row " << k << " did not match!\n";
    load_dense_row(m, mat1, buf1);       // load dense
    load_aug_ccs_row(m, aug_ccs, buf2);  // load sparse
    for (int j = 0; j < ncols; ++j)
      ASSERT_EQ(buf1[j], buf2[j]) << "row " << m << " did not match!\n";
  }
}

TEST(AUG_CS_SWAP, crs) {
  aug_crs_t aug_crs;
  int       nrows = i_rand(), ncols = i_rand();
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
  // test incremental column swapping
  int nnz = 0;
  for (const auto &l : col_inds) nnz += l.size();
  aug_crs.reserve(nnz);
  std::vector<double> buf1(nrows), buf2(nrows);
  aug_crs.begin_assemble_rows();
  for (int i = 0; i < nrows; ++i) {
    const auto &col_ind = col_inds[i];
    for (const auto col : col_ind) mat1[i][col] = r_rand();
    aug_crs.push_back_row(i, col_ind.cbegin(), col_ind.cend(), mat1[i]);
    ASSERT_EQ(aug_crs.nnz_in_row(i), col_ind.size());
    // swap here
    GET_RANDOM_K_M;
    std::cout << "interchange columns " << k << " and " << m << '\n';
    aug_crs.interchange_cols(k, m);
    interchange_dense_cols(mat1, k, m);
    load_dense_col(k, mat1, buf1);       // load dense
    load_aug_crs_col(k, aug_crs, buf2);  // load sparse
    for (int j = 0; j < i; ++j)
      ASSERT_EQ(buf1[j], buf2[j]) << "col " << k << " did not match!\n";
    load_dense_col(m, mat1, buf1);       // load dense
    load_aug_crs_col(m, aug_crs, buf2);  // load sparse
    for (int j = 0; j < i; ++j)
      ASSERT_EQ(buf1[j], buf2[j]) << "col " << m << " did not match!\n";
  }
  aug_crs.end_assemble_rows();
  // test static interchanges
  for (int i = 0; i < std::min(nrows, 20); ++i) {
    GET_RANDOM_K_M;
    std::cout << "interchange columns " << k << " and " << m << '\n';
    aug_crs.interchange_cols(k, m);
    interchange_dense_cols(mat1, k, m);
    load_dense_col(k, mat1, buf1);       // load dense
    load_aug_crs_col(k, aug_crs, buf2);  // load sparse
    for (int j = 0; j < nrows; ++j)
      ASSERT_EQ(buf1[j], buf2[j]) << "col " << k << " did not match!\n";
    load_dense_col(m, mat1, buf1);       // load dense
    load_aug_crs_col(m, aug_crs, buf2);  // load sparse
    for (int j = 0; j < nrows; ++j)
      ASSERT_EQ(buf1[j], buf2[j]) << "col " << m << " did not match!\n";
  }
}
