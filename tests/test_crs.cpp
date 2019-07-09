//@HEADER
//----------------------------------------------------------------------------
//                Copyright (C) 2019 The HILUCSI AUTHORS
//----------------------------------------------------------------------------
//@HEADER

#include "common.hpp"
// line break to avoid sorting
#include "hilucsi/ds/Array.hpp"
#include "hilucsi/ds/CompressedStorage.hpp"

#include <gtest/gtest.h>

using namespace hilucsi;

static const RandIntGen  i_rand(0, 50);
static const RandRealGen r_rand;

TEST(CRS_api, test_core) {
  typedef CRS<double, int> crs_t;
  typedef crs_t::size_type s_t;
  crs_t                    crs1;
  ASSERT_EQ(crs1.nrows(), 0u);
  ASSERT_EQ(crs1.ncols(), 0u);
  crs1.resize(10, 20);
  ASSERT_EQ(crs1.nrows(), 10u);
  ASSERT_EQ(crs1.ncols(), 20u);
  s_t nrows(i_rand()), ncols(i_rand());
  if (nrows < 5u) nrows = 5u;
  if (ncols < 5u) ncols = 5u;
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
  const auto mat2 = convert2dense(crs1);
  COMPARE_MATS(mat_ref, mat2);
  // test return
  const auto return_crs1 = [&]() -> crs_t { return crs1; };
  auto       crs2        = return_crs1();
  ASSERT_EQ(crs1.col_ind().data(), crs2.col_ind().data());
  ASSERT_EQ(crs2.nrows(), nrows);
  ASSERT_EQ(crs2.ncols(), ncols);
  // test wrap
  std::vector<int> rowptr(crs1.row_start().cbegin(), crs1.row_start().cend()),
      colind(crs1.col_ind().cbegin(), crs1.col_ind().cend());
  std::vector<double> vals(crs1.vals().cbegin(), crs1.vals().cend());
  crs_t crs3(nrows, ncols, rowptr.data(), colind.data(), vals.data(), true);
  ASSERT_EQ(crs3.status(), DATA_WRAP);
  ASSERT_EQ(crs3.vals().data(), vals.data());
  const auto mat3 = convert2dense(crs3);
  COMPARE_MATS(mat3, mat2);
}