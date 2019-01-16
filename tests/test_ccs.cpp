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

static const RandIntGen  i_rand(0, 100);
static const RandRealGen r_rand;

using namespace psmilu;

TEST(CCS_api, test_core_c) {
  typedef CCS<double, int> ccs_t;
  typedef ccs_t::size_type s_t;
  // typedef matrix<double> mat_t;
  ccs_t ccs1;
  ASSERT_EQ(ccs1.nrows(), 0u);
  ASSERT_EQ(ccs1.ncols(), 0u);
  ccs1.resize(30, 20);
  ASSERT_EQ(ccs1.nrows(), 30u);
  ASSERT_EQ(ccs1.ncols(), 20u);
  s_t nrows(i_rand()), ncols(i_rand());
  if (nrows < 5u) nrows = 5u;
  if (ncols < 5u) ncols = 5u;
  ccs1.resize(nrows, ncols);
  std::cout << "c-test, (nrows,ncols)=(" << nrows << ',' << ncols << ")\n";

  auto                          mat_ref = create_mat<double>(nrows, ncols);
  std::vector<std::vector<int>> nnz_list(ncols);
  std::vector<bool>             row_tags(nrows, false);
  for (s_t i = 0u; i < ncols; ++i) {
    const int nnz    = std::rand() % nrows;
    int       counts = 0;
    for (;;) {
      if (counts == nnz) break;
      const int idx = std::rand() % nrows;
      if (row_tags[idx]) continue;
      ++counts;
      nnz_list[i].push_back(idx);
      row_tags[idx] = true;
    }
    for (const auto j : nnz_list[i]) row_tags[j] = false;
  }
  // construct sparse matrix
  std::vector<double> buf(nrows);
  ccs1.begin_assemble_cols();
  for (s_t i = 0u; i < ncols; ++i) {
    for (const auto row : nnz_list[i]) mat_ref[row][i] = r_rand();
    for (s_t j = 0u; j < nrows; ++j) buf[j] = mat_ref[j][i];
    ccs1.push_back_col(i, nnz_list[i].cbegin(), nnz_list[i].cend(), buf);
    ASSERT_EQ(ccs1.nnz_in_col(i), nnz_list[i].size());
  }
  ccs1.end_assemble_cols();
  const auto mat2 = convert2dense(ccs1);
  COMPARE_MATS(mat_ref, mat2);
  const auto return_ccs1 = [&]() -> ccs_t { return ccs1; };
  auto       ccs2        = return_ccs1();
  ASSERT_EQ(ccs1.row_ind().data(), ccs2.row_ind().data());
  ASSERT_EQ(ccs2.nrows(), ccs1.nrows());
  ASSERT_EQ(ccs2.ncols(), ccs1.ncols());
  // test wrap
  std::vector<int> colptr(ccs1.col_start().cbegin(), ccs1.col_start().cend()),
      rowind(ccs1.row_ind().cbegin(), ccs1.row_ind().cend());
  std::vector<double> vals(ccs1.vals().cbegin(), ccs1.vals().cend());
  ccs_t ccs3(nrows, ncols, colptr.data(), rowind.data(), vals.data(), true);
  ASSERT_EQ(ccs3.status(), DATA_WRAP);
  const auto mat3 = convert2dense(ccs3);
  COMPARE_MATS(mat2, mat3);
}

TEST(CCS_api, test_core_fortran) {
  typedef CCS<double, int, true> ccs_t;
  typedef ccs_t::size_type       s_t;
  // typedef matrix<double> mat_t;
  ccs_t ccs1;
  ASSERT_EQ(ccs1.nrows(), 0u);
  ASSERT_EQ(ccs1.ncols(), 0u);
  ccs1.resize(30, 20);
  ASSERT_EQ(ccs1.nrows(), 30u);
  ASSERT_EQ(ccs1.ncols(), 20u);
  s_t nrows(i_rand()), ncols(i_rand());
  if (nrows < 5u) nrows = 5u;
  if (ncols < 5u) ncols = 5u;
  ccs1.resize(nrows, ncols);
  std::cout << "c-test, (nrows,ncols)=(" << nrows << ',' << ncols << ")\n";

  auto                          mat_ref = create_mat<double>(nrows, ncols);
  std::vector<std::vector<int>> nnz_list(ncols);
  std::vector<bool>             row_tags(nrows, false);
  for (s_t i = 0u; i < ncols; ++i) {
    const int nnz    = std::rand() % nrows;
    int       counts = 0;
    for (;;) {
      if (counts == nnz) break;
      const int idx = std::rand() % nrows;
      if (row_tags[idx]) continue;
      ++counts;
      nnz_list[i].push_back(idx + 1);
      row_tags[idx] = true;
    }
    for (const auto j : nnz_list[i]) row_tags[j - 1] = false;
  }
  // construct sparse matrix
  std::vector<double> buf(nrows);
  ccs1.begin_assemble_cols();
  for (s_t i = 0u; i < ncols; ++i) {
    for (const auto row : nnz_list[i]) mat_ref[row - 1][i] = r_rand();
    for (s_t j = 0u; j < nrows; ++j) buf[j] = mat_ref[j][i];
    ccs1.push_back_col(i, nnz_list[i].cbegin(), nnz_list[i].cend(), buf);
    ASSERT_EQ(ccs1.nnz_in_col(i), nnz_list[i].size());
  }
  ccs1.end_assemble_cols();
  const auto mat2 = convert2dense(ccs1);
  COMPARE_MATS(mat_ref, mat2);
  const auto return_ccs1 = [&]() -> ccs_t { return ccs1; };
  auto       ccs2        = return_ccs1();
  ASSERT_EQ(ccs1.row_ind().data(), ccs2.row_ind().data());
  ASSERT_EQ(ccs2.nrows(), ccs1.nrows());
  ASSERT_EQ(ccs2.ncols(), ccs1.ncols());
  // test wrap
  std::vector<int> colptr(ccs1.col_start().cbegin(), ccs1.col_start().cend()),
      rowind(ccs1.row_ind().cbegin(), ccs1.row_ind().cend());
  std::vector<double> vals(ccs1.vals().cbegin(), ccs1.vals().cend());
  ccs_t ccs3(nrows, ncols, colptr.data(), rowind.data(), vals.data(), true);
  ASSERT_EQ(ccs3.status(), DATA_WRAP);
  const auto mat3 = convert2dense(ccs3);
  COMPARE_MATS(mat2, mat3);
}
