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

#include <cstdlib>
#include <ctime>
#include <vector>

template <class T>
using matrix = std::vector<std::vector<T>>;

template <typename T>
static matrix<T> create_mat(const int nrows, const int ncols) {
  return matrix<T>(nrows, std::vector<T>(ncols, T()));
}

template <class Ccs>
static matrix<typename Ccs::value_type> convert_ccs2dense(const Ccs &ccs) {
  typedef typename Ccs::value_type v_t;
  typedef typename Ccs::size_type  i_t;
  typedef matrix<v_t>              mat_t;
  const auto c_idx = [](const i_t i) -> i_t { return i - Ccs::ONE_BASED; };
  mat_t      mat(create_mat<v_t>(ccs.nrows(), ccs.ncols()));
  for (i_t j = 0u; j < ccs.ncols(); ++j) {
    auto row_itr = ccs.row_ind_cbegin(j);
    for (auto val_itr = ccs.val_cbegin(j), val_end = ccs.val_cend(j);
         val_itr != val_end; ++row_itr, ++val_itr)
      mat.at(c_idx(*row_itr))[j] = *val_itr;
  }
  return mat;
}

#define COMPARE_MATS(mat1, mat2)                                              \
  do {                                                                        \
    ASSERT_EQ(mat1.size(), mat2.size());                                      \
    ASSERT_EQ(mat1.front().size(), mat2.front().size());                      \
    const auto n = mat1.size(), m = mat1.front().size();                      \
    for (decltype(mat1.size()) i = 0u; i < n; ++i)                            \
      for (decltype(i) j = 0u; j < m; ++j) ASSERT_EQ(mat1[i][j], mat2[i][j]); \
  } while (false)

#define get_a_rand (std::rand() % 100000) / 100000.0

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
  std::srand(std::time(0));
  s_t nrows(std::rand() % 100), ncols(std::rand() % 100u);
  if (nrows < 5u) nrows = 5u;
  if (ncols < 5u) ncols = 5u;
  ccs1.resize(nrows, ncols);

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
    for (const auto row : nnz_list[i]) mat_ref[row][i] = get_a_rand;
    for (s_t j = 0u; j < nrows; ++j) buf[j] = mat_ref[j][i];
    ccs1.push_back_col(i, nnz_list[i].cbegin(), nnz_list[i].cend(), buf);
    ASSERT_EQ(ccs1.nnz_in_col(i), nnz_list[i].size());
  }
  ccs1.end_assemble_cols();
  const auto mat2 = convert_ccs2dense(ccs1);
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
  const auto mat3 = convert_ccs2dense(ccs3);
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
  std::srand(std::time(0));
  s_t nrows(std::rand() % 100), ncols(std::rand() % 100u);
  if (nrows < 5u) nrows = 5u;
  if (ncols < 5u) ncols = 5u;
  ccs1.resize(nrows, ncols);

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
    for (const auto row : nnz_list[i]) mat_ref[row - 1][i] = get_a_rand;
    for (s_t j = 0u; j < nrows; ++j) buf[j] = mat_ref[j][i];
    ccs1.push_back_col(i, nnz_list[i].cbegin(), nnz_list[i].cend(), buf);
    ASSERT_EQ(ccs1.nnz_in_col(i), nnz_list[i].size());
  }
  ccs1.end_assemble_cols();
  const auto mat2 = convert_ccs2dense(ccs1);
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
  const auto mat3 = convert_ccs2dense(ccs3);
  COMPARE_MATS(mat2, mat3);
}
