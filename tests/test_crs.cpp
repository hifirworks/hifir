// psmilu_Array.hpp psmilu_CompressedStorage.hpp

//@HEADER
//----------------------------------------------------------------------------
//                Copyright (C) 2019 The PSMILU AUTHORS
//----------------------------------------------------------------------------
//@HEADER

#include "common.hpp"
// line break to avoid sorting
#include "psmilu_CompressedStorage.hpp"

#include <gtest/gtest.h>

#include <cstdlib>
#include <ctime>
#include <vector>

template <class Crs>
static std::vector<std::vector<typename Crs::value_type>> convert_crs2dense(
    const Crs &crs) {
  typedef typename Crs::value_type v_t;
  typedef typename Crs::size_type  i_t;
  const auto c_idx = [](const i_t i) -> i_t { return i - Crs::ONE_BASED; };
  std::vector<std::vector<v_t>> mat(crs.nrows(),
                                    std::vector<v_t>(crs.ncols(), v_t()));
  for (i_t i = 0u; i < crs.nrows(); ++i) {
    auto col_itr = crs.col_ind_cbegin(i);
    for (auto val_itr = crs.val_cbegin(i), val_end = crs.val_cend(i);
         val_itr != val_end; ++col_itr, ++val_itr)
      mat[i].at(c_idx(*col_itr)) = *val_itr;
  }
  return mat;
}

template <class T>
static std::vector<std::vector<T>> create_mat(const int nrows,
                                              const int ncols) {
  return std::vector<std::vector<T>>(nrows, std::vector<T>(ncols, T()));
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

TEST(CRS_api, test_core_c) {
  typedef CRS<double, int> crs_t;
  typedef crs_t::size_type s_t;
  crs_t                    crs1;
  ASSERT_EQ(crs1.nrows(), 0u);
  ASSERT_EQ(crs1.ncols(), 0u);
  crs1.resize(10, 20);
  ASSERT_EQ(crs1.nrows(), 10u);
  ASSERT_EQ(crs1.ncols(), 20u);
  std::srand(std::time(0));
  s_t nrows(std::rand() % 50), ncols(std::rand() % 50);
  if (nrows < 5u) nrows = 5u;
  if (ncols < 5u) ncols = 5u;
  crs1.resize(nrows, ncols);

  auto                          mat_ref = create_mat<double>(nrows, ncols);
  std::vector<std::vector<int>> nnz_list(nrows);
  std::vector<bool>             col_tags(ncols, false);
  for (s_t i = 0u; i < nrows; ++i) {
    const int nnz    = std::rand() % ncols;
    int       counts = 0;
    for (;;) {
      if (counts == nnz) break;
      const int idx = std::rand() % ncols;
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
    for (const auto col : nnz_list[i]) mat_ref[i][col] = get_a_rand;
    crs1.push_back_row(i, nnz_list[i].cbegin(), nnz_list[i].cend(), mat_ref[i]);
    ASSERT_EQ(crs1.nnz_in_row(i), nnz_list[i].size());
  }
  crs1.end_assemble_rows();
  const auto mat2 = convert_crs2dense(crs1);
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
  const auto mat3 = convert_crs2dense(crs3);
  COMPARE_MATS(mat3, mat2);
}

TEST(CRS_api, test_core_fortran) {
  typedef CRS<double, int, true> crs_t;
  typedef crs_t::size_type       s_t;
  crs_t                          crs1;
  ASSERT_EQ(crs1.nrows(), 0u);
  ASSERT_EQ(crs1.ncols(), 0u);
  crs1.resize(10, 20);
  ASSERT_EQ(crs1.nrows(), 10u);
  ASSERT_EQ(crs1.ncols(), 20u);
  std::srand(std::time(0));
  s_t nrows(std::rand() % 50), ncols(std::rand() % 50);
  if (nrows < 5u) nrows = 5u;
  if (ncols < 5u) ncols = 5u;
  crs1.resize(nrows, ncols);

  auto                          mat_ref = create_mat<double>(nrows, ncols);
  std::vector<std::vector<int>> nnz_list(nrows);
  std::vector<bool>             col_tags(ncols, false);
  for (s_t i = 0u; i < nrows; ++i) {
    const int nnz    = std::rand() % ncols;
    int       counts = 0;
    for (;;) {
      if (counts == nnz) break;
      const int idx = std::rand() % ncols;
      if (col_tags[idx]) continue;
      ++counts;
      nnz_list[i].push_back(idx + 1);  // to fortran
      col_tags[idx] = true;
    }
    for (const auto j : nnz_list[i]) col_tags[j - 1] = false;  // back to c
  }
  // construct sparse matrix
  crs1.begin_assemble_rows();
  for (s_t i = 0u; i < nrows; ++i) {
    for (const auto col : nnz_list[i])
      mat_ref[i][col - 1] = get_a_rand;  // to c
    crs1.push_back_row(i, nnz_list[i].cbegin(), nnz_list[i].cend(), mat_ref[i]);
    ASSERT_EQ(crs1.nnz_in_row(i), nnz_list[i].size());
  }
  crs1.end_assemble_rows();
  const auto mat2 = convert_crs2dense(crs1);
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
  const auto mat3 = convert_crs2dense(crs3);
  COMPARE_MATS(mat3, mat2);
}
