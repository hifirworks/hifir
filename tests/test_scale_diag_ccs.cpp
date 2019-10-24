///////////////////////////////////////////////////////////////////////////////
//                This file is part of HILUCSI project                       //
///////////////////////////////////////////////////////////////////////////////

#include "common.hpp"
// line break to avoid sorting
#include "hilucsi/ds/Array.hpp"
#include "hilucsi/ds/CompressedStorage.hpp"

#include <gtest/gtest.h>
#include <numeric>

using namespace hilucsi;

static const RandIntGen  i_rand(0, 100);
static const RandRealGen r_rand;

constexpr static double eps = std::numeric_limits<double>::epsilon();

TEST(CCS_SCALE_DIAG, core) {
  typedef CCS<double, int> ccs_t;
  typedef ccs_t::size_type s_t;
  ccs_t                    ccs1;
  s_t                      nrows(i_rand() + 1), ncols(i_rand() + 1);
  ccs1.resize(nrows, ncols);
  std::cout << "c-test, (nrows,ncols)=(" << nrows << ',' << ncols << ")\n";

  auto                          mat_ref = create_mat<double>(nrows, ncols);
  std::vector<std::vector<int>> nnz_list(ncols);
  std::vector<bool>             row_tags(nrows, false);
  for (s_t i = 0u; i < ncols; ++i) {
    const int nnz    = i_rand() % nrows;
    int       counts = 0;
    for (;;) {
      if (counts == nnz) break;
      const int idx = i_rand() % nrows;
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
  Array<double>       s(nrows);
  std::vector<double> t(ncols);
  for (auto &v : s) v = r_rand();
  for (auto &v : t) v = r_rand();
  ccs1.scale_diag_left(s);
  scale_dense_left(mat_ref, s);
  auto mat1 = convert2dense(ccs1);
  COMPARE_MATS_TOL(mat1, mat_ref, 10. * eps);
  ccs1.scale_diag_right(t);
  scale_dense_right(mat_ref, t);
  auto mat2 = convert2dense(ccs1);
  COMPARE_MATS_TOL(mat2, mat_ref, 10. * eps);
}