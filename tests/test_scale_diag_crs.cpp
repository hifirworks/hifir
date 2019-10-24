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

static const RandIntGen  i_rand(0, 50);
static const RandRealGen r_rand;

constexpr static double eps = std::numeric_limits<double>::epsilon();

TEST(CRS_SCALE_DIAG, core) {
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
  Array<double>       s(nrows);
  std::vector<double> t(ncols);
  for (auto &v : s) v = r_rand();
  for (auto &v : t) v = r_rand();
  crs1.scale_diag_left(s);
  scale_dense_left(mat_ref, s);
  auto mat1 = convert2dense(crs1);
  COMPARE_MATS_TOL(mat1, mat_ref, 10. * eps);
  crs1.scale_diag_right(t);
  scale_dense_right(mat_ref, t);
  auto mat2 = convert2dense(crs1);
  COMPARE_MATS_TOL(mat2, mat_ref, 10. * eps);
}