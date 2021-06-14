///////////////////////////////////////////////////////////////////////////////
//                  This file is part of the HIFIR library                   //
///////////////////////////////////////////////////////////////////////////////

#include "common.hpp"
// line break to avoid sorting
#include "hif/ds/Array.hpp"
#include "hif/ds/DenseMatrix.hpp"

#include <gtest/gtest.h>

using namespace hif;

const static RandIntGen  i_rand(0, 100);
const static RandRealGen r_rand(1.0, 2.0);

TEST(DENSE, core) {
  DenseMatrix<double> mat1;
  ASSERT_EQ(mat1.status(), DATA_UNDEF);
  const int nrows = i_rand() + 1, ncols = i_rand() + 1;
  mat1.resize(nrows, ncols);
  ASSERT_EQ(mat1.nrows(), nrows);
  ASSERT_EQ(mat1.ncols(), ncols);
  std::vector<double> rv_buf(ncols), cv_buf(nrows);
  for (int col = 0; col < ncols; ++col)
    for (int row = 0; row < nrows; ++row) mat1(row, col) = r_rand();

  // test row iterator, do not use this in practice
  const int row_idx = i_rand() % nrows;
  std::copy(mat1.row_begin(row_idx), mat1.row_end(row_idx), rv_buf.begin());
  for (int col = 0; col < ncols; ++col)
    ASSERT_EQ(mat1(row_idx, col), rv_buf[col])
        << "row,col=" << row_idx << ',' << col << " failed\n";

  // test column iterator, preferred way
  const int col_idx = i_rand() % ncols;
  std::copy(mat1.col_begin(col_idx), mat1.col_end(col_idx), cv_buf.begin());
  for (int row = 0; row < nrows; ++row)
    ASSERT_EQ(mat1(row, col_idx), cv_buf[row])
        << "row,col=" << row << ',' << col_idx << " failed\n";

  // test shallow copy
  ASSERT_EQ(DenseMatrix<double>(mat1).data(), mat1.data());
}
