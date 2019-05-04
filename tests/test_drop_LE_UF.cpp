//@HEADER
//----------------------------------------------------------------------------
//                Copyright (C) 2019 The PSMILU AUTHORS
//----------------------------------------------------------------------------
//@HEADER

#include "common.hpp"
// line break to avoid sorting
#include "psmilu_Array.hpp"
#include "psmilu_CompressedStorage.hpp"
#include "psmilu_Schur2.hpp"

#include <gtest/gtest.h>

using namespace psmilu;

TEST(L_E, c) {
  using mat_t        = CRS<double, int>;
  const auto    A    = gen_rand_sparse<mat_t>(100, 100);
  auto          L    = gen_rand_sparse<mat_t>(20, 90);
  const auto    nnz1 = L.nnz();
  Array<double> buf(100);
  Array<int>    ibuf(100);
  drop_L_E(A, 80, 2, L, buf, ibuf);
  ASSERT_LE(L.nnz(), nnz1);
}

TEST(L_E, fortran) {
  using mat_t        = CRS<double, int, true>;
  const auto    A    = gen_rand_sparse<mat_t>(100, 100);
  auto          L    = gen_rand_sparse<mat_t>(20, 90);
  const auto    nnz1 = L.nnz();
  Array<double> buf(100);
  Array<int>    ibuf(100);
  drop_L_E(A, 80, 2, L, buf, ibuf);
  ASSERT_LE(L.nnz(), nnz1);
}

TEST(U_F, c) {
  using mat_t        = CCS<double, int>;
  const auto    A    = gen_rand_sparse<mat_t>(100, 100);
  auto          U    = gen_rand_sparse<mat_t>(90, 20);
  const auto    nnz1 = U.nnz();
  Array<double> buf(100);
  Array<int>    ibuf(100);
  drop_U_F(A, 80, 2, U, buf, ibuf);
  ASSERT_LE(U.nnz(), nnz1);
}

TEST(U_F, fortran) {
  using mat_t        = CCS<double, int, true>;
  const auto    A    = gen_rand_sparse<mat_t>(100, 100);
  auto          U    = gen_rand_sparse<mat_t>(90, 20);
  const auto    nnz1 = U.nnz();
  Array<double> buf(100);
  Array<int>    ibuf(100);
  drop_U_F(A, 80, 2, U, buf, ibuf);
  ASSERT_LE(U.nnz(), nnz1);
}