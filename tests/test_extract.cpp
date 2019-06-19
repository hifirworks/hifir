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

using namespace psmilu;

static const RandIntGen i_rand(1, 100);

TEST(crs, c) {
  using mat_t   = CRS<double, int>;
  const auto A  = gen_rand_sparse<mat_t>(120, 120);
  const auto Ad = convert2dense(A);
  const int  n  = i_rand();
  const auto B  = A.extract_leading(n);
  const auto Bd = convert2dense(B);
  COMPARE_MATS_BLOCK(Ad, Bd, n);
}

TEST(crs, fortran) {
  using mat_t   = CRS<double, int, true>;
  const auto A  = gen_rand_sparse<mat_t>(120, 120);
  const auto Ad = convert2dense(A);
  const int  n  = i_rand();
  const auto B  = A.extract_leading(n);
  const auto Bd = convert2dense(B);
  COMPARE_MATS_BLOCK(Ad, Bd, n);
}

TEST(ccs, c) {
  using mat_t   = CCS<double, int>;
  const auto A  = gen_rand_sparse<mat_t>(120, 120);
  const auto Ad = convert2dense(A);
  const int  n  = i_rand();
  const auto B  = A.extract_leading(n);
  const auto Bd = convert2dense(B);
  COMPARE_MATS_BLOCK(Ad, Bd, n);
}

TEST(ccs, fortran) {
  using mat_t   = CCS<double, int, true>;
  const auto A  = gen_rand_sparse<mat_t>(120, 120);
  const auto Ad = convert2dense(A);
  const int  n  = i_rand();
  const auto B  = A.extract_leading(n);
  const auto Bd = convert2dense(B);
  COMPARE_MATS_BLOCK(Ad, Bd, n);
}