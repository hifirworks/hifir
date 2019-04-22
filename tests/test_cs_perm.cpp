//@HEADER
//----------------------------------------------------------------------------
//                Copyright (C) 2019 The PSMILU AUTHORS
//----------------------------------------------------------------------------
//@HEADER

#include "common.hpp"
// line break to avoid sorting
#include "psmilu_Array.hpp"
#include "psmilu_CompressedStorage.hpp"
#include "psmilu_PermMatrix.hpp"

#include <gtest/gtest.h>

using namespace psmilu;

static const RandIntGen i_rand(1, 100);

using perm_t = BiPermMatrix<int>;

TEST(crs, c) {
  using mat_t  = CRS<double, int>;
  const auto A = gen_rand_sparse<mat_t>(100, 100);
  perm_t     p(100), q(100);
  p.make_eye();
  q.make_eye();
  std::shuffle(p().begin(), p().end(), std::mt19937_64(std::time(0)));
  p.build_inv();
  std::shuffle(q().begin(), q().end(), std::mt19937_64(std::time(0)));
  q.build_inv();
  const auto AD = convert2dense(A);
  const auto BD = convert2dense(A.compute_perm(p(), q.inv()));
  for (int i = 0; i < 100; ++i)
    for (int j = 0; j < 100; ++j) EXPECT_EQ(BD[i][j], AD[p[i]][q[j]]);
  const int  m   = i_rand();
  const auto BD2 = convert2dense(A.compute_perm(p(), q.inv(), m));
  for (int i = 0; i < m; ++i)
    for (int j = 0; j < m; ++j) EXPECT_EQ(BD2[i][j], AD[p[i]][q[j]]);
}

TEST(crs, fortran) {
  using mat_t  = CRS<double, int>;
  const auto A = gen_rand_sparse<mat_t>(100, 100);
  perm_t     p(100), q(100);
  p.make_eye();
  q.make_eye();
  std::shuffle(p().begin(), p().end(), std::mt19937_64(std::time(0)));
  p.build_inv();
  std::shuffle(q().begin(), q().end(), std::mt19937_64(std::time(0)));
  q.build_inv();
  const auto AD = convert2dense(A);
  const auto BD = convert2dense(A.compute_perm(p(), q.inv()));
  for (int i = 0; i < 100; ++i)
    for (int j = 0; j < 100; ++j) EXPECT_EQ(BD[i][j], AD[p[i]][q[j]]);
  const int  m   = i_rand();
  const auto BD2 = convert2dense(A.compute_perm(p(), q.inv(), m));
  for (int i = 0; i < m; ++i)
    for (int j = 0; j < m; ++j) EXPECT_EQ(BD2[i][j], AD[p[i]][q[j]]);
}

TEST(ccs, c) {
  using mat_t  = CCS<double, int>;
  const auto A = gen_rand_sparse<mat_t>(100, 100);
  perm_t     p(100), q(100);
  p.make_eye();
  q.make_eye();
  std::shuffle(p().begin(), p().end(), std::mt19937_64(std::time(0)));
  p.build_inv();
  std::shuffle(q().begin(), q().end(), std::mt19937_64(std::time(0)));
  q.build_inv();
  const auto AD = convert2dense(A);
  const auto BD = convert2dense(A.compute_perm(p.inv(), q()));
  for (int i = 0; i < 100; ++i)
    for (int j = 0; j < 100; ++j) EXPECT_EQ(BD[i][j], AD[p[i]][q[j]]);
  const int  m   = i_rand();
  const auto BD2 = convert2dense(A.compute_perm(p.inv(), q(), m));
  for (int i = 0; i < m; ++i)
    for (int j = 0; j < m; ++j) EXPECT_EQ(BD2[i][j], AD[p[i]][q[j]]);
}

TEST(ccs, fortran) {
  using mat_t  = CCS<double, int, true>;
  const auto A = gen_rand_sparse<mat_t>(100, 100);
  perm_t     p(100), q(100);
  p.make_eye();
  q.make_eye();
  std::shuffle(p().begin(), p().end(), std::mt19937_64(std::time(0)));
  p.build_inv();
  std::shuffle(q().begin(), q().end(), std::mt19937_64(std::time(0)));
  q.build_inv();
  const auto AD = convert2dense(A);
  const auto BD = convert2dense(A.compute_perm(p.inv(), q()));
  for (int i = 0; i < 100; ++i)
    for (int j = 0; j < 100; ++j) EXPECT_EQ(BD[i][j], AD[p[i]][q[j]]);
  const int  m   = i_rand();
  const auto BD2 = convert2dense(A.compute_perm(p.inv(), q(), m));
  for (int i = 0; i < m; ++i)
    for (int j = 0; j < m; ++j) EXPECT_EQ(BD2[i][j], AD[p[i]][q[j]]);
}