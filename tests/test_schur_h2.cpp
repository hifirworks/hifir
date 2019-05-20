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
#include "psmilu_Schur2.hpp"
#include "psmilu_SparseVec.hpp"

#include <gtest/gtest.h>

using namespace psmilu;

using crs_t   = CRS<double, int>;
using ccs_t   = crs_t::other_type;
using array_t = crs_t::array_type;
using perm_t  = BiPermMatrix<int>;

#define N 100
#define M 80
#define NM (N - M)

TEST(SCHUR, h) {
  // create diagonal
  const auto d = gen_ran_vec<array_t>(M);
  // create permutation
  perm_t p(N);
  p.make_eye();
  // create identity scaling arrays
  array_t s(N), t(N);
  for (int i = 0; i < N; ++i) s[i] = t[i] = 1.0;
  // create the sparse matrix A
  const auto A = gen_rand_sparse<ccs_t>(N, N);
  // create L_E and U_F
  const auto L_E = gen_rand_sparse<crs_t>(NM, M);
  const auto U_F = gen_rand_sparse<ccs_t>(M, NM);
  // create a dummy simple Schur
  const auto SC = gen_rand_sparse<ccs_t>(NM, NM);
  // create lower and upper
  const auto L = gen_rand_strict_lower_sparse<ccs_t>(M),
             U = ccs_t(gen_rand_strict_upper_sparse<crs_t>(M));
  // compute the dense H version
  auto H_d = convert2dense(SC);
  do {
    // compute the advanced part and add it to the simple part
    const auto H_d2 = compute_dense_Schur_h(
        convert2dense(A), M, convert2dense(L), convert2dense(U), d,
        convert2dense(L_E), convert2dense(U_F));
    for (int i = 0; i < NM; ++i) {
      auto &      l = H_d[i];
      const auto &r = H_d2[i];
      for (int j = 0; j < NM; ++j) l[j] += r[j];
    }
  } while (false);
  // compute the sparse version
  SparseVector<double, int> buf(N);
  const auto                H =
      compute_Schur_hybrid(L_E, L, s, A, t, p, p, d, U, U_F, SC, buf);
  const auto Hd = convert2dense(H);
  COMPARE_MATS_TOL(Hd, H_d, 1e-9);
}