///////////////////////////////////////////////////////////////////////////////
//                  This file is part of the HIFIR library                   //
///////////////////////////////////////////////////////////////////////////////

#include "common.hpp"
// line break to avoid sorting
#include "hif/alg/Schur.hpp"
#include "hif/ds/Array.hpp"
#include "hif/ds/CompressedStorage.hpp"
#include "hif/ds/PermMatrix.hpp"
#include "hif/ds/SparseVec.hpp"

#include <gtest/gtest.h>

using namespace hif;

TEST(compute_Schur, core) {
  using mat_t   = CRS<double, int>;
  using perm_t  = BiPermMatrix<int>;
  using array_t = Array<double>;
  using spvec_t = SparseVector<double, int>;
  const auto A  = gen_rand_sparse<mat_t>(100, 100);
  const auto d  = gen_ran_vec<array_t>(100);
  perm_t     p(100), q(100);
  p.make_eye();
  q.make_eye();
  array_t s(100), t(100);
  std::fill(s.begin(), s.end(), 1.0);
  std::fill(t.begin(), t.end(), 1.0);
  const mat_t::size_type m(80);
  spvec_t                buf(100);
  const auto             LE = gen_rand_sparse<mat_t>(20, 80),
             UF             = gen_rand_sparse<mat_t>(80, 20);
  const auto SC     = compute_Schur_simple(s, A, t, p, q, m, LE, d, UF, buf);
  const auto SC_ref = compute_dense_Schur_c(convert2dense(A), convert2dense(LE),
                                            convert2dense(UF), d, m);
  const auto SC_d   = convert2dense(SC);
  COMPARE_MATS_TOL(SC_d, SC_ref, 1e-10);
}