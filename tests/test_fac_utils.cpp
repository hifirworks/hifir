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
#include "psmilu_fac.hpp"

#include <gtest/gtest.h>

using namespace psmilu;

TEST(FAC_U, diag) {
  {
    using ccs_t         = CCS<double, int>;
    const auto        A = gen_rand_sparse<ccs_t>(100, 100);
    const int         m = RandIntGen(1, 99)();
    BiPermMatrix<int> p(100), q(100);
    p.make_eye();
    q.make_eye();
    std::shuffle(p().begin(), p().end(), std::mt19937_64(std::time(0)));
    std::shuffle(q().begin(), q().end(), std::mt19937_64(std::time(0)));
    p.build_inv();
    q.build_inv();
    const auto d   = internal::extract_perm_diag(A, m, p, q);
    const auto A_d = convert2dense(A);
    for (int i = 0; i < m; ++i) EXPECT_EQ(d[i], A_d[p[i]][q[i]]);
  }
  {
    using ccs_t         = CCS<double, int, true>;
    const auto        A = gen_rand_sparse<ccs_t>(100, 100);
    const int         m = RandIntGen(1, 99)();
    BiPermMatrix<int> p(100), q(100);
    p.make_eye();
    q.make_eye();
    std::shuffle(p().begin(), p().end(), std::mt19937_64(std::time(0)));
    std::shuffle(q().begin(), q().end(), std::mt19937_64(std::time(0)));
    p.build_inv();
    q.build_inv();
    const auto d   = internal::extract_perm_diag(A, m, p, q);
    const auto A_d = convert2dense(A);
    for (int i = 0; i < m; ++i) EXPECT_EQ(d[i], A_d[p[i]][q[i]]);
  }
}
