///////////////////////////////////////////////////////////////////////////////
//                  This file is part of the HIFIR library                   //
///////////////////////////////////////////////////////////////////////////////

#include "common.hpp"
// line break to avoid sorting
#include "hif/alg/Schur.hpp"
#include "hif/ds/Array.hpp"
#include "hif/ds/CompressedStorage.hpp"

#include <gtest/gtest.h>

using namespace hif;

TEST(drop_LE_UF, core) {
  do {
    using mat_t        = CRS<double, int>;
    const auto    A    = gen_rand_sparse<mat_t>(100, 100);
    auto          L    = gen_rand_sparse<mat_t>(20, 90);
    const auto    nnz1 = L.nnz();
    Array<double> buf(100);
    Array<int>    ibuf(100);
    drop_L_E(A.row_start(), 2, L, buf, ibuf);
    ASSERT_LE(L.nnz(), nnz1);
    std::cout << "nnz(LE)-b4: " << nnz1 << ", nnz(LE)-after: " << L.nnz()
              << std::endl;
  } while (false);

  do {
    using mat_t        = CCS<double, int>;
    const auto    A    = gen_rand_sparse<mat_t>(100, 100);
    auto          U    = gen_rand_sparse<mat_t>(90, 20);
    const auto    nnz1 = U.nnz();
    Array<double> buf(100);
    Array<int>    ibuf(100);
    drop_U_F(A.col_start(), 2, U, buf, ibuf);
    ASSERT_LE(U.nnz(), nnz1);
    std::cout << "nnz(UF)-b4: " << nnz1 << ", nnz(UF)-after: " << U.nnz()
              << std::endl;
  } while (false);
}
