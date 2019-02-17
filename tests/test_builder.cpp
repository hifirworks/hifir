//@HEADER
//----------------------------------------------------------------------------
//                Copyright (C) 2019 The PSMILU AUTHORS
//----------------------------------------------------------------------------
//@HEADER

#include "common.hpp"
// line break to avoid sorting
#include "psmilu_Array.hpp"
#include "psmilu_AugmentedStorage.hpp"
#include "psmilu_Builder.hpp"
#include "psmilu_CompressedStorage.hpp"
#include "psmilu_Crout.hpp"
#include "psmilu_PermMatrix.hpp"
#include "psmilu_Schur.hpp"
#include "psmilu_SparseVec.hpp"
#include "psmilu_diag_pivot.hpp"
#include "psmilu_fac.hpp"
#include "psmilu_inv_thres.hpp"

#include <gtest/gtest.h>

using namespace psmilu;

TEST(BUILDER, c) {
  using build_t = Builder<double, int>;
  using crs_t   = build_t::crs_type;
  const auto A  = gen_rand_sparse<crs_t>(10, 10);
  {
    const auto A_d = convert2dense(A);
    for (int i = 0; i < 10; ++i) {
      for (int j = 0; j < 10; ++j) std::cout << A_d[i][j] << ' ';
      std::cout << std::endl;
    }
  }
  build_t builder;
  Options opts = get_default_options();
  opts.tau_d   = 100.0;
  builder.compute(A, 0, opts);
  std::cout << builder.levels() << '\n';
  std::cout << builder.prec(builder.levels() - 1).dense_solver.mat().nrows()
            << '\n';
}