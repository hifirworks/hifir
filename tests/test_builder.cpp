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
  const auto A  = crs_t::from_native_bin("../tools/symm_K_F_2d_2955.psmilu");
  build_t    builder;
  builder.compute(A);
  std::cout << "levels=" << builder.levels() << '\n';
  std::cout << builder.prec(0).m << ' ' << builder.prec(0).n << '\n';
}