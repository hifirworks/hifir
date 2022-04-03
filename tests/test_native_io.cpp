///////////////////////////////////////////////////////////////////////////////
//                  This file is part of the HIFIR library                   //
///////////////////////////////////////////////////////////////////////////////

#include "common.hpp"
// line break to avoid sorting
#include "hif/ds/Array.hpp"
#include "hif/ds/CompressedStorage.hpp"
#include "hif/utils/io.hpp"

#include <gtest/gtest.h>

using namespace hif;

TEST(IO, core) {
#ifdef HIF_HAS_HDF5
  warn_flag(0);
  const int n  = RandIntGen(100, 300)();
  using crs_t  = CRS<double, int>;
  const auto A = gen_rand_sparse<crs_t>(n, n);
  A.write_bin("foo.hif");
  crs_t      B  = crs_t::from_bin("foo.hif");
  const auto Ad = convert2dense(A), Bd = convert2dense(B);
  COMPARE_MATS(Ad, Bd);
  using crs2_t  = CRS<double, long>;
  crs2_t     C  = crs2_t::from_bin("foo.hif");
  const auto Cd = convert2dense(C);
  COMPARE_MATS(Cd, Ad);
  using ccs_t   = CCS<double, int>;
  ccs_t      E  = ccs_t::from_bin("foo.hif");
  const auto Ed = convert2dense(E);
  COMPARE_MATS(Ed, Ad);
  using ccs2_t  = CCS<float, int>;
  ccs2_t     F  = ccs2_t::from_bin("foo.hif");
  const auto Fd = convert2dense(F);
  COMPARE_MATS_TOL(Fd, Ad, 1e-6);
#else
  hif_warning("HDF5 is not available! IO tests were skipped");
#endif
}
