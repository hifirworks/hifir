///////////////////////////////////////////////////////////////////////////////
//                  This file is part of the HIFIR library                         //
///////////////////////////////////////////////////////////////////////////////

#include "common.hpp"
// line break to avoid sorting
#include "hif/ds/Array.hpp"
#include "hif/ds/CompressedStorage.hpp"
#include "hif/pre/matching_scaling.hpp"

#include <gtest/gtest.h>

using namespace hif;

constexpr static int N = 100;

const static RandIntGen i_rand(1, N);

TEST(NZ_PT_APAT, core) {
  using crs_t = CRS<double, int>;
  using ccs_t = crs_t::other_type;

  const int   m(i_rand());
  const crs_t A1 = gen_rand_sparse<crs_t>(m, m);
  const ccs_t A2(A1);

  struct {
    inline constexpr int operator[](const int i) const { return i; }
    inline constexpr int inv(const int i) const { return i; }
  } dummy_p;

  auto B = internal::compute_leading_block(A2, A1, m, dummy_p, dummy_p, true);

  for (int i = 0; i < m; ++i) {
    std::vector<bool> mask(m, false);
    for (auto itr = A1.col_ind_cbegin(i); itr != A1.col_ind_cend(i); ++itr)
      mask[*itr] = true;
    for (auto itr = A2.row_ind_cbegin(i); itr != A2.row_ind_cend(i); ++itr)
      mask[*itr] = true;
    for (auto itr = B.row_ind_cbegin(i); itr != B.row_ind_cend(i); ++itr) {
      EXPECT_TRUE(mask[*itr]);
      mask[*itr] = false;
    }
    EXPECT_TRUE(std::all_of(mask.cbegin(), mask.cend(),
                            [](const bool i) { return !i; }));
  }

  // assign dummy value array in order to build the crs of B
  B.vals() = gen_ran_vec<ccs_t::array_type>(B.nnz());
  const crs_t B2(B);
  EXPECT_TRUE(std::equal(B.col_start().cbegin(), B.col_start().cend(),
                         B2.row_start().cbegin()));
  EXPECT_TRUE(std::equal(B.row_ind().cbegin(), B.row_ind().cend(),
                         B2.col_ind().cbegin()));
}
