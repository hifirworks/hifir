//@HEADER
//----------------------------------------------------------------------------
//                Copyright (C) 2019 The PSMILU AUTHORS
//----------------------------------------------------------------------------
//@HEADER

#include "common.hpp"
// line break to avoid sorting
#include "psmilu_Array.hpp"
#include "psmilu_CompressedStorage.hpp"
#include "psmilu_DeferredCrout_thin.hpp"

#include <gtest/gtest.h>

#define N 50

using namespace psmilu;

using crs_t              = CRS<double, int>;
using ccs_t              = crs_t::other_type;
using iarray_t           = crs_t::iarray_type;
using array_t            = crs_t::array_type;
constexpr static int nil = -1;

template <class Cs>
inline void load_across(const std::size_t k, const Cs &T, const iarray_t &list,
                        const iarray_t &start, iarray_t &indices, array_t &v) {
  // reset indices work space
  indices.resize(0);
  int idx = list[k];
  while (idx != nil) {
    indices.push_back(idx);
    v[idx] = *(T.val_cbegin(idx) + start[idx]);
    idx    = list[idx];
  }
  // sort
  if (indices.size()) std::sort(indices.begin(), indices.end());
}

TEST(linked_list, L) {
  const auto L_ref      = gen_rand_strict_lower_sparse<ccs_t>(N);
  const auto L_ref_dual = crs_t(L_ref);
  ccs_t      L;
  L.resize(N, N);
  L.reserve(L_ref.nnz());
  L.begin_assemble_cols();
  iarray_t start(N), list(N), i_buf;
  for (auto &v : list) v = nil;
  i_buf.reserve(N);
  array_t            buf(N);
  DeferredCrout_thin k;
  for (; k < (size_t)N; ++k) {
    if (k) {
      load_across(k, L, list, start, i_buf, buf);
      EXPECT_EQ(i_buf.size(), L_ref_dual.nnz_in_row(k));
      EXPECT_TRUE(std::equal(i_buf.cbegin(), i_buf.cend(),
                             L_ref_dual.col_ind_cbegin(k)));
    }
    auto v_itr = L_ref.val_cbegin(k);
    for (auto itr = L_ref.row_ind_cbegin(k); itr != L_ref.row_ind_cend(k);
         ++itr, ++v_itr)
      buf[*itr] = *v_itr;
    L.push_back_col(k, L_ref.row_ind_cbegin(k), L_ref.row_ind_cend(k), buf);
    k.update_compress(L, list, start);
  }
  L.end_assemble_cols();
}