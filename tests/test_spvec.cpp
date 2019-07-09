//@HEADER
//----------------------------------------------------------------------------
//                Copyright (C) 2019 The HILUCSI AUTHORS
//----------------------------------------------------------------------------
//@HEADER

#include "common.hpp"
// line break to avoid sorting
#include "hilucsi/ds/SparseVec.hpp"

#include <gtest/gtest.h>

using namespace hilucsi;

const static RandIntGen i_rand(0, 4000);

template <typename V, typename I>
class Caster : public SparseVector<V, I> {
 public:
  typedef SparseVector<V, I> base;
  const std::vector<bool> &  get_sparse_tags() const {
    return base::_sparse_tags;
  }
  const std::vector<I> &get_dense_tags() const { return base::_dense_tags; }
};

constexpr static int empty = -1;

TEST(SP_VEC, core) {
  using sp_t  = SparseVector<double, int>;
  const int n = i_rand() + 1;
  auto      v = sp_t(n);
  v.reset_counter();
  ASSERT_EQ(v.size(), 0u);
  // push to zero step
  for (int i = 0; i < n; ++i) v.push_back(i_rand() % n, 0);
  ASSERT_LE(v.size(), n);
  const int m = v.size();
  std::cout << "n=" << n << ", m=" << m << '\n';
  // test there should not have overlap indices
  std::vector<bool> tags(n, false);
  for (int i = 0; i < m; ++i) {
    const auto idx = v.idx(i);
    if (!tags[idx]) {
      tags[idx] = true;
      continue;
    }
    ASSERT_FALSE(true) << idx << " appears at least twice!\n";
  }
  // remove all even indices
  for (int i = 0; i < m; ++i) v.idx(i) % 2 ? (void)0 : v.mark_delete(i);
  v.compress_indices();  // NOTE that internal tags have been reset here
  const int m2 = v.size();
  ASSERT_LT(m2, m) << "new size should be no larger than the old size\n";
  for (int i = 0; i < m2; ++i)
    ASSERT_EQ(v.idx(i) % 2, 1)
        << v.idx(i) << " should not appear in the list!\n";
  // internal test, the sparse tags should be reset all to false!
  const auto &caster  = static_cast<const Caster<double, int> &>(v);
  const auto &sp_tags = caster.get_sparse_tags();
  ASSERT_TRUE(std::none_of(sp_tags.cbegin(), sp_tags.cend(), [](bool foo) {
    return foo;
  })) << "sparse tags should be reset to false!\n";
  // test sort
  v.sort_indices();
  ASSERT_TRUE(std::is_sorted(v.inds().cbegin(), v.inds().cbegin() + v.size()));
  // test resize
  const auto n2 = 2 * n;
  v.resize(n2);
  v.reset_counter();
  ASSERT_EQ(v.size(), 0u);
  // internally, both dense tags and sparse tags should be of size n2
  const auto &ds_tags = caster.get_dense_tags();
  ASSERT_EQ(sp_tags.size(), ds_tags.size());
  ASSERT_EQ((int)sp_tags.size(), n2);
  // all should be reset to default stage
  ASSERT_TRUE(std::none_of(sp_tags.cbegin(), sp_tags.cend(), [](bool foo) {
    return foo;
  })) << "sparse tags should be reset to false!\n";
  ASSERT_TRUE(std::all_of(ds_tags.cbegin(), ds_tags.cend(),
                          [](int v) { return v == empty; }));
}