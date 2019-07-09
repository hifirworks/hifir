//@HEADER
//----------------------------------------------------------------------------
//                Copyright (C) 2019 The PSMILU AUTHORS
//----------------------------------------------------------------------------
//@HEADER

#include "common.hpp"
// line break to avoid sorting
#include "psmilu_SparseVec.hpp"
#include "psmilu_inv_thres.hpp"

#include <gtest/gtest.h>

using namespace psmilu;

const static RandIntGen  i_rand(0, 1000);
const static RandRealGen r_rand(-100.0, 100.0);

template <class SpVecType>
class SparseTagExtactor : public SpVecType {
 public:
  using base = SpVecType;
  const std::vector<bool> &sp_tags() const { return base::_sparse_tags; }
};

TEST(INV_THRES, c) {
  using sp_t = SparseVector<double, int>;

  const int n = i_rand() + 1;
  sp_t      v(n);
  int       runs = (i_rand() % 10) + 1;
  do {
    std::cout << runs << '\n';
    v.reset_counter();
    const int m1 = i_rand() % n;
    for (int counts = 0; counts < m1;)
      counts += v.push_back(i_rand() % n, runs);
    ASSERT_EQ((int)v.size(), m1);
    auto &vals = v.vals();
    for (int i = 0; i < m1; ++i) vals[v.c_idx(i)] = r_rand();
    // alpha=2, tau=10, kappa=1, nnz >= 1

    const int m2 =
        apply_dropping_and_sort(10.0, 1.0, std::max(m1 / 3, 1), 2, v);
    const int m3 = v.size();
    std::cout << "original size: " << m1 << ", intermediate size: " << m2
              << ", final size: " << m3 << '\n';
    const auto &inds = v.inds();
    ASSERT_TRUE(std::is_sorted(inds.cbegin(), inds.cbegin() + m3));
    if (m2 > m3) {
      // test the selection, using low-level std::vector

      const auto &vals = v.vals();
      for (int i = 0; i < m3; ++i) {
        const double a = std::abs(vals[inds[i]]);
        for (int j = m3; j < m2; ++j)
          EXPECT_GE(a, std::abs(vals[inds[j]]))
              << i << " entry should no smaller than " << j << " entry!\n";
      }
    }
    // internal assertion
    using sp2_t         = SparseTagExtactor<sp_t>;
    const auto &sp_tags = static_cast<const sp2_t &>(v).sp_tags();
    ASSERT_TRUE(std::none_of(sp_tags.cbegin(), sp_tags.cbegin() + m1,
                             [](bool v) { return v; }));
  } while ((runs -= 1) != 0);
}
