//@HEADER
//----------------------------------------------------------------------------
//                Copyright (C) 2019 The PSMILU AUTHORS
//----------------------------------------------------------------------------
//@HEADER

#include "common.hpp"
// line break to avoid sorting
#include "hilucsi/ds/Array.hpp"

#include <gtest/gtest.h>
#include <list>
#include <utility>
#include <vector>

using namespace hilucsi;

TEST(Array_api, test_core) {
  Array<double> v;
  ASSERT_EQ(v.status(), DATA_UNDEF);
  v.resize(100);
  ASSERT_EQ(v.status(), DATA_OWN);
  ASSERT_EQ(v.size(), 100u);
  for (auto &i : v) i = 1.0;
  for (const auto i : v) ASSERT_EQ(i, 1.0);
  v.resize(0);
  ASSERT_GE(v.capacity(), 100u);
  // test shallow
  {
    Array<double> b(200u, 0.0);
    v = b;
  }
  ASSERT_EQ(v.size(), 200u);
  for (int i = 0; i < 200; ++i) v[i] = (double)i;
  for (int i = 0; i < 200; ++i) ASSERT_EQ(*(v.cbegin() + i), (double)i);
  // shallow again
  Array<double> b(v);
  ASSERT_EQ(v.data(), b.data());
  for (auto &i : b) i = -1.0;
  for (const auto i : v) ASSERT_EQ(i, -1.0);
}

TEST(Array_api, test_wrap) {
  std::vector<int> _v(100);
  Array<int>       v(100u, _v.data(), true);
  ASSERT_EQ(v.status(), DATA_WRAP);
  for (auto &i : v) i = -100.0;
  for (const auto i : _v) ASSERT_EQ(i, -100.0);
}

TEST(Array_api, test_pushback) {
  Array<long> v;
  v.push_back(1l);
  ASSERT_EQ(v.front(), 1l);
  ASSERT_EQ(v.back(), 1l);
  for (int i = 0; i < 9; ++i) v.push_back(v.front() + i + 1);
  {
    long j(1l);
    for (const auto i : v) ASSERT_EQ(i, j++);
  }
  const auto        n = v.size();
  std::vector<long> b(v.cbegin(), v.cend());
  v.push_back(b.cbegin(), b.cend());
  ASSERT_EQ(v.size(), n * 2u);
  for (long i = 0l; i < (long)v.size(); ++i) ASSERT_EQ((i % n) + 1l, v[i]);
  // test list
  std::list<long> c(b.cbegin(), b.cend());
  v.push_back(c.cbegin(), c.cend());
  ASSERT_EQ(v.size(), n * 3u);
  for (long i = 0l; i < (long)v.size(); ++i) ASSERT_EQ((i % n) + 1l, v[i]);
}

TEST(Array_api, test_move) {
  Array<float> v1(100);
  Array<float> v2(std::move(v1));
  ASSERT_EQ(v1.status(), DATA_UNDEF);
  ASSERT_EQ(v1.size(), 0u);
  ASSERT_EQ(v2.size(), 100u);
  ASSERT_EQ(v2.status(), DATA_OWN);
  Array<float> v3;
  v3 = std::move(v2);
  ASSERT_EQ(v2.status(), DATA_UNDEF);
  ASSERT_EQ(v2.size(), 0u);
  ASSERT_EQ(v3.size(), 100u);
  ASSERT_EQ(v3.status(), DATA_OWN);
}

// run this with valgrind as well
TEST(Array_api, test_steal) {
  Array<double> v1;
  v1.reserve(100u);
  v1.resize(10u);
  // create several aliases
  std::vector<Array<double>> pools;
  for (;;) {
    if (pools.size() == 10u) break;
    pools.emplace_back(v1);
  }
  double *                 data = nullptr;
  Array<double>::size_type size, cap;
  // steal
  steal_array_ownership(v1, data, size, cap);
  ASSERT_NE(data, nullptr);
  ASSERT_EQ(size, 10u);
  ASSERT_EQ(cap, 100u);
  data[99] = 100.0;
  delete[] data;
}
