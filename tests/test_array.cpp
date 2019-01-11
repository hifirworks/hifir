// psmilu_Array.hpp psmilu_log.hpp

//@HEADER
//----------------------------------------------------------------------------
//                Copyright (C) 2019 The PSMILU AUTHORS
//----------------------------------------------------------------------------
//@HEADER

#include "common.hpp"
// line break to avoid sorting
#include "psmilu_Array.hpp"

#include <gtest/gtest.h>
#include <vector>

using namespace psmilu;

TEST(Array_api, test1) {
  Array<double> v;
  ASSERT_EQ(v.status(), DATA_UNDEF);
  v.resize(100);
  ASSERT_EQ(v.status(), DATA_OWN);
  ASSERT_EQ(v.size(), 100u);
  for (auto &i : v) i = 1.0;
  for (const auto i : v) ASSERT_EQ(i, 1.0);
  v.resize(0);
  ASSERT_GT(v.capacity(), 100u);
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

TEST(Array_api, test2) {
  std::vector<int> _v(100);
  Array<int>       v(100u, _v.data(), true);
  ASSERT_EQ(v.status(), DATA_WRAP);
  for (auto &i : v) i = -100.0;
  for (const auto i : _v) ASSERT_EQ(i, -100.0);
}
