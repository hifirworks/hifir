//@HEADER
//----------------------------------------------------------------------------
//                Copyright (C) 2019 The PSMILU AUTHORS
//----------------------------------------------------------------------------
//@HEADER

#include "common.hpp"
// line break to avoid sorting
#include "psmilu_Array.hpp"
#include "psmilu_utils.hpp"

#include <gtest/gtest.h>

#include <cstdlib>
#include <ctime>

using namespace psmilu;

TEST(ROTATE, left) {
  std::srand(std::time(0));
  Array<int> v(200);
  for (auto &val : v) val = std::rand() % 100000;
  const int n   = (std::rand() % 200) + 1;
  int       pos = std::rand() % 200;
  if (pos + n >= 200) pos = 200 - n;
  decltype(v) v2(n);
  // copy the original range
  std::cout << "rot:left,size=200,n=" << n << ",pos=" << pos << '\n';
  for (int i = 0, j = pos; i < n; ++i, ++j) v2[i] = v[j];
  rotate_left(n, pos, v);
  int j = pos;
  for (int i = 1; i < n; ++i, ++j) ASSERT_EQ(v[j], v2[i]);
  ASSERT_EQ(v2[0], v[j]);
}

TEST(ROTATE, right) {
  std::srand(std::time(nullptr));
  Array<int> v(2000);
  for (auto &val : v) val = std::rand() % 1000000;
  const int n   = (std::rand() % 2000) + 1;
  int       pos = std::rand() % 2000;
  if (n > pos + 1) pos = n - 1;
  decltype(v) v2(n);
  // copy the original range
  std::cout << "rot:right,size=2000,n=" << n << ",pos=" << pos << '\n';
  for (int i = 0, j = pos - n + 1; i < n; ++i, ++j) v2[i] = v[j];
  rotate_right(n, pos, v);
  for (int i = 0, j = pos - n + 2; i < n - 1; ++i, ++j) ASSERT_EQ(v[j], v2[i]);
  ASSERT_EQ(v[pos - n + 1], v2[n - 1]);
}
