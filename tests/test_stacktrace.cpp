// psmilu_log.hpp psmilu_stacktrace.hpp

//@HEADER
//----------------------------------------------------------------------------
//                Copyright (C) 2019 The PSMILU AUTHORS
//----------------------------------------------------------------------------
//@HEADER

#include "common.hpp"
// line break to avoid sorting
#include "psmilu_Array.hpp"

#include <gtest/gtest.h>

using namespace psmilu;

template <typename T>
void bad_func(const Array<T> &v) {
  const T foobar = v[v.size() + 1];
  (void)foobar;
}

template <typename T>
void foo1(const Array<T> &v) {
  bad_func(v);
}

template <typename T>
void foo2(const Array<T> &v) {
  foo1(v);
}

TEST(StackTrace, test) {
  try {
    Array<int> v;
    foo2(v);
  } catch (const std::runtime_error &e) {
    PSMILU_STDOUT(e.what());
  }
}
