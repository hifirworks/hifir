///////////////////////////////////////////////////////////////////////////////
//                This file is part of HILUCSI project                       //
///////////////////////////////////////////////////////////////////////////////

#include "common.hpp"
// line break to avoid sorting
#include "hilucsi/ds/Array.hpp"
#include "hilucsi/utils/log.hpp"
#include "hilucsi/utils/stacktrace.hpp"

#include <gtest/gtest.h>
#include <regex>

#define TAG "stack trace:"

using namespace hilucsi;

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
#ifdef __GNUC__
    const char *msg = e.what();
    std::regex  st(TAG "*");
    std::cmatch mt;
    const bool  found = std::regex_search(msg, mt, st);
    ASSERT_TRUE(found);
#else
    psmilu_warning("stack trace test only avail on Linux");
#endif
  }
}
