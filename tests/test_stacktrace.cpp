///////////////////////////////////////////////////////////////////////////////
//                  This file is part of HIF project                         //
///////////////////////////////////////////////////////////////////////////////

#include "common.hpp"
// line break to avoid sorting
#include "hif/ds/Array.hpp"
#include "hif/utils/log.hpp"
#include "hif/utils/stacktrace.hpp"

#include <gtest/gtest.h>

#define TAG "stack trace:"

using namespace hif;

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
    const std::string msg   = e.what();
    const auto        n     = msg.find(TAG);
    const bool        found = n != std::string::npos;
    ASSERT_TRUE(found);
#else
    psmilu_warning("stack trace test only avail on Linux");
#endif
  }
}
