//@HEADER
//----------------------------------------------------------------------------
//                Copyright (C) 2019 The PSMILU AUTHORS
//----------------------------------------------------------------------------
//@HEADER

#include "common.hpp"
// line break to avoid sorting
#include "psmilu_Timer.hpp"

#include <gtest/gtest.h>
#include <cmath>
#include <thread>

using namespace psmilu;

TEST(Timer, default_) {
  DefaultTimer timer;
  timer.start();
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  timer.finish();
  ASSERT_LE(std::fabs(timer.time() - 0.1), 1e-3);
  ASSERT_LE(std::fabs(timer.time_milli() - 100.0), 2);
  ASSERT_LE(std::fabs(timer.time_micro() - 1e2 * 1e3), 500.0);
  ASSERT_LE(std::fabs(timer.time_nano() - 1e2 * 1e6), 1e6);
  // reuse
  timer.start();
  std::this_thread::sleep_for(std::chrono::milliseconds(200));
  timer.finish();
  ASSERT_LE(std::fabs(timer.time() - 0.2), 1e-3);
  ASSERT_LE(std::fabs(timer.time_milli() - 200.0), 2);
  ASSERT_LE(std::fabs(timer.time_micro() - 2e2 * 1e3), 500.0);
  ASSERT_LE(std::fabs(timer.time_nano() - 2e2 * 1e6), 1e6);
}
