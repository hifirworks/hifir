///////////////////////////////////////////////////////////////////////////////
//                This file is part of HILUCSI project                       //
///////////////////////////////////////////////////////////////////////////////

#include "common.hpp"
// line break to avoid sorting
#include "hilucsi/utils/Timer.hpp"

#include <gtest/gtest.h>
#include <cmath>
#include <thread>

using namespace hilucsi;

// takes about 300ms
TEST(Timer, serial) {
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

// takes about 200ms if MT is avail
TEST(Timer, MT) {
  const auto func1 = []() {
    DefaultTimer timer;
    timer.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    timer.finish();
    ASSERT_LE(std::fabs(timer.time() - 0.1), 1e-3);
    ASSERT_LE(std::fabs(timer.time_milli() - 100.0), 2);
    ASSERT_LE(std::fabs(timer.time_micro() - 1e2 * 1e3), 500.0);
    ASSERT_LE(std::fabs(timer.time_nano() - 1e2 * 1e6), 1e6);
  };
  const auto func2 = []() {
    DefaultTimer timer;
    timer.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    timer.finish();
    ASSERT_LE(std::fabs(timer.time() - 0.2), 1e-3);
    ASSERT_LE(std::fabs(timer.time_milli() - 200.0), 2);
    ASSERT_LE(std::fabs(timer.time_micro() - 2e2 * 1e3), 500.0);
    ASSERT_LE(std::fabs(timer.time_nano() - 2e2 * 1e6), 1e6);
  };
  std::thread t1(func1), t2(func2);
  t1.join();
  t2.join();
}
