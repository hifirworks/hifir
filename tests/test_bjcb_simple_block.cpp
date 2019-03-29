//@HEADER
//----------------------------------------------------------------------------
//                Copyright (C) 2019 The PSMILU AUTHORS
//----------------------------------------------------------------------------
//@HEADER

#include "common.hpp"
// line break to avoid sorting
#include "psmilu_Array.hpp"
#include "psmilu_CompressedStorage.hpp"
#include "psmilu_blockJacobi/block_buillder.hpp"

#include <gtest/gtest.h>

using namespace psmilu;

const static RandIntGen i_rand(0, 100);

TEST(SIMPLE, c) {
  do {
    using crs_t   = CRS<double, int>;
    const auto A  = gen_rand_sparse<crs_t>(100, 100);
    const auto AD = convert2dense(A);
    // test leading block
    do {
      const auto B  = bjacobi::simple_block_build(A, 0, 30);
      const auto BD = convert2dense(B);
      for (int i = 0; i < 20; ++i)
        for (int j = 0; j < 20; ++j)
          EXPECT_EQ(AD[i][j], BD[i][j]) << i << ',' << j << " failed\n";
    } while (false);
    // test ending block
    do {
      const auto B  = bjacobi::simple_block_build(A, 30, 100);
      const auto BD = convert2dense(B);
      for (int i = 30; i < 100; ++i)
        for (int j = 30; j < 100; ++j)
          EXPECT_EQ(AD[i][j], BD[i - 30][j - 30])
              << i << ',' << j << " failed\n";
    } while (false);
    // randome tests
    for (int test = 0; test < 10; ++test) {
      const int start = i_rand();
      int       end   = i_rand() + start;
      if (end > 100) end = 100;
      const auto B  = bjacobi::simple_block_build(A, start, end);
      const auto BD = convert2dense(B);
      for (int i = start; i < end; ++i)
        for (int j = start; j < end; ++j)
          EXPECT_EQ(AD[i][j], BD[i - start][j - start])
              << "failed with start=" << start << " and end=" << end << '\n';
    }
  } while (0);
  do {
    using ccs_t   = CCS<double, int>;
    const auto A  = gen_rand_sparse<ccs_t>(100, 100);
    const auto AD = convert2dense(A);
    // test leading block
    do {
      const auto B  = bjacobi::simple_block_build(A, 0, 30);
      const auto BD = convert2dense(B);
      for (int i = 0; i < 20; ++i)
        for (int j = 0; j < 20; ++j)
          EXPECT_EQ(AD[i][j], BD[i][j]) << i << ',' << j << " failed\n";
    } while (false);
    // test ending block
    do {
      const auto B  = bjacobi::simple_block_build(A, 30, 100);
      const auto BD = convert2dense(B);
      for (int i = 30; i < 100; ++i)
        for (int j = 30; j < 100; ++j)
          EXPECT_EQ(AD[i][j], BD[i - 30][j - 30])
              << i << ',' << j << " failed\n";
    } while (false);
    // randome tests
    for (int test = 0; test < 10; ++test) {
      const int start = i_rand();
      int       end   = i_rand() + start;
      if (end > 100) end = 100;
      const auto B  = bjacobi::simple_block_build(A, start, end);
      const auto BD = convert2dense(B);
      for (int i = start; i < end; ++i)
        for (int j = start; j < end; ++j)
          EXPECT_EQ(AD[i][j], BD[i - start][j - start])
              << "failed with start=" << start << " and end=" << end << '\n';
    }
  } while (0);
}

TEST(SIMPLE, fortran) {
  do {
    using crs_t   = CRS<double, int, true>;
    const auto A  = gen_rand_sparse<crs_t>(100, 100);
    const auto AD = convert2dense(A);
    // test leading block
    do {
      const auto B  = bjacobi::simple_block_build(A, 0, 30);
      const auto BD = convert2dense(B);
      for (int i = 0; i < 20; ++i)
        for (int j = 0; j < 20; ++j)
          EXPECT_EQ(AD[i][j], BD[i][j]) << i << ',' << j << " failed\n";
    } while (false);
    // test ending block
    do {
      const auto B  = bjacobi::simple_block_build(A, 30, 100);
      const auto BD = convert2dense(B);
      for (int i = 30; i < 100; ++i)
        for (int j = 30; j < 100; ++j)
          EXPECT_EQ(AD[i][j], BD[i - 30][j - 30])
              << i << ',' << j << " failed\n";
    } while (false);
    // randome tests
    for (int test = 0; test < 10; ++test) {
      const int start = i_rand();
      int       end   = i_rand() + start;
      if (end > 100) end = 100;
      const auto B  = bjacobi::simple_block_build(A, start, end);
      const auto BD = convert2dense(B);
      for (int i = start; i < end; ++i)
        for (int j = start; j < end; ++j)
          EXPECT_EQ(AD[i][j], BD[i - start][j - start])
              << "failed with start=" << start << " and end=" << end << '\n';
    }
  } while (0);
  do {
    using ccs_t   = CCS<double, int, true>;
    const auto A  = gen_rand_sparse<ccs_t>(100, 100);
    const auto AD = convert2dense(A);
    // test leading block
    do {
      const auto B  = bjacobi::simple_block_build(A, 0, 30);
      const auto BD = convert2dense(B);
      for (int i = 0; i < 20; ++i)
        for (int j = 0; j < 20; ++j)
          EXPECT_EQ(AD[i][j], BD[i][j]) << i << ',' << j << " failed\n";
    } while (false);
    // test ending block
    do {
      const auto B  = bjacobi::simple_block_build(A, 30, 100);
      const auto BD = convert2dense(B);
      for (int i = 30; i < 100; ++i)
        for (int j = 30; j < 100; ++j)
          EXPECT_EQ(AD[i][j], BD[i - 30][j - 30])
              << i << ',' << j << " failed\n";
    } while (false);
    // randome tests
    for (int test = 0; test < 10; ++test) {
      const int start = i_rand();
      int       end   = i_rand() + start;
      if (end > 100) end = 100;
      const auto B  = bjacobi::simple_block_build(A, start, end);
      const auto BD = convert2dense(B);
      for (int i = start; i < end; ++i)
        for (int j = start; j < end; ++j)
          EXPECT_EQ(AD[i][j], BD[i - start][j - start])
              << "failed with start=" << start << " and end=" << end << '\n';
    }
  } while (0);
}