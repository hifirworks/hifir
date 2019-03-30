//@HEADER
//----------------------------------------------------------------------------
//                Copyright (C) 2019 The PSMILU AUTHORS
//----------------------------------------------------------------------------
//@HEADER

#include "common.hpp"
// line break to avoid sorting
#include "psmilu_Array.hpp"
#include "psmilu_CompressedStorage.hpp"
#include "psmilu_blockJacobi/GraphPart.hpp"

#include <gtest/gtest.h>

using namespace psmilu;

const static RandIntGen i_rand(2, 10);

using graph_t = bjacobi::GraphPart<int>;

TEST(GRAPH, c) {
  do {
    using crs_t      = CRS<double, int>;
    const auto A     = gen_rand_sparse<crs_t>(100, 100);
    const auto nnz_A = A.nnz();
    const auto AD    = convert2dense(A);
    graph_t    G;
    const int  parts = i_rand();
    std::cout << "\ntotal partitions " << parts << '\n';
    std::cout << "\ntotal nnz " << nnz_A << '\n';
    std::cout << "\ntotal number of cuts " << G.create_parts(A, parts) << '\n'
              << std::endl;
    G.create_parts(A, parts);
    const auto &p = G.P()(), part_start = G.part_start();
    ASSERT_EQ(part_start.back(), (int)A.nrows());
    ASSERT_EQ(part_start.size(), 1u + parts);
    for (int i = 0; i < parts; ++i) {
      const auto    start = part_start[i];
      Array<double> buf(part_start[i + 1] - start);
      const auto    B  = G.extract_block(A, i, buf);
      const auto    BD = convert2dense(B);
      for (int r = start; r < part_start[i + 1]; ++r) {
        const auto &R = AD[p[r]], L = BD[r - start];
        for (int c = start; c < part_start[i + 1]; ++c)
          EXPECT_EQ(L[c - start], R[p[c]]);
      }
    }
  } while (0);
  do {
    using ccs_t   = CCS<double, int>;
    const auto A  = gen_rand_sparse<ccs_t>(100, 100);
    const auto AD = convert2dense(A);
    std::cout << A.row_ind().size() << ' ' << A.col_start().size() << std::endl;
    graph_t   G;
    const int parts = i_rand();
    G.create_parts(A, parts);
    const auto &p = G.P()(), part_start = G.part_start();
    ASSERT_EQ(part_start.back(), (int)A.nrows());
    ASSERT_EQ(part_start.size(), 1u + parts);
    for (int i = 0; i < parts; ++i) {
      const auto    start = part_start[i];
      Array<double> buf(part_start[i + 1] - start);
      const auto    B  = G.extract_block(A, i, buf);
      const auto    BD = convert2dense(B);
      for (int r = start; r < part_start[i + 1]; ++r) {
        const auto &R = AD[p[r]], L = BD[r - start];
        for (int c = start; c < part_start[i + 1]; ++c)
          EXPECT_EQ(L[c - start], R[p[c]]);
      }
    }
  } while (0);
}

TEST(GRAPH, fortran) {
  do {
    using crs_t      = CRS<double, int, true>;
    const auto A     = gen_rand_sparse<crs_t>(100, 100);
    const auto nnz_A = A.nnz();
    const auto AD    = convert2dense(A);
    graph_t    G;
    const int  parts = i_rand();
    std::cout << "\ntotal partitions " << parts << '\n';
    std::cout << "\ntotal nnz " << nnz_A << '\n';
    std::cout << "\ntotal number of cuts " << G.create_parts(A, parts) << '\n'
              << std::endl;
    G.create_parts(A, parts);
    const auto &p = G.P()(), part_start = G.part_start();
    ASSERT_EQ(part_start.back(), (int)A.nrows());
    ASSERT_EQ(part_start.size(), 1u + parts);
    for (int i = 0; i < parts; ++i) {
      const auto    start = part_start[i];
      Array<double> buf(part_start[i + 1] - start);
      const auto    B  = G.extract_block(A, i, buf);
      const auto    BD = convert2dense(B);
      for (int r = start; r < part_start[i + 1]; ++r) {
        const auto &R = AD[p[r]], L = BD[r - start];
        for (int c = start; c < part_start[i + 1]; ++c)
          EXPECT_EQ(L[c - start], R[p[c]]);
      }
    }
  } while (0);
  do {
    using ccs_t   = CCS<double, int, true>;
    const auto A  = gen_rand_sparse<ccs_t>(100, 100);
    const auto AD = convert2dense(A);
    std::cout << A.row_ind().size() << ' ' << A.col_start().size() << std::endl;
    graph_t   G;
    const int parts = i_rand();
    G.create_parts(A, parts);
    const auto &p = G.P()(), part_start = G.part_start();
    ASSERT_EQ(part_start.back(), (int)A.nrows());
    ASSERT_EQ(part_start.size(), 1u + parts);
    for (int i = 0; i < parts; ++i) {
      const auto    start = part_start[i];
      Array<double> buf(part_start[i + 1] - start);
      const auto    B  = G.extract_block(A, i, buf);
      const auto    BD = convert2dense(B);
      for (int r = start; r < part_start[i + 1]; ++r) {
        const auto &R = AD[p[r]], L = BD[r - start];
        for (int c = start; c < part_start[i + 1]; ++c)
          EXPECT_EQ(L[c - start], R[p[c]]);
      }
    }
  } while (0);
}