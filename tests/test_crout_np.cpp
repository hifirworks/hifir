//@HEADER
//----------------------------------------------------------------------------
//                Copyright (C) 2019 The PSMILU AUTHORS
//----------------------------------------------------------------------------
//@HEADER

#include "common.hpp"
// line break to avoid sorting
#include "psmilu_Array.hpp"
#include "psmilu_AugmentedStorage.hpp"
#include "psmilu_CompressedStorage.hpp"
#include "psmilu_Crout.hpp"
#include "psmilu_PermMatrix.hpp"
#include "psmilu_SparseVec.hpp"

#include <gtest/gtest.h>

using namespace psmilu;

static const RandIntGen  i_rand(0, 50);
static const RandRealGen r_rand;

// this test script is to test the Crout by factorizing a matrix with
// LU w/o pivoting. We start with multiplying exact dense matrices L and U
// and convert is to sparse, then factorize it with Crout routines, to further
// simplify the process, we make the system diagonal dominant.

TEST(LU, c) {
  const int n = 40;  // i_rand() + 2;
  std::cout << "Problem size is " << n << '\n';
  auto                L = create_mat<double>(n, n);
  auto                U = create_mat<double>(n, n);
  std::vector<double> d(n);
  // assign L, omit unit diagonal
  for (int i = 1; i < n; ++i)
    for (int j = 0; j < i; ++j) L.at(i).at(j) = r_rand();
  // assign U, omit unit diagonal
  for (int i = 0; i < n - 1; ++i)
    for (int j = i + 1; j < n; ++j) U.at(i).at(j) = r_rand();

  // build diagonal matrix
  for (int i = 0; i < n; ++i) d[i] = r_rand() + 2 * n;

  // form a within local scope
  matrix<double> A;
  {
    // temp assign 1 to L
    for (int i = 0; i < n; ++i) L[i][i] = 1.0;
    decltype(U) U_tmp(U);
    for (int i = 0; i < n; ++i) U_tmp[i][i] = 1.0;
    scale_dense_left(U_tmp, d);
    A = dense_mm(L, U_tmp);
    // revert L back
    for (int i = 0; i < n; ++i) L[i][i] = 0.0;
  }

  using crs_t = CRS<double, int>;
  crs_t A_crs(n, n);
  A_crs.reserve(n * n);
  std::vector<int> inds(n);
  for (int i = 0; i < n; ++i) inds[i] = i;
  // construct sparse matrix
  A_crs.begin_assemble_rows();
  for (int i = 0; i < n; ++i) {
    A_crs.push_back_row(i, inds.cbegin(), inds.cend(), A[i]);
    ASSERT_EQ(A_crs.nnz_in_row(i), n);
  }
  A_crs.end_assemble_rows();

  // build ccs
  typedef crs_t::other_type ccs_t;
  ccs_t                     A_ccs(A_crs);

  SparseVector<double, int> l(n), ut(n);
  BiPermMatrix<int>         p(n), q(n);
  p.make_eye();
  q.make_eye();
  Array<double> s(n, 1.0), t(n, 1.0);  // dummy scaling diagonal matrices
  Array<int>    L_start(n), U_start(n);

  Crout crout;

  auto d2 = extract_diag(A);
  // build L and U
  AugCRS<crs_t> U2(n, n);
  U2.reserve(n * n);
  AugCCS<ccs_t> L2(n, n);
  L2.reserve(n * n);
  U2.begin_assemble_rows();
  L2.begin_assemble_cols();

  for (; (int)crout < n; ++crout) {
    std::cout << "enter crout step " << crout << '\n';
    // compute l and u
    l.reset_counter();
    std::cout << 1 << '\n';
    crout.compute_l(A_ccs, p, crout, L2, L_start, d2, U2, l);
    ut.reset_counter();
    std::cout << 2 << '\n';
    crout.compute_ut<false>(s, A_crs, t, q, crout, n, L2, d2, U2, U_start, ut);
    crout.update_B_diag<false>(l, ut, n, d2);
    // sort indices
    l.sort_indices();
    ut.sort_indices();
    L2.push_back_col(crout, l.inds().cbegin(), l.inds().cend(), l.vals());
    U2.push_back_row(crout, ut.inds().cbegin(), ut.inds().cend(), ut.vals());
    crout.update_L_start(L2, L_start);
    crout.update_U_start<false>(U2, n, U_start);
  }

  U2.end_assemble_rows();
  L2.end_assemble_cols();
}
