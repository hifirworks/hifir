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
#include <numeric>

using namespace psmilu;

static const RandIntGen  i_rand(0, 50);
static const RandRealGen r_rand;

constexpr static double eps = 1e-12;

// this test script is to test the Crout by factorizing a matrix with
// LU w/o pivoting. We start with multiplying exact dense matrices L and U
// and convert is to sparse, then factorize it with Crout routines, to further
// simplify the process, we make the system diagonal dominant.

TEST(LU, c) {
  const int n = i_rand() + 2;
  std::cout << "Problem size is " << n << '\n';
  auto          L = create_mat<double>(n, n);
  auto          U = create_mat<double>(n, n);
  Array<double> d(n);
  // assign L, omit unit diagonal
  for (int i = 1; i < n; ++i)
    for (int j = 0; j < i; ++j) L.at(i).at(j) = r_rand();
  // assign U, omit unit diagonal, copy L'
  for (int i = 0; i < n - 1; ++i)
    for (int j = i + 1; j < n; ++j) U.at(i).at(j) = L[j][i];

  // build diagonal matrix
  for (int i = 0; i < n; ++i) d[i] = r_rand() + 2 * n;

  const double tol = *std::max_element(d.cbegin(), d.cend()) * eps;

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

  const int m = (i_rand() % n) + 1;

  auto d2 = extract_diag(A);
  // build L and U
  AugCRS<crs_t> U2(m, n);
  U2.reserve(n * m);
  AugCCS<ccs_t> L2(n, m);
  L2.reserve(n * m);
  U2.begin_assemble_rows();
  L2.begin_assemble_cols();

  // NOTE that within the crout steps, one should not see any of the
  // desctructors been called!
  std::cout << "begin crout update for leading block " << m << "\n\n";
  for (; (int)crout < m; ++crout) {
    std::cout << "enter crout step " << crout << '\n';
    std::cout << "\tupdating L_start...\n";
    crout.update_L_start(L2, L_start);
    std::cout << "\tupdating U_start...\n";
    crout.update_U_start<true>(U2, m, U_start);
    // compute l and u
    l.reset_counter();
    std::cout << "\tcomputing l_k...\n";
    crout.compute_l(A_ccs, p, crout, L2, L_start, d2, U2, l);
    std::cout << "\tl_k size: " << l.size() << '\n';
    ut.reset_counter();
    std::cout << "\tcomputing u_k\'...\n";
    crout.compute_ut<true>(s, A_crs, t, q, crout, m, L2, d2, U2, U_start, ut);
    // scale inv d
    EXPECT_FALSE(crout.scale_inv_diag(d, l))
        << "singular at step " << crout << " for l\n";
    EXPECT_FALSE(crout.scale_inv_diag(d, ut))
        << "singular at step " << crout << " for ut\n";
    std::cout << "\tu_k\' size: " << ut.size() << '\n';
    std::cout << "\tupdating diagonal matrix...\n";
    crout.update_B_diag<true>(l, ut, m, d2);
    // sort indices
    std::cout << "\tsorting indices...\n";
    l.sort_indices();
    ut.sort_indices();
    if (m != n) {
      ASSERT_LE(*ut.inds().cbegin(), m);
    }

    std::cout << "\tpushing back to L...\n";
    L2.push_back_col(crout, l.inds().cbegin(), l.inds().cbegin() + l.size(),
                     l.vals());
    std::cout << "\tpushing back to U...\n";
    U2.push_back_row(crout, l.inds().cbegin(), l.inds().cbegin() + l.size(),
                     l.vals(), ut.inds().cbegin(),
                     ut.inds().cbegin() + ut.size(), ut.vals());
  }

  U2.end_assemble_rows();
  L2.end_assemble_cols();

  std::cout << "\nfinished crout update\n";

  // comparing diagonals
  for (int i = 0; i < m; ++i)
    EXPECT_NEAR(d[i], d2[i], tol)
        << "diagonal " << i << " out of " << n << " mismatched\n";

  const auto mat_L = extract_leading_block(convert2dense(L2), m);
  const auto mat_U = extract_leading_block(convert2dense(U2), m);
  COMPARE_MATS_TOL(mat_L, L, tol);
  COMPARE_MATS_TOL(mat_U, U, tol);

  // because we start with symmetric system, the offset should preserve
  // the symmetry property
  {
    const auto mat_L = convert2dense(L2);
    const auto mat_U = convert2dense(U2);
    COMPARE_MATS_TOL(mat_L, L, tol);
    COMPARE_MATS_TOL(mat_U, U, tol);
  }
}
