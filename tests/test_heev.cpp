///////////////////////////////////////////////////////////////////////////////
//                  This file is part of the HIFIR library                         //
///////////////////////////////////////////////////////////////////////////////

#include "common.hpp"
// line break to avoid sorting
#include "hif/Options.h"
#include "hif/ds/CompressedStorage.hpp"
#include "hif/ds/DenseMatrix.hpp"
#include "hif/small_scale/solver.hpp"

#include <gtest/gtest.h>

using namespace hif;

constexpr static double tol(1e-10);

TEST(SYEV, d) {
  // randomized Hermitian matrix from MATLAB of 5x5, in column order
  // generate rhs b then solve with backslash and save in x_ref
  const static double
      a_real[25] = {2.6394658521267012e-01, 1.1768305041478915e+00,
                    1.1251245696925065e+00, 1.1222174873648105e+00,
                    2.4329069868442554e-01, 1.1768305041478915e+00,
                    7.0631714244414212e-01, 1.4703095151544110e+00,
                    3.1172424325932824e-01, 4.1150839814814433e-01,
                    1.1251245696925065e+00, 1.4703095151544110e+00,
                    1.4634447713173406e+00, 1.3924387702104628e+00,
                    1.0765422671606353e+00, 1.1222174873648105e+00,
                    3.1172424325932824e-01, 1.3924387702104628e+00,
                    3.7791003006508905e-01, 1.4670028685166918e+00,
                    2.4329069868442554e-01, 4.1150839814814433e-01,
                    1.0765422671606353e+00, 1.4670028685166918e+00,
                    1.6225153773157053e-01},
      a_imag[25] = {0.0000000000000000e+00,  4.6936320659184494e-01,
                    -1.5752649779051930e-01, -5.0314297341896774e-01,
                    2.3904145669677779e-01,  -4.6936320659184494e-01,
                    0.0000000000000000e+00,  1.3216218151184123e-01,
                    -5.8315103170721749e-02, 4.9358508660396239e-01,
                    1.5752649779051930e-01,  -1.3216218151184123e-01,
                    0.0000000000000000e+00,  -1.7330754098967338e-02,
                    -1.2019624494070735e-01, 5.0314297341896774e-01,
                    5.8315103170721749e-02,  1.7330754098967338e-02,
                    0.0000000000000000e+00,  3.5655654431985828e-01,
                    -2.3904145669677779e-01, -4.9358508660396239e-01,
                    1.2019624494070735e-01,  -3.5655654431985828e-01,
                    0.0000000000000000e+00},
      b_real[5]  = {2.6187118387071606e-01, 3.3535683996279653e-01,
                   6.7972795137733799e-01, 1.3655313735536967e-01,
                   7.2122749858174018e-01},
      b_imag[5]  = {1.0676186160724144e-01, 6.5375734866855961e-01,
                   4.9417393663927012e-01, 7.7905172323127514e-01,
                   7.1503707840069408e-01},
      x_ref_real[5] =
          {
              -1.3450929260359141e+01, 1.0769380429574321e+01,
              -3.9642298017388413e+01, -2.3973134786403683e+00,
              6.6686760028733630e+01,
          },
      x_ref_imag[5] = {-6.1175858579830560e+01, 1.6903301578437244e+01,
                       7.7121370999551843e+01, -5.4368433541740124e+01,
                       -3.9039206062656948e+00};
  using value_t     = std::complex<double>;
  Array<value_t> x(5);
  for (int i = 0; i < 5; ++i) {
    x[i].real(b_real[i]);
    x[i].imag(b_imag[i]);
  }
  using ccs_t = CCS<value_t, int>;
  ccs_t                crs(5, 5);
  std::vector<value_t> buf(5);
  const static int     inds[10] = {0, 1, 2, 3, 4};
  const double *       v_real = a_real, *v_imag = a_imag;
  crs.begin_assemble_cols();
  for (int i = 0; i < 5; ++i) {
    for (int i = 0; i < 5; ++i) {
      buf[i].real(v_real[i]);
      buf[i].imag(v_imag[i]);
    }
    crs.push_back_col(i, inds, inds + 5, buf);
    v_real += 5;
    v_imag += 5;
  }
  crs.end_assemble_cols();

  SYEIG<value_t> eig;
  eig.set_matrix(crs);
  eig.factorize(get_default_options());
  ASSERT_TRUE(eig.full_rank()) << "should be full rank!\n";
  eig.solve(x);
  for (int i = 0; i < 5; ++i) {
    EXPECT_NEAR(x[i].real(), x_ref_real[i], tol)
        << i << " real entry doesn\'t agree with reference solution\n";
    EXPECT_NEAR(x[i].imag(), x_ref_imag[i], tol)
        << i << " real entry doesn\'t agree with reference solution\n";
  }
}