///////////////////////////////////////////////////////////////////////////////
//                  This file is part of the HIFIR library                   //
///////////////////////////////////////////////////////////////////////////////

/*

  This file contains an example of how to apply multilevel matrix-vector
  multiplication using a HIF preconditioner. Note that this is considered
  to be an advanced feature in HIF, because it may be only used in a few
  of advanced applications, such null-space computation.

  Author: Qiao Chen
  Level: Intermediate

*/

#include "../demo_utils.hpp"
#include "hifir.hpp"

using prec_t = hif::HIF<double, int>;

int main() {
  // read inputs
  system_t prob = get_input_data();

  // create HIF preconditioner, and factorize with default params
  auto M = prec_t();
  M.factorize(prob.A);

  // call multilevel triangular solve, which is the core in KSP solvers
  array_t x(prob.A.nrows()), b2(x.size());
  M.mmultiply(prob.b, x);  // x = Mb
  // or x = M^{T}b
  // M.mmultiply(prob.b, x, true);
  // NOTE, M^{-1}x should reproduce b up to machine precision for nonsingular M
  M.solve(x, b2);
  hif_info("relative error is %g", compute_error(prob.b, b2));

  return 0;
}
