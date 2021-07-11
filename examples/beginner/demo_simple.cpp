///////////////////////////////////////////////////////////////////////////////
//                  This file is part of the HIFIR library                   //
///////////////////////////////////////////////////////////////////////////////

/*

  This file contains an example of the simplest use of HIF preconditioner, i.e.
  use its multilevel triangular solve

  Author: Qiao Chen
  Level: Beginner

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
  array_t x(prob.A.nrows());
  M.solve(prob.b, x);  // x = M^{-1}b

  return 0;
}
