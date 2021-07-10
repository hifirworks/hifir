///////////////////////////////////////////////////////////////////////////////
//                  This file is part of the HIFIR library                   //
///////////////////////////////////////////////////////////////////////////////

/*

  This file contains an example of using HIFIR with mixed-precision support.

  Author: Qiao Chen
  Level: Beginner

*/

#include "../demo_utils.hpp"
#include "hifir.hpp"

// all you need to do is to use a single-precision preconditioner, which
// supports double-precision triangular solves
using single_prec_t = hif::HIF<float, int>;

int main() {
  // read inputs
  system_t prob = get_input_data();

  // create HIF preconditioner, and factorize with default params
  auto M = single_prec_t();
  M.factorize(prob.A);

  // call multilevel triangular solve, which is the core in KSP solvers
  array_t x(prob.A.nrows());
  M.solve(prob.b, x);  // x = M^{-1}b

  return 0;
}
