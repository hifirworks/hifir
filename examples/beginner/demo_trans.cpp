///////////////////////////////////////////////////////////////////////////////
//                  This file is part of the HIFIR library                   //
///////////////////////////////////////////////////////////////////////////////

/*

  This file contains an example of how to use transpose/Hermitian multilevel
  triangular solve

  Author: Qiao Chen
  Level: Beginner

*/

#include "../demo_utils.hpp"
#include "hifir.hpp"

using prec_t            = hif::HIF<double, int>;
static const bool trans = true;

int main() {
  // read inputs
  system_t prob = get_input_data();

  // create HIF preconditioner, and factorize with default params
  auto M = prec_t();
  M.factorize(prob.A);

  // call multilevel triangular solve, which is the core in KSP solvers
  array_t x(prob.A.nrows());
  M.solve(prob.b, x, trans);  // x = M^{-T}b

  return 0;
}
