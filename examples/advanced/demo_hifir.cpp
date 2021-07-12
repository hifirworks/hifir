///////////////////////////////////////////////////////////////////////////////
//                  This file is part of the HIFIR library                   //
///////////////////////////////////////////////////////////////////////////////

/*

  This file contains an example of using HIFIR operation, i.e., HIF with
  iterative refinement.

  Author: Qiao Chen
  Level: Advanced

*/

#include "../demo_utils.hpp"
#include "hifir.hpp"

using prec_t = hif::HIF<double, int>;

int main(int argc, char *argv[]) {
  // read inputs
  system_t prob = parse_cmd4input(argc, argv);

  // create HIF preconditioner, and factorize with default params
  auto M = prec_t();
  // tune parameters if necessary, see beginner/demo_params
  M.factorize(prob.A);

  // call multilevel triangular solve, which is the core in KSP solvers
  array_t x(prob.A.nrows());

  // Method I: calling with fixed number of refinement (e.g., four iterations)
  M.hifir(prob.A, prob.b, 4, x);

  // Method II: calling with residual bounds and upper bound of refinement
  // steps. The residual bound is given by a length-two double array, of which
  // the first entry (beta_L) is the lower bound and the second entry (beta_U)
  // is the upper bound (e.g., 16 maximum iterations).
  const double betas[2] = {1e-6, 1.0};
  const auto   info     = M.hifir(prob.A, prob.b, 16, betas, x);
  if (info.second == 0) {
    // converged
    hif_info("\nIterative refinement converged with %zd iterations,",
             info.first);
    hif_info("and relative residual norm(b-Ax)/norm(b) is %g",
             compute_relres(prob.A, prob.b, x));
  }
  return 0;
}
