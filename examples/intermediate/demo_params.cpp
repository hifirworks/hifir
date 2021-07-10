///////////////////////////////////////////////////////////////////////////////
//                  This file is part of the HIFIR library                   //
///////////////////////////////////////////////////////////////////////////////

/*

  This file contains an example of how to customize control parameters for
  HIFIR, which is needed for optimization of performance.

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
  // get parameters
  hif::Params params = hif::DEFAULT_PARAMS;
  // For PDE-based problem, we can typically relax the parameters to gain
  // better performance (in terms of total runtimes). Note that the testing
  // matrix comes for a Taylor-Hood-element-discretized Stokes equations,
  // which is symmetric saddle-point system. For this kind of system, we can
  // use droptol 1e-2, fill factor 3, and inverse-norm threshold 5
  params.tau_L = params.tau_U = 1e-2;     // droptol
  params.alpha_L = params.alpha_U = 3.0;  // fill factors
  params.kappa = params.kappa_d = 5.0;    // inverse-norm thres
  M.factorize(prob.A, params);

  return 0;
}
