///////////////////////////////////////////////////////////////////////////////
//                  This file is part of the HIFIR library                   //
///////////////////////////////////////////////////////////////////////////////

/*

  This file contains an example of how to deal with verbose logging in HIFIR

  Author: Qiao Chen
  Level: Beginner

*/

#include "../demo_utils.hpp"
#include "hifir.hpp"

using prec_t = hif::HIF<double, int>;

int main(int argc, char *argv[]) {
  // read inputs
  system_t prob = parse_cmd4input(argc, argv);

  // create HIF preconditioner, and factorize with default params
  auto M = prec_t();
  // get parameters
  hif::Params params = hif::DEFAULT_PARAMS;
  // enable verbose for factorization, which can be very messy and slow
  hif::enable_verbose(hif::VERBOSE_FAC, params);
  // similarly, we can enable, e.g., hif::VERBOSE_PRE for detailed information
  // regarding precprocessing
  // But most commonly, we want to disable verbose logging
  params.verbose = hif::VERBOSE_NONE;  // no logging
  M.factorize(prob.A, params);

  return 0;
}
