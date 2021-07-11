///////////////////////////////////////////////////////////////////////////////
//                  This file is part of the HIFIR library                   //
///////////////////////////////////////////////////////////////////////////////

/*

  This file contains an example of using complex HIFIR.

  Author: Qiao Chen
  Level: Beginner

*/

#include "hifir.hpp"

// define our complex preconditioner and matrix
using complex_prec_t   = hif::HIF<std::complex<double>, int>;
using complex_matrix_t = hif::CRS<std::complex<double>, int>;

int main() {
  // read the testing complex sparse matrix
  auto A = complex_matrix_t::from_mm("demo_inputs/young1c.mtx");

  // create HIF preconditioner, and factorize with default params
  auto M = complex_prec_t();
  M.factorize(A);

  // then you can perform triangular solve by calling M.solve(...)

  return 0;
}
