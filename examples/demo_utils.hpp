///////////////////////////////////////////////////////////////////////////////
//                  This file is part of the HIFIR library                   //
///////////////////////////////////////////////////////////////////////////////

/// \headerfile demo_utils.hpp "../demo_utils.hpp"

#pragma once

#include "hifir.hpp"

using matrix_t = hif::CRS<double, int>;
using array_t  = hif::Array<double>;

// a simple structure represents a linear system
typedef struct {
  matrix_t A;
  array_t  b;
} system_t;

// load input data
inline system_t get_input_data() {
  return (system_t){.A = matrix_t::from_mm("demo_inputs/A.mm"),
                    .b = array_t::from_mm("demo_inputs/b.mm")};
}

// compute relative error
inline double compute_error(const array_t &v1, const array_t &v2) {
  const auto n = v1.size();
  array_t    work(n);
  for (auto i(0u); i < n; ++i) work[i] = v1[i] - v2[i];
  return hif::norm2(work) / hif::norm2(v1);
}

// compute relative residual
inline double compute_relres(const matrix_t &A, const array_t &b,
                             const array_t &x) {
  const auto n = b.size();
  array_t    work(n);
  A.multiply(x, work);
  for (auto i(0u); i < n; ++i) work[i] = b[i] - work[i];
  return hif::norm2(work) / hif::norm2(b);
}
