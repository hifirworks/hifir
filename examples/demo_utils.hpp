///////////////////////////////////////////////////////////////////////////////
//                  This file is part of the HIFIR library                   //
///////////////////////////////////////////////////////////////////////////////

/*

    Copyright (C) 2021 NumGeom Group at Stony Brook University

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

*/

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
