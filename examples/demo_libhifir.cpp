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

// This file contains an example of using HIF preconditioner with its multilevel
// triangular solve and matrix-vector multiplication interfaces.
// Author: Qiao Chen

#include <fstream>
#include <string>
#include <utility>

#include "hifir.hpp"

#include "libhifir.h"

using prec_t   = hif::HIF<double, int>;
using matrix_t = prec_t::crs_type;
using array_t  = hif::Array<double>;

inline std::pair<matrix_t, array_t> get_input_data();
inline double compute_error(const array_t &v1, const array_t &v2,
                            array_t &work);

int main() {
  // read inputs
  matrix_t A;
  array_t  b;
  std::tie(A, b)    = get_input_data();
  const long long n = b.size();

  // create HIF preconditioner
  char M[LIBHIFIR_PREC_LEN];
  int  info = libhifir_create(nullptr, nullptr, nullptr, M);
  hif_error_if(info != LIBHIFIR_SUCCESS, "create prec failed");

  // create parameters
  double params[LIBHIFIR_NUMBER_PARAMS];
  libhifir_setup_params(params);
#ifdef HIF_VERBOSE_NULL
  params[LIBHIFIR_VERBOSE] = LIBHIFIR_VERBOSE_NULL;
#endif

  // compute factorization
  info = libhifir_factorize(M, &n, A.row_start().data(), A.col_ind().data(),
                            A.vals().data(), params);
  hif_error_if(info != LIBHIFIR_SUCCESS, "factorization failed");

  // perform triangular solve and matrix-vector multiplication
  array_t x(b.size()), b2(b.size());
  info = libhifir_solve(M, &n, b.data(), nullptr, nullptr, x.data());
  hif_error_if(info != LIBHIFIR_SUCCESS, "failed to solve");
  info = libhifir_mmultiply(M, &n, x.data(), nullptr, nullptr, b2.data());
  hif_error_if(info != LIBHIFIR_SUCCESS, "failed to multiply");
  // compute error
  const double err1 = compute_error(b, b2, x);
  hif_info("\nrelative error no-tran is %g...\n", err1);
  hif_warning_if(err1 >= 1e-10, "error is too large!");

  // perform triangular solve and matrix-vector multiplication transpose
  const int trans = 1;
  info            = libhifir_solve(M, &n, b.data(), &trans, nullptr, x.data());
  hif_error_if(info != LIBHIFIR_SUCCESS, "failed to solve");
  info = libhifir_mmultiply(M, &n, x.data(), &trans, nullptr, b2.data());
  hif_error_if(info != LIBHIFIR_SUCCESS, "failed to multiply");
  // compute error
  const double err2 = compute_error(b, b2, x);
  hif_info("\nrelative error tran is %g...\n", err2);
  hif_warning_if(err2 >= 1e-10, "error is too large!");

  // destroy the preconditioner
  info = libhifir_destroy(M);
  hif_error_if(info != LIBHIFIR_SUCCESS, "destroy failed");

  return 0;
}

inline std::pair<matrix_t, array_t> get_input_data() {
  matrix_t::size_type m;
  // read A
  auto          A = matrix_t::from_bin("demo_inputs/A.data", &m);
  array_t       b(A.nrows());
  std::ifstream f("demo_inputs/b.txt");
  hif_error_if(!f.is_open(), "unable to open \"demo_inputs/b.txt\"");
  for (auto &v : b) f >> v;
  f.close();
  return std::make_pair(A, b);
}

inline double compute_error(const array_t &v1, const array_t &v2,
                            array_t &work) {
  const auto n = v1.size();
  for (auto i(0u); i < n; ++i) work[i] = v1[i] - v2[i];
  return hif::norm2(work) / hif::norm2(v1);
}
