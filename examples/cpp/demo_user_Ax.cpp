//===========================================================================//
//                  This file is part of the HIFIR library                         //
//===========================================================================//

// Demo with user callback for computing A*x
// authors: Qiao,
//
// Copyright (C) 2021 NumGeom Group at Stony Brook University
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

#include <fstream>

#ifdef HIF_DEMO_USE_MKL
#  include <mkl.h>
#endif

#include "get_inputs.hpp"
#include "parse_options.hpp"

using namespace hif;
using std::string;

using prec_t      = DefaultHIF;  // use default C CRS with double and int
using crs_t       = prec_t::crs_type;
using array_t     = prec_t::array_type;
using ksp_factory = ksp::KSPFactory<prec_t>;
using solver_t    = ksp_factory::abc_solver;
using func_t      = solver_t::func_type;

// create solver
static std::shared_ptr<solver_t> create_ksp(const int               choice,
                                            std::shared_ptr<prec_t> M) {
  switch (choice) {
    case 0:
      return std::make_shared<ksp_factory::fgmres>(M);
    case 1:
      return std::make_shared<ksp_factory::fqmrcgstab>(M);
    case 2:
      return std::make_shared<ksp_factory::tgmresr>(M);
    case 3:
      return std::make_shared<ksp_factory::fbicgstab>(M);
    default:
      hif_error("unknown ksp solver choice %d", choice);
  }
  return nullptr;
}

int main(int argc, char *argv[]) {
  Options opts;
  int     restart;
  double  rtol;
  bool    symm;
  int     kernel;
  int     ksp;
  bool    rhs_a1;
  // parse arguments
  std::tie(opts, restart, rtol, symm, kernel, ksp, rhs_a1) =
      parse_args(argc, argv);
  if (opts.verbose == VERBOSE_NONE) warn_flag(0);
  crs_t   A;
  array_t b;
  // read input data
  std::tie(A, b) = get_inputs<crs_t, array_t>(std::string(argv[1]), rhs_a1);
  std::cout << "rtol=" << rtol << ", restart=" << restart
            << "\nNumberOfUnknowns=" << A.nrows() << ", nnz(A)=" << A.nnz()
            << "\n"
            << "symmetric=" << symm << "\n\n"
            << opt_repr(opts) << std::endl;

  array_t x(b.size());  // solution
  crs_t   A2(A, true);
  std::cout << "eliminated " << A2.eliminate(1e-15) << " small entries\n";

  DefaultTimer timer;

  // build preconditioner
  timer.start();
  std::shared_ptr<prec_t> _M(new prec_t());
  auto &                  M = *_M;
  M.factorize(A2, opts, 0u);
  timer.finish();
  hif_info(
      "\nMLILU done!\n"
      "\tfill-in: %.2f%%\n"
      "\tfill-in (E and F): %.2f%%\n"
      "\tnnz(E+F)/nnz(M)=%.2f%%\n"
      "\tlevels: %zd\n"
      "\tspace-dropping ratio=%.2f%%\n"
      "\ttime: %.4gs\n",
      100.0 * M.nnz() / A.nnz(), 100.0 * M.nnz_ef() / A.nnz(),
      100.0 * M.nnz_ef() / M.nnz(), M.levels(), 100.0 * M.stats(5) / M.stats(4),
      timer.time());

#if 0
  timer.start();
  M.optimize();
  timer.finish();
  hif_info(
      "\nOptimization preconditioner done!\n"
      "\ttime: %.4gs\n",
      timer.time());
#endif

  // solve
  // wrap user callback, the interfaces are
  //   opaque pointer x: input array
  //   n: length of the array
  //   xdtype: data type for x, 'd' for double, 's' for float
  //   opaque pointer y: output array
  //   ydtype: data type for y
  // User needs to ensure data types are correct, and then cast x and y to
  // proper target types.
  const func_t AA = [&](const void *x, const std::size_t n, const char xdtype,
                        void *y, const char ydtype, const bool) {
    hif_error_if(n != A.nrows(), "mismatched sizes");  // sizes must match
    hif_error_if(xdtype != 'd', "input array must be double");
    hif_error_if(ydtype != 'd', "output array must be double");
    A.multiply_nt_low(reinterpret_cast<const double *>(x),
                reinterpret_cast<double *>(y));
  };
  timer.start();
  std::shared_ptr<solver_t> solver(create_ksp(ksp, _M));
  solver->set_rtol(rtol);
  if (solver->is_arnoldi()) solver->set_restart_or_cycle(restart);
  int                 flag;
  solver_t::size_type iters;
  std::tie(flag, iters) = solver->solve(AA, b, x, kernel, false, opts.verbose);
  timer.finish();
  const auto   normb = norm2(b);
  const double rs    = solver->get_resids().back() / normb;
  double       act_rs;
  do {
    array_t r(b.size());
    A.multiply(x, r);
    for (array_t::size_type i(0); i < b.size(); ++i) r[i] -= b[i];
    act_rs = norm2(r) / normb;
  } while (false);
  hif_info(
      "\n%s(%.1e) done!\n"
      "\tflag: %s\n"
      "\titers: %zd\n"
      "\tres: %.4g\n"
      "\tact-res: %.4g\n"
      "\ttime: %.4gs\n",
      solver->repr(), rtol, ksp::flag_repr(solver->repr(), flag).c_str(), iters,
      rs, act_rs, timer.time());
  return flag;
}
