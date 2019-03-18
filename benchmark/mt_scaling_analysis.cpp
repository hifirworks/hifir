//@HEADER
//----------------------------------------------------------------------------
//                Copyright (C) 2019 The PSMILU AUTHORS
//----------------------------------------------------------------------------
//@HEADER

// This is a template implementation of multi-threading GMRES with right
// preconditioner for benchmark purpose.

#include <fstream>
#include <iostream>
#include <string>

#include "GMRES_MT.hpp"
#include "psmilu_CompressedStorage.hpp"

using namespace psmilu;

typedef CRS<double, int>                                             matrix_t;
typedef Array<double>                                                array_t;
typedef bench::GMRES<double, bench::internal::IdentityPrec, array_t> solver_t;
typedef bench::GMRES_MT<double, bench::IdentityPrec_MT, array_t> solver_mt_t;

static const char *help =
    "\n"
    "Usage: ./mt_scaling_analysis [case dir [- [-v|--verbose]]] | [-h|--help]\n"
    "\n"
    "This program uses the identity preconditioner mainly for GMRES_MT sca-\n"
    "ling analysis. It follows the same input structure of the driver rout-\n"
    "ine, i.e. a directory containing the input matrix \"A.psmilu\" and rhs\n"
    "vector ASCII file \"b.txt\".\n"
    "\n"
    " case-dir\n"
    "\tcase directory containing input matrix and rhs\n"
    " -\n"
    "\tusing \"stdin\" to read solver parameters (later)\n"
    " -v|--verbose\n"
    "\tverbose iteration and warning process\n"
    " -h|--help\n"
    "\tshow this message and exit\n"
    "\n"
    "One can redirect/pipe standard input to control the GMRES parameters;\n"
    "the order is \"rtol\", \"restart\", \"maxit\", and \"threads\". Reg-\n"
    "arding \"threads\", assign negative values to disable MT and positive\n"
    "values to enforce using certain number of threads.\n"
    "\n"
    "In addition, one can define environment variables to control number of\n"
    "threads, either \"PSMILU_NUM_THREADS\" (higher priority) or the standard\n"
    "\"OMP_NUM_THREADS\".\n"
    "\n"
    "Examples:\n"
    "\n"
    "use 4 threads\n"
    " env OMP_NUM_THREADS=4 ./mt_scaling_analysis /my/test/case\n"
    "modify control parameters\n"
    " echo \"1e-8 50 1000 -1\"|./mt_scaling_analysis /my/test/case\n"
    "\n";

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cerr << "at least 1 parameter is needed (input case directory)\n";
    std::cerr << help;
    return 1;
  }
  std::string kase = argv[1];
  if (kase != "" && kase.back() != '/') kase += "/";
  const std::string mat_file = kase + "A.psmilu", rhs_file = kase + "b.txt";
  bool              use_stdin = false;
  bool              verbose   = false;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "-")
      use_stdin = true;
    else if (arg == "-v" || arg == "--verbose")
      verbose = true;
    else if (arg == "-h" || arg == "--help") {
      std::cout << help;
      return 0;
    }
  }
  solver_t    solver;
  solver_mt_t solver_mt;
  double      rtol;
  int         restart;
  std::size_t maxit;
  int         threads(0);
  if (use_stdin) {
    std::cin >> rtol >> restart >> maxit >> threads;
    solver.rtol       = rtol;
    solver.maxit      = maxit;
    solver.restart    = restart;
    solver_mt.rtol    = rtol;
    solver_mt.maxit   = maxit;
    solver_mt.restart = restart;
  }
  const auto A = matrix_t::from_native_bin(mat_file.c_str());
  array_t    b(A.nrows()), x1(b.size()), x2(b.size());
  do {
    std::ifstream f(rhs_file.c_str());
    if (!f.is_open()) {
      std::cerr << "cannot open rhs file " << rhs_file << '\n';
      return 1;
    }
    for (auto &v : b) f >> v;
    f.close();
  } while (false);

  int         flag;
  std::size_t iters;
  double      t;

  std::cout << "solve with MT solver...\n";
  std::tie(flag, iters, t) = solver_mt.solve(A, b, x1, verbose, threads);
  std::cout << "flag: " << flag << ", iters: " << iters << ", time: " << t
            << "s.\n";
  std::cout << "solve with serial solver...\n";
  std::tie(flag, iters, t) = solver.solve(A, b, x2, verbose);
  std::cout << "flag: " << flag << ", iters: " << iters << ", time: " << t
            << "s.\n";
  const auto beta2 = std::inner_product(b.cbegin(), b.cend(), b.cbegin(), 0.);
  double     diff(0);
  for (std::size_t i(0); i < b.size(); ++i)
    diff += (x1[i] - x2[i]) * (x1[i] - x2[i]);
  std::cout << "relative error between serial and MT solutions is "
            << std::sqrt(diff / beta2) << std::endl;
  return 0;
}
