#ifdef PSMILU_UNIT_TESTING
#  undef PSMILU_UNIT_TESTING
#endif
#ifdef PSMILU_MEMORY_DEBUG
#  undef PSMILU_MEMORY_DEBUG
#endif

#include <cstdio>
#include <iostream>

extern "C" {
#include <hsl_mc64d.h>
}

#include "GMRES.hpp"
#include "PSMILU.hpp"

using namespace psmilu;

using crs_t    = C_DefaultBuilder::crs_type;
using array_t  = C_DefaultBuilder::array_type;
using solver_t = bench::GMRES<double, C_DefaultBuilder, array_t>;

const static char *help =
    "\n"
    "Usage: ./driver [case-dir | [-h|--help]]\n\n"
    " This is the driver problem for benchmark purpose; the *case-dir* is an\n"
    " input directory that contains 2 files: 1) A.psmilu, the matrix file of\n"
    " native PSMILU format and 2) b.txt, an ASCII file storing the rhs.\n\n"
    " In addition, if a ref_x.txt file exists, then the driver routine will\n"
    " load it and treat it as the reference solution and compute the relative\n"
    " error.\n\n"
    " The driver script will compute both symmetric and asymmetric precs and\n"
    " use GMRES(30) solve the problems with both 1e-6 and 1e-12 rtol thres.\n"
    "\n"
    " Regarding PSMILU parameters, they are passed in through stdin, you can\n"
    " either read from pipe, redirect from file or do standard keyboard. The\n"
    " order of parameters is the same as how they get defined in the struct.\n"
    "\n"
    " case-dir:\n"
    "\tinput case directory\n"
    " -h | --help:\n"
    "\tshow this help message and exits\n\n"
    "Examples:\n\n"
    " ./driver my_test < ./parameters.cfg\n"
    " echo \"0.01 0.01 10.0 20.0 4 4 0.25 1.0 2.0 -1 1\"|./driver my_test\n"
    " ./driver my_test # then type keyboard inputs\n\n"
    " On successful exit, the problem will produce four solutions files that\n"
    " store the results from each of the four cases. The solutions are\n"
    " stored in double precision with scientific notation.\n\n";

int main(int argc, char *argv[]) {
  std::string kase("");
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "-h" || arg == "--help") {
      std::cout << help;
      return 0;
    }
    if (kase != "") kase = arg;
  }
  // we don't allow use PWD as test root!
  if (kase == "") {
    std::cerr << "Not enough input(s), see help!\n" << help;
    return 1;
  }
  if (kase.back() != '/') kase += "/";
  const std::string A_file = kase + "A.psmilu", b_file = kase + "b.txt",
                    ref_x_file = kase + "ref_x.txt",
                    sol_file1  = kase + "symm_sol_1e-12.txt",
                    sol_file2  = kase + "symm_sol_1e-6.txt",
                    sol_file3  = kase + "asymm_sol_1e-12.txt",
                    sol_file4  = kase + "asymm_sol_1e-6.txt";
  bool        have_ref         = true;
  std::size_t m;
  const crs_t A = crs_t::from_native_bin(A_file.c_str(), &m);
  array_t     b(A.nrows()), x(b.size()), x_ref;
  {
    std::ifstream f(b_file.c_str());
    if (!f.is_open()) {
      std::cerr << "cannot open file " << b_file << std::endl;
      return 1;
    }
    for (auto &v : b) f >> v;
    f.close();
  }
  {
    std::ifstream f(ref_x_file.c_str());
    if (!f.is_open()) {
      std::cerr << "no reference solution found for case " << kase << std::endl;
      have_ref = false;
    }
    if (have_ref) {
      x_ref.resize(b.size());
      for (auto &v : x_ref) f >> v;
      f.close();
    }
  }

  auto opts = get_default_options();
  std::cin >> opts;

  std::cout << "doing PS analysis with leading block size " << m << '\n';

  solver_t solver_symm;
  solver_symm.M.compute(A, m, opts);

  solver_symm.rtol = 1e-12;
  std::cout << "solve with 1e-12 rtol...\n\n";
  solver_symm.solve(A, b, x);
  {
    std::ofstream f(sol_file1.c_str());
    if (!f.is_open()) {
      std::cerr << "cannot open file " << sol_file1 << std::endl;
      return 1;
    }
    f.precision(16);
    f.setf(std::ios_base::scientific);
    for (const auto v : x) f << v << '\n';
    f.close();
  }
  if (have_ref) {
    double vv(0), bot(0);
    for (int i = 0; i < (int)x.size(); ++i) {
      bot += x_ref[i] * x_ref[i];
      const double diff = x[i] - x_ref[i];
      vv += diff * diff;
    }
    std::printf("%e\n", std::sqrt(vv / bot));
  }
  solver_symm.rtol = 1e-6;
  std::cout << "solve with 1e-6 rtol...\n\n";
  solver_symm.solve(A, b, x);
  {
    std::ofstream f(sol_file2.c_str());
    if (!f.is_open()) {
      std::cerr << "cannot open file " << sol_file2 << std::endl;
      return 1;
    }
    f.precision(16);
    f.setf(std::ios_base::scientific);
    for (const auto v : x) f << v << '\n';
    f.close();
  }
  if (have_ref) {
    double vv(0), bot(0);
    for (int i = 0; i < (int)x.size(); ++i) {
      bot += x_ref[i] * x_ref[i];
      const double diff = x[i] - x_ref[i];
      vv += diff * diff;
    }
    std::printf("%e\n", std::sqrt(vv / bot));
  }

  std::cout << "doing general (asymmetric) analysis\n";

  solver_t solver_asymm;

  solver_asymm.M.compute(A, 0u, opts);

  solver_asymm.rtol = 1e-12;
  std::cout << "solve with 1e-12 rtol...\n\n";
  solver_asymm.solve(A, b, x);
  {
    std::ofstream f(sol_file3.c_str());
    if (!f.is_open()) {
      std::cerr << "cannot open file " << sol_file3 << std::endl;
      return 1;
    }
    f.precision(16);
    f.setf(std::ios_base::scientific);
    for (const auto v : x) f << v << '\n';
    f.close();
  }
  if (have_ref) {
    double vv(0), bot(0);
    for (int i = 0; i < (int)x.size(); ++i) {
      bot += x_ref[i] * x_ref[i];
      const double diff = x[i] - x_ref[i];
      vv += diff * diff;
    }
    std::printf("%e\n", std::sqrt(vv / bot));
  }
  solver_asymm.rtol = 1e-6;
  std::cout << "solve with 1e-6 rtol...\n\n";
  solver_asymm.solve(A, b, x);
  {
    std::ofstream f(sol_file4.c_str());
    if (!f.is_open()) {
      std::cerr << "cannot open file " << sol_file4 << std::endl;
      return 1;
    }
    f.precision(16);
    f.setf(std::ios_base::scientific);
    for (const auto v : x) f << v << '\n';
    f.close();
  }
  if (have_ref) {
    double vv(0), bot(0);
    for (int i = 0; i < (int)x.size(); ++i) {
      bot += x_ref[i] * x_ref[i];
      const double diff = x[i] - x_ref[i];
      vv += diff * diff;
    }
    std::printf("%e\n", std::sqrt(vv / bot));
  }
}
