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

#include "FGMRES.hpp"
#include "PSMILU.hpp"

using namespace psmilu;

using prec_t   = C_Default_PSMILU;
using crs_t    = prec_t::crs_type;
using array_t  = prec_t::array_type;
using solver_t = FGMRES<prec_t>;

int main(int argc, char *argv[]) {
  std::string kase("");
  warn_flag(1);
  bool   symm = false;
  double rtol = 1.5e-8;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "-s") {
      symm = true;
    } else if (arg == "-t") {
      if (i + 1 >= argc) {
        std::cerr << "missing tolerance..\n";
        return 1;
      }
      rtol = std::atof(argv[i + 1]);
      ++i;
    } else if (kase == "") {
      kase = arg;
    }
  }
  // we don't allow use PWD as test root!
  if (kase == "") {
    std::cerr << "Not enough input(s)\n";
    return 1;
  }
  if (kase.back() != '/') kase += "/";
  const std::string A_file = kase + "A.psmilu", b_file = kase + "b.txt",
                    ref_x_file = kase + "ref_x.txt",
                    sol_file1  = kase + "sol.txt";
  bool        have_ref         = true;
  std::size_t m;
  const crs_t A = crs_t::from_native_bin(A_file.c_str(), &m);

  array_t b(A.nrows()), x(b.size()), x_ref;
  do {
    std::ifstream f(b_file.c_str());
    if (!f.is_open()) {
      std::cerr << "cannot open file " << b_file << std::endl;
      return 1;
    }
    for (auto &v : b) f >> v;
    f.close();
  } while (0);
  do {
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
  } while (0);

  auto opts = get_default_options();
  std::cin >> opts;

  std::cout << "\nSolve with " << (symm ? "symmetric" : "asymmetric")
            << " setting with tolerance " << rtol << ".\n\n";

  prec_t M;
  M.deferred_factorize(A, symm ? A.nrows() : std::size_t(0), opts);
  do {
    M.prec(0).L_B.write_native_bin("L.psmilu");
    M.prec(0).U_B.write_native_bin("U.psmilu");
    do {
      std::ofstream f("d.txt");
      f.precision(16);
      f.setf(std::ios_base::scientific);
      for (const auto v : M.prec(0).d_B) f << v << '\n';
      f.close();
    } while (0);
  } while (0);

  solver_t solver(M);

  solver.rtol = rtol;
  solver.solve_pre(A, b, x);
  if (!have_ref) {
    do {
      std::ofstream f(sol_file1.c_str());
      if (!f.is_open()) {
        std::cerr << "cannot open file " << sol_file1 << std::endl;
        return 1;
      }
      f.precision(16);
      f.setf(std::ios_base::scientific);
      for (const auto v : x) f << v << '\n';
      f.close();
    } while (0);
  } else {
    double vv(0), bot(0);
    for (int i = 0; i < (int)x.size(); ++i) {
      bot += x_ref[i] * x_ref[i];
      const double diff = x[i] - x_ref[i];
      vv += diff * diff;
    }
    std::printf("%e\n", std::sqrt(vv / bot));
  }
}
