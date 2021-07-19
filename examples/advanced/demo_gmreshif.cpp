///////////////////////////////////////////////////////////////////////////////
//                  This file is part of the HIFIR library                   //
///////////////////////////////////////////////////////////////////////////////

/*

  This file contains an example of using HIF as right-preconditioner for
  GMRES(m). The example uses the testing system under "demo_inputs." The users
  can use their systems by (1) calling hif::wrap_const_{crs,ccs} and
  hif::wrap_const_array to directly wrap their systems or (2) loading data in
  Matrix Market file format, e.g.,

      auto A = hif::CRS<double,int>::from_mm("/path/to/sparse-mm-file")
      auto b = hif::Array<double>::from_mm("/path/to/dense-mm-file").

  Note that our GMRES implementation also works for complex arithmetic if
  matrix_t, array_t, and prec_t are complex values (with some minor
  modifications). See examples/intermediate/demo_complex.cpp.

  Author: Qiao Chen
  Level: Advanced

*/

#include <cstdlib>
#include <iostream>
#include <tuple>

#include "../demo_utils.hpp"
#include "hifir.hpp"

using prec_t = hif::HIF<double, int>;

#define SUCCESS 0
#define STAGNATED 1
#define DIVERGED 2

// Right-preconditioned GMRES with HIF preconditioner for consistent system
// using MGS for Arnoldi
//
// A, b. M: System (A, b) with HIF preconditioner (M)
// restart, rtol, maxit: (30, 1e-6, 500) by default
// return is the solution, i.e., $x \approx A^{-1}b$
std::tuple<array_t, int, int> gmres_hif(const matrix_t &A, const array_t &b,
                                        const prec_t &M, const int restart = 30,
                                        const double rtol    = 1e-6,
                                        const int    maxit   = 500,
                                        const int    verbose = 1);

// parse command-line arguments for system, restart, rtol, maxit, verbose, and
// robust parameter flag
std::tuple<system_t, int, double, int, int, bool> parse_args(int   argc,
                                                             char *argv[]);

int main(int argc, char *argv[]) {
  // parse options for gmres
  int      restart, maxit, verbose;
  double   rtol;
  system_t prob;
  bool     robust;
  std::tie(prob, restart, rtol, maxit, verbose, robust) =
      parse_args(argc, argv);

  // get timer
  hif::DefaultTimer timer;

  // create HIF preconditioner, and factorize with default params
  auto M      = prec_t();
  auto params = hif::DEFAULT_PARAMS;
  // The following parameters are essential to a HIF preconditioner, namely
  // droptol, fill factor, and inverse-norm threshold. Note that the default
  // settings are for robustness. The following parameters are optimized for
  // well-posed PDE systems, which are typically (nearly) pattern symmetric.
  // If you have very ill-conditioned or pattern asymmetric system, then please
  // use try robust parameters if the following setting fails.
  if (!robust) {
    params.tau_L = params.tau_U = 1e-2;     // droptol
    params.alpha_L = params.alpha_U = 3.0;  // fill factors
    params.kappa = params.kappa_d = 5.0;    // inverse-norm thres
  }
  if (verbose < 2) params.verbose = hif::VERBOSE_NONE;
  if (verbose) {
    hif_info("droptols (tau_L/tau_U) are %g/%g", params.tau_L, params.tau_U);
    hif_info("fill factors (alpha_L/alpha_U) are %g/%g", params.alpha_L,
             params.alpha_U);
    hif_info("inverse-norm thres (kappa/kappa_D) are %g/%g\n", params.kappa,
             params.kappa_d);
  }
  timer.start();
  M.factorize(prob.A, params);
  timer.finish();

  if (verbose) {
    hif_info("HIF(lvls=%zd) finished in %g seconds with nnz ratio %.2f%%.\n",
             M.levels(), timer.time(), 100.0 * M.nnz() / prob.A.nnz());
    // call HIF-preconditioned GMRES
    hif_info("Invoke HIF-preconditioned GMRES(%d) with rtol=%g and maxit=%d",
             restart, rtol, maxit);
  }

  array_t x;
  int     flag, iters;
  timer.start();
  std::tie(x, flag, iters) =
      gmres_hif(prob.A, prob.b, M, restart, rtol, maxit, verbose);
  timer.finish();
  if (verbose) {
    if (flag == SUCCESS) {
      hif_info("Finished GMRES in %g seconds and %d iterations.", timer.time(),
               iters);
      hif_info("Relative residual of ||b-Ax||/||b||=%e",
               compute_relres(prob.A, prob.b, x));
      hif_info("Success!");
    } else if (flag == STAGNATED)
      hif_info("GMRES stagnated in %d iterations.", iters);
    else if (flag == DIVERGED)
      hif_info("GMRES diverged.");
  }

  if (flag != SUCCESS) {
    hif_info(
        "\nTry to rerun with flag \'--robust\' to enable robust parameters\n");
    hif_info("  If you already did so, then please decrease droptol");
    hif_info("  (params.tau_L, params.tau_U), increase the fill factors");
    hif_info("  (params.alpha_L, params.alpha_U), and/or decrease the");
    hif_info("  inverse-norm threshold (params.kappa, params.kappa_d). Then,");
    hif_info("  recompile the problem.");
  }
  return 0;
}

std::tuple<array_t, int, int> gmres_hif(const matrix_t &A, const array_t &b,
                                        const prec_t &M, const int restart,
                                        const double rtol, const int maxit,
                                        const int verbose) {
  using size_type = array_t::size_type;

  int             iter(0), flag(SUCCESS);
  const size_type n     = b.size();
  const auto      beta0 = hif::norm2(b);

  // create solution vector, starting with all zeros
  array_t x(n);
  std::fill_n(x.begin(), n, 0.0);
  // quick return if possible
  if (beta0 == 0.0) return std::make_tuple(x, SUCCESS, iter);

  // create local workspace
  array_t                  v(n), w(n), y(restart + 1), w2(restart);
  hif::DenseMatrix<double> Q(n, restart), R(restart, restart), J(restart, 2);

  const int max_outer_iters = std::ceil((double)maxit / restart);
  double    resid(1);  // residual norm
  for (int it_outer = 0; it_outer < max_outer_iters; ++it_outer) {
    if (verbose > 1) hif_info(" Enter outer iteration %d...", it_outer + 1);
    // initial residual
    if (iter) {
      A.multiply(x, v);                                       // A*x
      for (size_type i = 0u; i < n; ++i) v[i] = b[i] - v[i];  // b-A*x
    } else
      std::copy_n(b.cbegin(), n, v.begin());
    const double beta = hif::norm2(v);
    y[0]              = beta;
    for (size_type i = 0u; i < n; ++i) Q(i, 0) = v[i] / beta;
    int j(0);  // inner counter
    for (;;) {
      std::copy(Q.col_cbegin(j), Q.col_cend(j), v.begin());
      M.solve(v, w);     // multilevel triangular solve
      A.multiply(w, v);  // matrix-vector

      // Perform Gram-Schmidt orthogonalization
      for (int k = 0u; k <= j; ++k) {
        w2[k] = hif::inner(v, Q.col_cbegin(k));
        for (size_type i = 0u; i < n; ++i) v[i] -= w2[k] * Q(i, k);
      }
      const auto v_norm2 = hif::norm2_sq(v);
      const auto v_norm  = std::sqrt(v_norm2);
      if (j + 1 < restart)
        for (size_type i = 0u; i < n; ++i) Q(i, j + 1) = v[i] / v_norm;

      // Perform Given's rotation to w2
      for (int colJ = 0; colJ + 1 <= j; ++colJ) {
        const auto tmp = w2[colJ];
        w2[colJ]       = hif::conjugate(J(colJ, 0)) * tmp +
                   hif::conjugate(J(colJ, 1)) * w2[colJ + 1];
        w2[colJ + 1] = -J(colJ, 1) * tmp + J(colJ, 0) * w2[colJ + 1];
      }
      const auto rho = std::sqrt(hif::conjugate(w2[j]) * w2[j] + v_norm2);
      J(j, 0)        = w2[j] / rho;
      J(j, 1)        = v_norm / rho;
      y[j + 1]       = -J(j, 1) * y[j];
      y[j]           = hif::conjugate(J(j, 0)) * y[j];
      w2[j]          = rho;
      std::copy_n(w2.cbegin(), j + 1, R.col_begin(j));

      // get residual
      const auto resid_prev = resid;
      resid                 = hif::abs(y[j + 1]) / beta0;  // current residual
      if (resid >= resid_prev * (1.0 - 1e-8)) {
        if (verbose > 1) hif_info("  Solver stagnated!");
        flag = STAGNATED;
        break;
      } else if (iter >= maxit) {
        if (verbose > 1) hif_info("  Reached maxit iteration limit %d", maxit);
        flag = DIVERGED;
        break;
      }
      ++iter;
      if (verbose > 1)
        hif_info("  At iteration %d, relative residual is %g.", iter, resid);
      if (resid <= rtol || j + 1 >= restart) break;
      ++j;
    }  // inf loop
    // backsolve
    for (int k = j; k > -1; --k) {
      y[k] /= R(k, k);
      const auto tmp = y[k];
      for (int i = k - 1; i > -1; --i) y[i] -= tmp * R(i, k);
    }
    // compute Q*y
    std::fill_n(v.begin(), n, 0);
    for (int i = 0; i <= j; ++i) {
      const auto tmp = y[i];
      for (size_type k = 0u; k < n; ++k) v[k] += tmp * Q(k, i);
    }
    // compute M solve
    M.solve(v, w);
    for (size_type k(0); k < n; ++k) x[k] += w[k];  // accumulate sol
    if (resid <= rtol || flag != SUCCESS) break;
  }
  return std::make_tuple(x, flag, iter);
}

void print_help_message(std::ostream &ostr, const char *cmd) {
  ostr
      << "Usage:\n\n"
      << "\t" << cmd << " [options]\n\n"
      << "Options:\n\n"
      << " -h, --help\n"
      << "    Show this help message and exit\n"
      << " -v, --verbose\n"
      << "    Show more output\n"
      << " -q, --quiet\n"
      << "    Show less output\n"
      << " -r, --robust\n"
      << "    Enable robust parameters, default is false\n"
      << " -m, --restart <m>\n"
      << "    Restart in GMRES, default is m=30\n"
      << " -t, --rtol <rtol>\n"
      << "    Relative residual tolerance in GMRES, default is rtol=1e-6\n"
      << " -n, --maxit <maxit>\n"
      << "    Maximum iteration limit in GMRES, default is maxit=500\n"
      << " -Afile, --Afile <Afile>\n"
      << "    LHS matrix stored in Matrix Market format (coordinate), default\n"
      << "    is \'demo_inputs/A.mm\'\n"
      << " -bfile, --bfile <bfile>\n"
      << "    RHS vector stored in Matrix Market format (array), default is\n"
      << "    \'demo_inputs/b.mm\'. If \'Afile\' is provided by the \'bfile\'\n"
      << "    is missing, then b=A*1 will be used\n\n";
}

std::tuple<system_t, int, double, int, int, bool> parse_args(int   argc,
                                                             char *argv[]) {
  using std::string;

  int      restart(30), maxit(500), verbose(1);
  double   rtol(1e-6);
  bool     robust(false);
  system_t prob;

  string Afile, bfile;

  for (int i = 1; i < argc; ++i) {
    auto arg = string(argv[i]);
    if (arg == "-h" || arg == "--help") {
      print_help_message(std::cerr, argv[0]);
      std::exit(0);
    }
    if (arg == "-m" || arg == "--restart") {
      if (i + 1 >= argc) {
        std::cerr << "Missing restart value!\n\n";
        print_help_message(std::cerr, argv[0]);
        std::exit(1);
      }
      restart = std::atoi(argv[++i]);
      if (restart <= 0) restart = 30;
    } else if (arg == "-t" || arg == "--rtol") {
      if (i + 1 >= argc) {
        std::cerr << "Missing rtol value!\n\n";
        print_help_message(std::cerr, argv[0]);
        std::exit(1);
      }
      rtol = std::atof(argv[++i]);
      if (rtol <= 0.0) rtol = 1e-6;
    } else if (arg == "-n" || arg == "--maxit") {
      if (i + 1 >= argc) {
        std::cerr << "Missing maxit value!\n\n";
        print_help_message(std::cerr, argv[0]);
        std::exit(1);
      }
      maxit = std::atoi(argv[++i]);
      if (maxit <= 0) maxit = 500;
    } else if (arg == "-v" || arg == "verbose") {
      verbose = 2;
    } else if (arg == "-q" || arg == "--quiet") {
      verbose = 0;
    } else if (arg == "-Afile" || arg == "--Afile") {
      if (i + 1 >= argc) {
        std::cerr << "Missing Afile!\n\n";
        print_help_message(std::cerr, argv[0]);
        std::exit(1);
      }
      Afile = argv[++i];
    } else if (arg == "-bfile" || arg == "--bfile") {
      if (i + 1 >= argc) {
        std::cerr << "Missing bfile!\n\n";
        print_help_message(std::cerr, argv[0]);
        std::exit(1);
      }
      bfile = argv[++i];
    } else if (arg == "-r" || arg == "--robust")
      robust = true;
  }
  if (Afile.empty()) {
    if (verbose) {
      const bool  dir_prev = std::ifstream("../demo_inputs/A.mm").is_open();
      const char *prefix   = dir_prev ? "../" : "";
      hif_info("Afile is \'%sdemo_inputs/A.mm\'", prefix);
      hif_info("bfile is \'%sdemo_inputs/b.mm\'\n", prefix);
    }
    prob = get_input_data();
  } else {
    const char *bfile_cstr = nullptr;
    if (!bfile.empty()) bfile_cstr = bfile.c_str();
    std::cout << "Afile is \'" << Afile << "\'\n";
    if (!bfile_cstr)
      std::cout << "b=A*1\n\n";
    else
      std::cout << "bfile is \'" << bfile_cstr << "\'\n\n";
    prob = get_input_data(Afile.c_str(), bfile_cstr);
  }
  return std::make_tuple(prob, restart, rtol, maxit, verbose, robust);
}
