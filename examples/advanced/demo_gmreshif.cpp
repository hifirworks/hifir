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
                                        const bool   verbose = true);

// parse command-line arguments for restart, rtol, maxit, and verbose
std::tuple<system_t, int, double, int, bool> parse_args(int argc, char *argv[]);

int main(int argc, char *argv[]) {
  // parse options for gmres
  int      restart, maxit;
  double   rtol;
  bool     verbose;
  system_t prob;
  std::tie(prob, restart, rtol, maxit, verbose) = parse_args(argc, argv);

  // get timer
  hif::DefaultTimer timer;

  // create HIF preconditioner, and factorize with default params
  auto M      = prec_t();
  auto params = hif::DEFAULT_PARAMS;
  // tune the following parameters if necessary
  params.tau_L = params.tau_U = 1e-2;     // droptol
  params.alpha_L = params.alpha_U = 3.0;  // fill factors
  params.kappa = params.kappa_d = 5.0;    // inverse-norm thres
  params.verbose                = hif::VERBOSE_NONE;
  timer.start();
  M.factorize(prob.A, params);
  timer.finish();
  hif_info("HIF(lvls=%zd) finished in %.2g seconds with nnz ratio %.2f%%.\n",
           M.levels(), timer.time(), 100.0 * M.nnz() / prob.A.nnz());

  // call HIF-preconditioned GMRES
  hif_info("Invoke HIF-preconditioned GMRES(%d) with rtol=%g and maxit=%d",
           restart, rtol, maxit);
  array_t x;
  int     flag, iters;
  timer.start();
  std::tie(x, flag, iters) =
      gmres_hif(prob.A, prob.b, M, restart, rtol, maxit, verbose);
  timer.finish();
  if (flag == SUCCESS)
    hif_info("Finished GMRES in %.2g seconds and %d iterations.", timer.time(),
             iters);
  else if (flag == STAGNATED)
    hif_info("GMRES stagnated in %d iterations.", iters);
  else if (flag == DIVERGED)
    hif_info("GMRES diverged.");
  return 0;
}

std::tuple<array_t, int, int> gmres_hif(const matrix_t &A, const array_t &b,
                                        const prec_t &M, const int restart,
                                        const double rtol, const int maxit,
                                        const bool verbose) {
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
    if (verbose) hif_info(" Enter outer iteration %d...", it_outer + 1);
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
      std::copy_n(w.cbegin(), j + 1, R.col_begin(j));

      // get residual
      const auto resid_prev = resid;
      resid                 = hif::abs(y[j + 1]) / beta0;  // current residual
      if (resid >= resid_prev * (1.0 - 1e-8)) {
        if (verbose) hif_info("  Solver stagnated!");
        flag = STAGNATED;
        break;
      } else if (iter >= maxit) {
        if (verbose) hif_info("  Reached maxit iteration limit %d", maxit);
        flag = DIVERGED;
        break;
      }
      ++iter;
      if (verbose)
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

std::tuple<system_t, int, double, int, bool> parse_args(int   argc,
                                                        char *argv[]) {
  using std::string;
  static const char *help_message =
      "usage:\n\n"
      "\t./demo_gmreshif.exe [options] [flags]\n\n"
      "Options:\n\n"
      " -m|--restart m\n"
      "    Restart in GMRES, default is m=30\n"
      " -t|--rtol rtol\n"
      "    Relative residual tolerance in GMRES, default is rtol=1e-6\n"
      " -n|--maxit maxit\n"
      "    Maximum iteration limit in GMRES, default is maxit=500\n"
      " -Afile Afile\n"
      "    LHS matrix stored in Matrix Market format (coordinate)\n"
      " -bfile bfile\n"
      "    RHS vector stored in Matrix Market format (array), if \'Afile\'\n"
      "    is provided but \'bfile\' is missing, then b=A*1 will be used\n\n"
      "Flags:\n\n"
      " -s|--silent\n"
      "    Disable verbose message in GMRES, default is false (verbose on)\n"
      " -h|--help\n"
      "    Show this help message and exit\n";

  int      restart(30), maxit(500);
  double   rtol(1e-6);
  bool     verbose(true);
  system_t prob;

  string Afile, bfile;

  for (int i = 1; i < argc; ++i) {
    auto arg = string(argv[i]);
    if (arg == "-h" || arg == "--help") {
      std::cout << help_message;
      std::exit(0);
    }
    if (arg == "-s" || arg == "--silent")
      verbose = false;
    else if (arg == "-m" || arg == "--restart") {
      if (i + 1 >= argc) {
        std::cerr << "Missing restart value!\n\n" << help_message;
        std::exit(1);
      }
      restart = std::atoi(argv[++i]);
      if (restart <= 0) restart = 30;
    } else if (arg == "-t" || arg == "--rtol") {
      if (i + 1 >= argc) {
        std::cerr << "Missing rtol value!\n\n" << help_message;
        std::exit(1);
      }
      rtol = std::atof(argv[++i]);
      if (rtol <= 0.0) rtol = 1e-6;
    } else if (arg == "-n" || arg == "--maxit") {
      if (i + 1 >= argc) {
        std::cerr << "Missing maxit value!\n\n" << help_message;
        std::exit(1);
      }
      maxit = std::atoi(argv[++i]);
      if (maxit <= 0) maxit = 500;
    } else if (arg == "-Afile") {
      if (i + 1 >= argc) {
        std::cerr << "Missing Afile!\n\n" << help_message;
        std::exit(1);
      }
      Afile = argv[++i];
    } else if (arg == "-bfile") {
      if (i + 1 >= argc) {
        std::cerr << "Missing bfile!\n\n" << help_message;
        std::exit(1);
      }
      bfile = argv[++i];
    }
  }
  if (Afile.empty())
    prob = get_input_data();
  else {
    const char *bfile_cstr = nullptr;
    if (!bfile.empty()) bfile_cstr = bfile.c_str();
    prob = get_input_data(Afile.c_str(), bfile_cstr);
  }
  return std::make_tuple(prob, restart, rtol, maxit, verbose);
}
