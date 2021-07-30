///////////////////////////////////////////////////////////////////////////////
//                  This file is part of the HIFIR library                   //
///////////////////////////////////////////////////////////////////////////////

/*

  This file contains an example of using HIF as right-preconditioner for
  GMRES(m), where HIF can be computed on a sparser matrix than the one used
  in the GMRES solver, i.e.,
          A*M^{g}*y=b then x=M^{g}*y,
  where M is computed on a sparser matrix (hence the name "sparsifier") S, s.t.
  nnz(S) <= nnz(A). In this example, we solve the advection-diffusion (AD)
  equation with FDM method on the unit square discretized by an equidistance
  structured grid of 64x64. For the coefficient matrix, we use 4th order
  FDM, whereas the sparsifier uses 2nd order FDM.

  Try with

    ./demo_sparsifier.exe [GMRES options]  # using sparsifier

  and compare with

    ./demo_gmreshif.exe -Afile ../demo_inputs/ad-fdm4.mm

  Author: Qiao Chen
  Level: Advanced

*/

#include <cstdlib>
#include <iostream>
#include <tuple>

#include "gmres.hpp"
#include "hifir.hpp"

using prec_t = hif::HIF<double, int>;

// parse command-line arguments for system, sparsifier restart, rtol, maxit,
// verbose, and robust parameters, full rank flag, dense thres
std::tuple<system_t, matrix_t, int, double, int, int, bool, bool, int>
parse_args(int argc, char *argv[]);

int main(int argc, char *argv[]) {
  // parse options for gmres
  int      restart, maxit, verbose, dense_thres;
  double   rtol;
  double   robust, full_rank;
  system_t prob;
  matrix_t S;
  std::tie(prob, S, restart, rtol, maxit, verbose, robust, full_rank,
           dense_thres) = parse_args(argc, argv);

  if (S.nrows() == 0u) {
    // We will prune tiny values in A, and make a copy of the result in S.
    // Note that by default, the constructor only performs shallow copies
    bool deep_copy = true;
    S              = matrix_t(prob.A, deep_copy);
    // the following functionwill prune any values that close to machine
    // precision. In particular, S.prune(tol) will eliminate an entry a_ij
    // whose mag is smaller than tol*min{max(abs(ith row)),max(abs(jth column)}
    const auto pruned = S.prune(1e-15);
    if (verbose) hif_info("Eliminated %zd tiny entries in S.", pruned);
  }

  // See if we have user-provide sparsifier
  if (verbose) {
    hif_info("Coefficient matrix and sparsifier have %zd and %zd nnz, resp.",
             prob.A.nnz(), S.nnz());
  }
  // get timer
  hif::DefaultTimer timer;

  // create HIF preconditioner, and factorize with default params
  auto M      = prec_t();
  auto params = hif::DEFAULT_PARAMS;
  if (full_rank) params.rrqr_cond = hif::Const<double>::MAX;
  if (dense_thres > 0) params.dense_thres = dense_thres;
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
  M.factorize(S, params);  // we factorize S here not A
  timer.finish();

  if (verbose) {
    hif_info("HIF(lvls=%zd) finished in %g seconds with nnz ratio %.2f%%.\n",
             M.levels(), timer.time(), 100.0 * M.nnz() / S.nnz());
    // call HIF-preconditioned GMRES
    hif_info("Invoke HIF-preconditioned GMRES(%d) with rtol=%g and maxit=%d",
             restart, rtol, maxit);
  }

  array_t x;
  int     flag, iters;
  timer.start();
  // NOTE: The input is A here not S
  std::tie(x, flag, iters) =
      gmres_hif(prob.A, prob.b, M, restart, rtol, maxit, verbose, full_rank);
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
      << "    Use robust parameters for HIF, default is false\n"
      << " -f, --full-rank\n"
      << "    Using full rank preconditioner (for singular M only)\n"
      << " -d, --dense-thres <dense-thres>\n"
      << "    Dense threshold for final Schur complement (dense-thres=2000)\n"
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
      << "    is missing, then b=A*1 will be used\n"
      << " -Sfile, --Sfile <Sfile>\n"
      << "    Sparsifier, on which we will compute the HIF preconditioner.\n"
      << "    If omitted, then (1) we will compute S by clipping tiny values\n"
      << "    in A if \'Afile\' is specified, or (2) we will load the 2nd\n"
      << "    FDM operator, which is used as sparsifier\n\n";
}

std::tuple<system_t, matrix_t, int, double, int, int, bool, bool, int>
parse_args(int argc, char *argv[]) {
  using std::string;

  int    restart(30), maxit(500), verbose(1), dense_thres(-1);
  double rtol(1e-6);
  bool   robust(false), full_rank(false);

  string Afile, bfile, Sfile;

  for (int i = 1; i < argc; ++i) {
    auto arg = string(argv[i]);
    if (arg == "-h" || arg == "--help") {
      print_help_message(std::cout, argv[0]);
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
    } else if (arg == "-v" || arg == "--verbose") {
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
    } else if (arg == "-Sfile" || arg == "--Sfile") {
      if (i + 1 >= argc) {
        std::cerr << "Missing Sfile!\n\n";
        print_help_message(std::cerr, argv[0]);
        std::exit(1);
      }
      Sfile = argv[++i];
    } else if (arg == "-r" || arg == "--robust")
      robust = true;
    else if (arg == "-f" || arg == "--full-rank")
      full_rank = true;
    else if (arg == "-d" || arg == "--dense-thres") {
      if (i + 1 >= argc) {
        std::cerr << "Missing dense-thres!\n\n";
        print_help_message(std::cerr, argv[0]);
        std::exit(1);
      }
      dense_thres = std::atoi(argv[++i]);
    }
  }
  // load 4th order FDM with A*1 as rhs
  if (Afile.empty()) {
    const bool prev_dir = std::ifstream("../demo_inputs/ad-fdm4.mm").is_open();
    if (verbose) {
      const char *prefix = prev_dir ? "../" : "";
      hif_info("Afile is \'%sdemo_inputs/ad-fdm4.mm\'", prefix);
      hif_info("b=A*1");
      hif_info("Sfile is \'%sdemo_inputs/ad-fdm2.mm\'\n", prefix);
    }
    return std::make_tuple(
        prev_dir ? get_input_data("../demo_inputs/ad-fdm4.mm")
                 : get_input_data("demo_inputs/ad-fdm4.mm"),
        prev_dir ? matrix_t::from_mm("../demo_inputs/ad-fdm2.mm")
                 : matrix_t::from_mm("demo_inputs/ad-fdm2.mm"),
        restart, rtol, maxit, verbose, robust, full_rank, dense_thres);
  }

  const char *bfile_cstr = nullptr;
  if (!bfile.empty()) bfile_cstr = bfile.c_str();
  auto     prob = get_input_data(Afile.c_str(), bfile_cstr);
  matrix_t S;
  if (!Sfile.empty()) S = matrix_t::from_mm(Sfile.c_str());
  if (verbose) {
    std::cout << "Afile is \'" << Afile << "\'\n";
    if (bfile_cstr)
      std::cout << "bfile is \'" << bfile << "\'\n";
    else
      std::cout << "b=A*1\n";
    if (Sfile.empty())
      std::cout << "Sfile is omitted (will prune A as sparsifier)\n\n";
    else
      std::cout << "Sfile is \'" << Sfile << "\'\n\n";
  }
  return std::make_tuple(prob, S, restart, rtol, maxit, verbose, robust,
                         full_rank, dense_thres);
}
