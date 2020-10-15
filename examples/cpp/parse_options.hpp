//===========================================================================//
//                This file is part of HILUCSI project                       //
//===========================================================================//

// common interface for parsing user options for demo programs
// authors: Qiao,
//
// Copyright (C) 2020 NumGeom Group at Stony Brook University
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

#pragma once

#include <cstdlib>
#include <iostream>
#include <string>
#include <tuple>

#include "HILUCSI.hpp"

using std::string;

const static char *help =
    "\n"
    "usage:\n"
    "\n"
    " ./demo_mixed case [options] [flags]\n"
    "\n"
    "where the \"case\" is an HILUCSI benchmark input directory that contains\n"
    "a matrix file and an ASCII rhs file. The matrix file is in HILUCSI\n"
    "binary format and has name of \"A.psmilu\", while the rhs file is a "
    "list\n"
    "of rhs entries that are separated by native C++ separator, i.e. space or\n"
    "new line (the name is \"b.txt\".)\n"
    "\n"
    "options:\n"
    "\n"
    " -t|--tau tau\n"
    "\tdrop tolerance (1e-4)\n"
    " -k|--kappa kappa\n"
    "\tinverse norm threshold (3)\n"
    " -a|--alpha alpha\n"
    "\tlocal space control parameter (10)\n"
    " -v|--verbose level\n"
    "\tverbose information printing level (0)\n"
    " -r|--reorder method\n"
    "\treordering methods:\n"
    "\t\t0: off\n"
    "\t\t1: auto (amd), default\n"
    "\t\t2: amd\n"
    "\t\t3: rcm\n"
    " -I|-pre-scale\n"
    "\ta priori scaling before calling matching/scaling\n"
    "\t\t0: off (default)\n"
    "\t\t1: local extreme scale\n"
    "\t\t2: iterative scaling based on inf-norm (experimental)\n"
    " -p|--pivoting\n"
    "\tpivoting strategy:\n"
    "\t\t0: off (default)\n"
    "\t\t1: on\n"
    "\t\t2: auto\n"
    " -g|--gamma\n"
    "\tpivoting threshold (1.0)\n"
    " -P|--ksp\n"
    "\tKSP methods:\n"
    "\t\t0: FGMRES (default)\n"
    "\t\t1: FQMRCGSTAB\n"
    "\t\t2: TGMRESR\n"
    "\t\t3: FBICGSTAB\n"
    " -S|--symm-pre-lvls\n"
    "\tlevels to apply symmetric preprocessing, nonpositive number indicates\n"
    "\tapply symmetric preprocessing for all levels (1)\n"
    " -T|--rtol\n"
    "\trelative tolerance for KSP (1e-6) solver\n"
    " -R|--restart\n"
    "\trestart/cycle for KSP (30)\n"
    " -K|--kernel\n"
    "\tinner kernel for KSP\n"
    "\t\t0: traditional right preconditioner, default\n"
    "\t\t1: Iterative refinement (IR) process as right preconditioner\n"
    "\t\t2: Chebyshev-IR process as right precond\n"
    "\n"
    "flags:\n"
    "\n"
    " -h|--help\n"
    "\tshow help information and exit\n"
    " -V|--version\n"
    "\tshow version and exit\n"
    " -n|--no-saddle\n"
    "\tdisable static deferals for saddle point problems\n"
    " -N|--no-refine\n"
    "\tdisable parameter refinement from level to level\n"
    " -s|--symm\n"
    "\ttreat as symmetric problems\n"
    " -\n"
    "\tindicator for reading Options from stdin\n"
    "\n"
    "examples:\n"
    "\n"
    "\t./hilucsi_demo my_test\n"
    " use default setting solve the problem in \"my_test\" directory.\n"
    "\t./hilucsi_demo my_test - < ./parameters.cfg\n"
    " solve with user-defined parameters and read in from stdin\n"
    "\t./hilucsi_demo my_test -R 1e-12 -t 1e-4\n"
    " solve with rtol=1e-12 and drop-tol=1e-4\n"
    "\n";

static std::tuple<hilucsi::Options, int, double, bool, int, int> parse_args(
    int argc, char *argv[]) {
  hilucsi::Options opts    = hilucsi::get_default_options();
  int              restart = 30;
  double           tol     = 1e-6;
  bool             symm    = false;
  int              kernel  = hilucsi::ksp::TRADITION;
  int              ksp     = 0;
  opts.verbose             = hilucsi::VERBOSE_NONE;
  if (argc < 2) {
    std::cerr << "Missing input directory!\n" << help;
    std::exit(1);
  }
  std::string arg;
  arg = argv[1];
  if (arg == string("-h") || arg == string("--help")) {
    std::cout << help;
    std::exit(0);
  }
  if (arg == string("-V") || arg == string("--version")) {
    std::cout << "HILUCSI version: " << hilucsi::version() << std::endl;
    std::exit(0);
  }
  if (arg == string("") || arg[0] == '-') {
    std::cerr << "Invalid input directory " << arg << std::endl;
    std::exit(1);
  }
  auto fatal_exit = [&](const char *reason) {
    std::cerr << reason << std::endl;
    std::exit(1);
  };
  int i(2);
  for (;;) {
    if (i >= argc) break;
    arg = argv[i];
    if (arg == string("-h") || arg == string("--help")) {
      std::cout << help;
      std::exit(0);
    }
    if (arg == string("-V") || arg == string("--version")) {
      std::cout << "HILUCSI version: " << hilucsi::version() << std::endl;
      std::exit(0);
    }
    if (arg == string("-t") || arg == string("--tau")) {
      ++i;
      if (i >= argc) fatal_exit("missing drop tolerance (tau) value!");
      opts.tau_L = opts.tau_U = std::atof(argv[i]);
    } else if (arg == string("-k") || arg == string("--kappa")) {
      ++i;
      if (i >= argc) fatal_exit("missing inverse norm thres (kappa) value!");
      opts.tau_d = opts.tau_kappa = std::atof(argv[i]);
    } else if (arg == string("-a") || arg == string("--alpha")) {
      ++i;
      if (i >= argc) fatal_exit("missing space control (alpha) value!");
      opts.alpha_L = opts.alpha_U = std::atof(argv[i]);
    } else if (arg == string("-v") || arg == string("--verbose")) {
      ++i;
      if (i >= argc) fatal_exit("missing verbose level!");
      opts.verbose = std::atoi(argv[i]);
    } else if (arg == string("-r") || arg == string("--reorder")) {
      ++i;
      if (i >= argc) fatal_exit("missing reorder method tag!");
      opts.reorder = std::atoi(argv[i]);
    } else if (arg == string("-I") || arg == string("-pre-scale")) {
      ++i;
      if (i >= argc) fatal_exit("missing pre-scale type!");
      opts.pre_scale = std::atoi(argv[i]);
    } else if (arg == string("-S") || arg == string("--symm-pre-lvls")) {
      ++i;
      if (i >= argc) fatal_exit("missing number of symmetric pre levels!");
      opts.symm_pre_lvls = std::atoi(argv[i]);
    } else if (arg == string("-n") || arg == string("--no-saddle")) {
      opts.saddle = 0;
    } else if (arg == string("-N") || arg == string("--no-par-refine")) {
      opts.rf_par = 0;
    } else if (arg == string("-")) {
      // read options from stdin
      std::cout << "read options from stdin" << std::endl;
      std::cin >> opts;
    } else if (arg == string("-s") || arg == string("--symm")) {
      symm = true;
    } else if (arg == string("-P") || arg == string("--ksp")) {
      ++i;
      if (i >= argc) fatal_exit("missing KSP solver choice!");
      ksp = std::atoi(argv[i]);
    } else if (arg == string("-T") || arg == string("--rtol")) {
      ++i;
      if (i >= argc) fatal_exit("missing KSP rtol value!");
      tol = std::atof(argv[i]);
    } else if (arg == string("-R") || arg == string("--restart")) {
      ++i;
      if (i >= argc) fatal_exit("missing GMRES restart/cycle value!");
      restart = std::atoi(argv[i]);
    } else if (arg == string("-K") || arg == string("--kernel")) {
      ++i;
      if (i >= argc) fatal_exit("missing KSP kernel choice!");
      kernel = std::atoi(argv[i]);
    } else if (arg == string("-p") || arg == string("--pivoting")) {
      ++i;
      if (i >= argc) fatal_exit("missing pivoting strategy!");
      opts.pivot = std::atoi(argv[i]);
    } else if (arg == string("-g") || arg == string("--gamma")) {
      ++i;
      if (i >= argc) fatal_exit("missing pivoting threshold!");
      opts.gamma = std::atof(argv[i]);
    }
    ++i;
  }
  return std::make_tuple(opts, restart, tol, symm, kernel, ksp);
}
