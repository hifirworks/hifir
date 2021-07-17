///////////////////////////////////////////////////////////////////////////////
//                  This file is part of the HIFIR library                   //
///////////////////////////////////////////////////////////////////////////////

/// \headerfile demo_utils.hpp "../demo_utils.hpp"

#ifndef _HIFIR_DEMOUTILS_HPP
#define _HIFIR_DEMOUTILS_HPP

#include <cstdlib>
#include <fstream>
#include <iostream>

#include "hifir.hpp"

using matrix_t = hif::CRS<double, int>;
using array_t  = hif::Array<double>;

// a simple structure represents a linear system
typedef struct {
  matrix_t A;
  array_t  b;
} system_t;

inline system_t get_user_data(const char *A_file, const char *b_file) {
  auto A = matrix_t::from_mm(A_file);
  // if the rhs file passed in
  if (b_file) return (system_t){.A = A, .b = array_t::from_mm(b_file)};
  // using A*1 as the rhs
  array_t b(A.nrows()), x(A.ncols());
  std::fill(x.begin(), x.end(), 1.0);
  A.multiply(x, b);
  return (system_t){.A = A, .b = b};
}

// load input data
inline system_t get_input_data(const char *A_file = nullptr,
                               const char *b_file = nullptr) {
  if (!A_file) {
    std::ifstream f("demo_inputs/A.mm");
    if (f.is_open()) {
      f.close();
      return (system_t){.A = matrix_t::from_mm("demo_inputs/A.mm"),
                        .b = array_t::from_mm("demo_inputs/b.mm")};
    }
    return (system_t){.A = matrix_t::from_mm("../demo_inputs/A.mm"),
                      .b = array_t::from_mm("../demo_inputs/b.mm")};
  } else
    return get_user_data(A_file, b_file);
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

inline std::pair<std::string, std::string> parse_input_files(int   argc,
                                                             char *argv[]) {
  using std::string;
  static const char *help_message =
      "usage:\n\n"
      "\t./demo_exe [-Afile file] [-bfile file]\n\n"
      "Where \'Afile\' is the LHS (matrix) file and \'bfile\' is the RHS "
      "file.\n"
      "The default files are \'demo_inputs/{A,b}.mm\'.\n"
      "If only the LHS file is provided by the user, then b=A*1 will be "
      "used.\n";

  string Afile, bfile;
  if (argc == 1) return std::make_pair(Afile, bfile);
  for (int i = 1; i < argc; ++i) {
    string arg = argv[i];
    if (arg == "-h" || arg == "--help") {
      std::cout << help_message;
      std::exit(0);
    }
    if (arg == "-Afile") {
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
  return std::make_pair(Afile, bfile);
}

inline system_t parse_cmd4input(int argc, char *argv[]) {
  const auto files = parse_input_files(argc, argv);
  if (files.first.empty()) return get_input_data();
  const char *bfile = nullptr;
  if (!files.second.empty()) bfile = files.second.c_str();
  return get_input_data(files.first.c_str(), bfile);
}

#endif  // _HIFIR_DEMOUTILS_HPP
