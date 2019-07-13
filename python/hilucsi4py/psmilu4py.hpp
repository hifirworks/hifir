//@HEADER
//----------------------------------------------------------------------------
//                Copyright (C) 2019 The PSMILU AUTHORS
//----------------------------------------------------------------------------
//@HEADER

// Authors:
//  Qiao,

#ifndef _PSMILU_PYTHON_PSMILU4PY_HPP
#define _PSMILU_PYTHON_PSMILU4PY_HPP

#include <string>
#include <vector>

// first of all, include the common
#include "common.hpp"

// then include the file descriptor wrapper
#include "file_dp_api.h"

// stdout wrapper
#define PSMILU_STDOUT(__msg)     \
  do {                           \
    import_psmilu4py__file_dp(); \
    psmilu4py_stdout(__msg);     \
  } while (false)

// stderr wrapper
#define PSMILU_STDERR(__msg)     \
  do {                           \
    import_psmilu4py__file_dp(); \
    psmilu4py_stderr(__msg);     \
  } while (false)

// test for mc64
#if PSMILU4PY_USE_MC64
#  define PSMILU_ENABLE_MC64
#endif  // PSMILU4PY_USE_MC64

// now, include the psmilu code generator
#include <PSMILU.hpp>

namespace psmilu {

// read native psmilu format
inline void read_native_psmilu(const std::string &fn, std::size_t &nrows,
                               std::size_t &ncols, std::size_t &m,
                               std::vector<int> &   indptr,
                               std::vector<int> &   indices,
                               std::vector<double> &vals,
                               const bool           is_crs = true,
                               const bool           is_bin = true) {
  Array<int>    ind_ptr, inds;
  Array<double> values;
  if (is_crs) {
    using crs_type = C_Default_PSMILU::crs_type;
    auto A         = is_bin ? crs_type::from_native_bin(fn.c_str(), &m)
                    : crs_type::from_native_ascii(fn.c_str(), &m);
    nrows = A.nrows();
    ncols = A.ncols();
    ind_ptr.swap(A.row_start());
    inds.swap(A.col_ind());
    values.swap(A.vals());
  } else {
    using ccs_type = C_Default_PSMILU::ccs_type;
    auto A         = is_bin ? ccs_type::from_native_bin(fn.c_str(), &m)
                    : ccs_type::from_native_ascii(fn.c_str(), &m);
    nrows = A.nrows();
    ncols = A.ncols();
    ind_ptr.swap(A.col_start());
    inds.swap(A.row_ind());
    values.swap(A.vals());
  }
  // efficient for c++11
  indptr  = std::vector<int>(ind_ptr.cbegin(), ind_ptr.cend());
  indices = std::vector<int>(inds.cbegin(), inds.cend());
  vals    = std::vector<double>(values.cbegin(), values.cend());
}

// write native psmilu format
inline void write_native_psmilu(const std::string &fn, const std::size_t nrows,
                                const std::size_t ncols, const int *indptr,
                                const int *indices, const double *vals,
                                const std::size_t m0, const bool is_crs = true,
                                const bool is_bin = true) {
  constexpr static bool WRAP = true;
  if (is_crs) {
    using crs_type = C_Default_PSMILU::crs_type;
    const crs_type A(nrows, ncols, const_cast<int *>(indptr),
                     const_cast<int *>(indices), const_cast<double *>(vals),
                     WRAP);
    // aggressively checking
    A.check_validity();
    if (is_bin)
      A.write_native_bin(fn.c_str(), m0);
    else
      A.write_native_ascii(fn.c_str(), m0);
  } else {
    using ccs_type = C_Default_PSMILU::ccs_type;
    const ccs_type A(nrows, ncols, const_cast<int *>(indptr),
                     const_cast<int *>(indices), const_cast<double *>(vals),
                     WRAP);
    if (is_bin)
      A.write_native_bin(fn.c_str(), m0);
    else
      A.write_native_ascii(fn.c_str(), m0);
  }
}

// In order to make things easier, we directly use the raw data types, thus
// we need to create a child class.
class PyPSMILU : public C_Default_PSMILU {
 public:
  using base      = C_Default_PSMILU;
  using size_type = base::size_type;

  // factorize crs
  inline void factorize_crs(const size_type nrows, const size_type ncols,
                            const int *rowptr, const int *colind,
                            const double *vals, const size_type m0,
                            const Options &opts, const bool check) {
    using crs_type             = base::crs_type;
    constexpr static bool WRAP = true;
    const crs_type        A(nrows, ncols, const_cast<int *>(rowptr),
                            const_cast<int *>(colind), const_cast<double *>(vals),
                            WRAP);
    base::factorize(A, m0, opts, check);
  }

  // factorize ccs
  inline void factorize_ccs(const size_type nrows, const size_type ncols,
                            const int *colptr, const int *rowind,
                            const double *vals, const size_type m0,
                            const Options &opts, const bool check) {
    using ccs_type             = base::ccs_type;
    constexpr static bool WRAP = true;
    const ccs_type        A(nrows, ncols, const_cast<int *>(colptr),
                            const_cast<int *>(rowind), const_cast<double *>(vals),
                            WRAP);
    base::factorize(A, m0, opts, check);
  }

  inline void factorize(const size_type nrows, const size_type ncols,
                        const int *indptr, const int *indices,
                        const double *vals, const size_type m0,
                        const Options &opts, const bool check,
                        const bool is_crs) {
    if (is_crs)
      factorize_crs(nrows, ncols, indptr, indices, vals, m0, opts, check);
    else
      factorize_ccs(nrows, ncols, indptr, indices, vals, m0, opts, check);
  }

  // overload solve
  inline void solve(const size_type n, const double *b, double *x) const {
    using array_type           = base::array_type;
    constexpr static bool WRAP = true;
    const array_type      B(n, const_cast<double *>(b), WRAP);
    array_type            X(n, x, WRAP);
    base::solve(B, X);
  }
};

}  // namespace psmilu

#endif  // _PSMILU_PYTHON_PSMILU4PY_HPP
