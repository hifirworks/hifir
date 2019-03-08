//@HEADER
//----------------------------------------------------------------------------
//                Copyright (C) 2019 The PSMILU AUTHORS
//----------------------------------------------------------------------------
//@HEADER

// Authors:
//  Qiao,

#ifndef _PSMILU_PYTHON_PSMILU4PY_HPP
#define _PSMILU_PYTHON_PSMILU4PY_HPP

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

// now, include the psmilu code generator
#include <PSMILU.hpp>

namespace psmilu {

// In order to make things easier, we directly use the raw data types, thus we
// need to create a child class.
class PyBuilder : public C_DefaultBuilder {
 public:
  using base      = C_DefaultBuilder;
  using size_type = base::size_type;

  // compute crs
  inline void compute_crs(const size_type nrows, const size_type ncols,
                          const int *rowptr, const int *colind,
                          const double *vals, const size_type m0,
                          const Options &opts, const bool check) {
    using crs_type             = base::crs_type;
    constexpr static bool WRAP = true;
    const crs_type        A(nrows, ncols, const_cast<int *>(rowptr),
                            const_cast<int *>(colind), const_cast<double *>(vals),
                            WRAP);
    base::compute(A, m0, opts, check);
  }

  // compute ccs
  inline void compute_ccs(const size_type nrows, const size_type ncols,
                          const int *colptr, const int *rowind,
                          const double *vals, const size_type m0,
                          const Options &opts, const bool check) {
    using ccs_type             = base::ccs_type;
    constexpr static bool WRAP = true;
    const ccs_type        A(nrows, ncols, const_cast<int *>(colptr),
                            const_cast<int *>(rowind), const_cast<double *>(vals),
                            WRAP);
    base::compute(A, m0, opts, check);
  }

  inline void compute(const size_type nrows, const size_type ncols,
                      const int *indptr, const int *indices, const double *vals,
                      const size_type m0, const Options &opts, const bool check,
                      const bool is_crs) {
    if (is_crs)
      compute_crs(nrows, ncols, indptr, indices, vals, m0, opts, check);
    else
      compute_ccs(nrows, ncols, indptr, indices, vals, m0, opts, check);
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
