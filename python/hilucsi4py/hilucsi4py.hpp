//@HEADER
//----------------------------------------------------------------------------
//                Copyright (C) 2019 The HILUCSI AUTHORS
//----------------------------------------------------------------------------
//@HEADER

// Authors:
//  Qiao,

#ifndef _HILUCSI_PYTHON_HILUCSI4PY_HPP
#define _HILUCSI_PYTHON_HILUCSI4PY_HPP

// NOTE we need to ensure C++ code throws exceptions
#ifndef HILUCSI_THROW
#  define HILUCSI_THROW
#endif  // HILUCSI_THROW

// we need to make sure that the stdout and stderr are not pre-defined
#ifdef HILUCSI_STDOUT
#  undef HILUCSI_STDOUT
#endif  // HILUCSI_STDOUT
#ifdef HILUCSI_STDERR
#  undef HILUCSI_STDERR
#endif  // HILUCSI_STDOUT

#include <string>
#include <vector>

// then include the file descriptor wrapper
#include "file_dp_api.h"

// stdout wrapper
#define HILUCSI_STDOUT(__msg)     \
  do {                            \
    import_hilucsi4py__file_dp(); \
    hilucsi4py_stdout(__msg);     \
  } while (false)

// stderr wrapper
#define HILUCSI_STDERR(__msg)     \
  do {                            \
    import_hilucsi4py__file_dp(); \
    hilucsi4py_stderr(__msg);     \
  } while (false)

// now, include the hilucsi code generator
#include <HILUCSI.hpp>

namespace hilucsi {

// types
using py_crs_type   = DefaultHILUCSI::crs_type;
using py_array_type = DefaultHILUCSI::array_type;
using size_type     = DefaultHILUCSI::size_type;

// read native hilucsi format
inline void read_hilucsi(const std::string &fn, std::size_t &nrows,
                         std::size_t &ncols, std::size_t &m,
                         std::vector<int> &indptr, std::vector<int> &indices,
                         std::vector<double> &vals, const bool is_bin = true) {
  Array<int>    ind_ptr, inds;
  Array<double> values;
  auto          A = is_bin ? py_crs_type::from_bin(fn.c_str(), &m)
                  : py_crs_type::from_ascii(fn.c_str(), &m);
  nrows = A.nrows();
  ncols = A.ncols();
  ind_ptr.swap(A.row_start());
  inds.swap(A.col_ind());
  values.swap(A.vals());
  // efficient for c++11
  indptr  = std::vector<int>(ind_ptr.cbegin(), ind_ptr.cend());
  indices = std::vector<int>(inds.cbegin(), inds.cend());
  vals    = std::vector<double>(values.cbegin(), values.cend());
}

// write native hilucsi format
inline void write_hilucsi(const std::string &fn, const std::size_t nrows,
                          const std::size_t ncols, const int *indptr,
                          const int *indices, const double *vals,
                          const std::size_t m0, const bool is_bin = true) {
  constexpr static bool WRAP = true;
  const py_crs_type     A(nrows, ncols, const_cast<int *>(indptr),
                          const_cast<int *>(indices), const_cast<double *>(vals),
                          WRAP);
  // aggressively checking
  A.check_validity();
  if (is_bin)
    A.write_bin(fn.c_str(), m0);
  else
    A.write_ascii(fn.c_str(), m0);
}

// query file information
inline void query_hilucsi_info(const std::string &fn, bool &is_row, bool &is_c,
                               bool &is_double, bool &is_real,
                               std::uint64_t &nrows, std::uint64_t &ncols,
                               std::uint64_t &nnz, std::uint64_t &m,
                               const bool is_bin = true) {
  std::tie(is_row, is_c, is_double, is_real, nrows, ncols, nnz, m) =
      is_bin ? query_info_bin(fn.c_str()) : query_info_ascii(fn.c_str());
}

// In order to make things easier, we directly use the raw data types, thus
// we need to create a child class.
class PyHILUCSI : public DefaultHILUCSI {
 public:
  using base = DefaultHILUCSI;

  // factorize crs
  inline void factorize(const size_type n, const int *rowptr, const int *colind,
                        const double *vals, const size_type m0,
                        const Options &opts) {
    constexpr static bool WRAP = true;

    py_crs_type A(n, n, const_cast<int *>(rowptr), const_cast<int *>(colind),
                  const_cast<double *>(vals), WRAP);
    base::factorize(A, m0, opts);
  }

  using base::solve;

  // overload solve
  inline void solve(const size_type n, const double *b, double *x) const {
    constexpr static bool WRAP = true;

    const array_type B(n, const_cast<double *>(b), WRAP);
    array_type       X(n, x, WRAP);
    solve(B, X);
  }
};

// mixed precision, using float preconditioner
class PyHILUCSI_Mixed : public HILUCSI<float, int> {
 public:
  using base = HILUCSI<float, int>;

  // factorize crs
  inline void factorize(const size_type n, const int *rowptr, const int *colind,
                        const double *vals, const size_type m0,
                        const Options &opts) {
    constexpr static bool WRAP = true;

    py_crs_type A(n, n, const_cast<int *>(rowptr), const_cast<int *>(colind),
                  const_cast<double *>(vals), WRAP);
    base::factorize(A, m0, opts);
  }

  using base::solve;

  // overload solve
  inline void solve(const size_type n, const double *b, double *x) const {
    constexpr static bool WRAP = true;

    const py_array_type B(n, const_cast<double *>(b), WRAP);
    py_array_type       X(n, x, WRAP);
    solve(B, X);
  }
};

// using a template base for Ksp solver
template <template <class, class> class Ksp, class MType = PyHILUCSI,
          class ValueType = void>
class KspAdapt : public Ksp<MType, ValueType> {
 public:
  using base = Ksp<MType, ValueType>;

  inline int  get_resids_length() const { return base::_resids.size(); }
  inline void get_resids(double *r) const {
    for (int i = 0; i < get_resids_length(); ++i) r[i] = base::_resids[i];
  }

  inline void check_pars() { base::_check_pars(); }

  inline std::pair<int, size_type> solve(const size_type n, const int *rowptr,
                                         const int *colind, const double *vals,
                                         const double *b, double *x,
                                         const int  kernel = ksp::TRADITION,
                                         const bool with_init_guess = false,
                                         const bool verbose = true) const {
    constexpr static bool WRAP = true;

    const py_crs_type A(n, n, const_cast<int *>(rowptr),
                        const_cast<int *>(colind), const_cast<double *>(vals),
                        WRAP);
#ifndef NDEBUG
    A.check_validity();
#endif
    const py_array_type bb(n, const_cast<double *>(b), WRAP);
    py_array_type       xx(n, x, WRAP);
    return base::solve(A, bb, xx, kernel, with_init_guess, verbose);
  }
};

using PyFGMRES           = KspAdapt<ksp::FGMRES>;      // fgmres
using PyFQMRCGSTAB       = KspAdapt<ksp::FQMRCGSTAB>;  // fqmrcgstab
using PyFGMRES_Mixed     = KspAdapt<ksp::FGMRES, PyHILUCSI_Mixed, double>;
using PyFQMRCGSTAB_Mixed = KspAdapt<ksp::FQMRCGSTAB, PyHILUCSI_Mixed, double>;

}  // namespace hilucsi

#endif  // _HILUCSI_PYTHON_HILUCSI4PY_HPP
