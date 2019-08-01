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
#include <type_traits>
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

// abstract interface for solver
class KspSolver {
 public:
  virtual ~KspSolver() {}
  virtual void                      set_rtol(const double)           = 0;
  virtual double                    get_rtol() const                 = 0;
  virtual void                      set_restart(const int)           = 0;
  virtual int                       get_restart() const              = 0;
  virtual void                      set_maxit(const size_type)       = 0;
  virtual size_type                 get_maxit() const                = 0;
  virtual void                      set_inner_steps(const size_type) = 0;
  virtual size_type                 get_inner_steps() const          = 0;
  virtual void                      set_lamb1(const double)          = 0;
  virtual double                    get_lamb1() const                = 0;
  virtual void                      set_lamb2(const double)          = 0;
  virtual double                    get_lamb2() const                = 0;
  virtual void                      check_pars()                     = 0;
  virtual int                       get_resids_length() const        = 0;
  virtual void                      get_resids(double *r) const      = 0;
  virtual std::pair<int, size_type> solve(const size_type n, const int *rowptr,
                                          const int *colind, const double *vals,
                                          const double *b, double *x,
                                          const int  kernel,
                                          const bool with_init_guess,
                                          const bool verbose) const  = 0;
};

// using a template base for Ksp solver
template <template <class, class> class Ksp, class MType = PyHILUCSI,
          class ValueType = void>
class KspAdapt : public Ksp<MType, ValueType>, public KspSolver {
 public:
  using base = Ksp<MType, ValueType>;
  virtual ~KspAdapt() {}

  virtual void   set_rtol(const double v) override final { base::rtol = v; }
  virtual double get_rtol() const override final { return base::rtol; }
  virtual void   set_restart(const int v) override final { base::restart = v; }
  virtual int    get_restart() const override final { return base::restart; }
  virtual void set_maxit(const size_type v) override final { base::maxit = v; }
  virtual size_type get_maxit() const override final { return base::maxit; }
  virtual void      set_inner_steps(const size_type v) override final {
    base::inner_steps = v;
  }
  virtual size_type get_inner_steps() const override final {
    return base::inner_steps;
  }
  virtual void   set_lamb1(const double v) override final { base::lamb1 = v; }
  virtual double get_lamb1() const override final { return base::lamb1; }
  virtual void   set_lamb2(const double v) override final { base::lamb2 = v; }
  virtual double get_lamb2() const override final { return base::lamb2; }

  virtual int get_resids_length() const override final {
    return base::_resids.size();
  }
  virtual void get_resids(double *r) const override final {
    for (int i = 0; i < get_resids_length(); ++i) r[i] = base::_resids[i];
  }

  virtual void check_pars() override final { base::_check_pars(); }

  virtual std::pair<int, size_type> solve(
      const size_type n, const int *rowptr, const int *colind,
      const double *vals, const double *b, double *x, const int kernel,
      const bool with_init_guess, const bool verbose) const override final {
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
using PyFBICGSTAB        = KspAdapt<ksp::FBICGSTAB>;   // fbicgstab
using PyFGMRES_Mixed     = KspAdapt<ksp::FGMRES, PyHILUCSI_Mixed, double>;
using PyFQMRCGSTAB_Mixed = KspAdapt<ksp::FQMRCGSTAB, PyHILUCSI_Mixed, double>;
using PyFBICGSTAB_Mixed  = KspAdapt<ksp::FBICGSTAB, PyHILUCSI_Mixed, double>;
using PyTGMRESR          = KspAdapt<ksp::TGMRESR>;
using PyTGMRESR_Mixed    = KspAdapt<ksp::TGMRESR, PyHILUCSI_Mixed, double>;

}  // namespace hilucsi

#endif  // _HILUCSI_PYTHON_HILUCSI4PY_HPP
