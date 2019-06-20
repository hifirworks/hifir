//@HEADER
//----------------------------------------------------------------------------
//                Copyright (C) 2019 The PSMILU AUTHORS
//----------------------------------------------------------------------------
//@HEADER

/// \file psmilu_direct/pardiso.hpp
/// \brief PARDISO interface
/// \authors Qiao,

#ifndef _PSMILU_DIRECT_PARDISO_HPP
#define _PSMILU_DIRECT_PARDISO_HPP

#include <utility>

#include "psmilu_Array.hpp"
#include "psmilu_matching/common.hpp"

#include <mkl_pardiso.h>

namespace psmilu {

/// \class Pardiso
/// \brief MKL pardiso wrapper
/// \tparam CrsType crs type, see \ref CRS
/// \ingroup sss
/// \waring Currently, only a single pardiso instance is allowed, need to
///         extend to have multiple in MT environment
template <class CrsType>
class Pardiso {
  static_assert(CrsType::ROW_MAJOR, "must be CRS");

 public:
  using value_type = typename CrsType::value_type;
  using size_type  = typename CrsType::size_type;
  using index_type = typename CrsType::index_type;

  constexpr static bool CONSIST_INT = sizeof(MKL_INT) == sizeof(index_type);
  constexpr static bool ONE_BASED   = CrsType::ONE_BASED;

  Pardiso() {
    _mtype = 11;                        // real and general
    pardisoinit(_pt, &_mtype, _iparm);  // set default parameters
    _iparm[0] = 1;                      // no solver default
    _error    = 0;
    // set index base, 0 for Fortran, 1 for C
    _iparm[34] = !ONE_BASED;
#ifndef NDEBUG
    _iparm[26] = 1;
#endif
    // IMPORTANT, assume real systems
    if (sizeof(value_type) == sizeof(float)) _iparm[27] = 1;
    _mnum  = 1;
    _empty = true;
    _ia = _ja = nullptr;

#ifndef NDEBUG
    _msglvl = 1;
#else
    _msglvl = 0;
#endif
  }

  ~Pardiso() {
    _error         = 0;
    MKL_INT phase  = -1;  // free
    MKL_INT maxfct = 1, nrhs = 1, n(_A.nrows());
    pardiso(_pt, &maxfct, &_mnum, &_mtype, &phase, &n, _A.vals().data(), _ia,
            _ja, nullptr, &nrhs, _iparm, &_msglvl, nullptr, nullptr, &_error);
    _free();
    _empty = true;
    _ia = _ja = nullptr;
  }

  inline int       info() const { return _error; }
  inline size_type nnz() const { return _iparm[17]; }

  inline void move_matrix(CrsType&& A) {
    _A     = std::move(A);
    _empty = false;
    _free();
    _ia = ensure_type_consistency<MKL_INT>(_A.row_start());
    _ja = ensure_type_consistency<MKL_INT>(_A.col_ind());
  }

  inline void factorize() {
    psmilu_error_if(_empty, "the system is empty!");
    _error         = 0;
    MKL_INT phase  = 11;  // symbolic
    MKL_INT maxfct = 1, nrhs = 1, n(_A.nrows());
    pardiso(_pt, &maxfct, &_mnum, &_mtype, &phase, &n, _A.vals().data(), _ia,
            _ja, nullptr, &nrhs, _iparm, &_msglvl, nullptr, nullptr, &_error);
    if (_error) {
      psmilu_warning("pardiso returned %d on phase 11", (int)_error);
      return;
    }
    phase = 22;  // fac
    pardiso(_pt, &maxfct, &_mnum, &_mtype, &phase, &n, _A.vals().data(), _ia,
            _ja, nullptr, &nrhs, _iparm, &_msglvl, nullptr, nullptr, &_error);
    if (_error) {
      psmilu_warning("pardiso returned %d on phase 22", (int)_error);
      return;
    }
  }

  template <class ArrayType>
  inline void solve(const ArrayType& b, ArrayType& x) {
    psmilu_error_if(_empty, "solver is empty!");
    psmilu_error_if(x.size() != b.size(), "inconsistent sizes");
    psmilu_error_if(x.size() != _A.nrows(), "inconsistent sizes");
    _error         = 0;
    MKL_INT phase  = 33;  // solve
    MKL_INT maxfct = 1, nrhs = 1, n(_A.nrows());
    pardiso(_pt, &maxfct, &_mnum, &_mtype, &phase, &n, _A.vals().data(), _ia,
            _ja, nullptr, &nrhs, _iparm, &_msglvl, (void*)b.data(), x.data(),
            &_error);
    psmilu_warning_if(_error, "pardiso returned %d on phase 33", (int)_error);
  }

 private:
  void*   _pt[64];  ///< handle, should not touch
  MKL_INT _error;
  MKL_INT _mnum;   ///< factorization number should be 1
  MKL_INT _mtype;  ///< matrix type, should be 11, general systems
  bool    _empty;  ///< empty flag
  MKL_INT _msglvl;

 protected:
  MKL_INT  _iparm[64];
  CrsType  _A;
  MKL_INT* _ia;
  MKL_INT* _ja;

  inline void _free() {
    if (!CONSIST_INT) {
      if (_ia) delete[] _ia;
      if (_ja) delete[] _ja;
    }
  }
};

}  // namespace psmilu

#endif  // _PSMILU_DIRECT_PARDISO_HPP