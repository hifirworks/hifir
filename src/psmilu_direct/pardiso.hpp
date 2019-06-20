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
  using value_type = typename CrsType::value_type;  ///< value type
  using size_type  = typename CrsType::size_type;   ///< size type
  using index_type = typename CrsType::index_type;  ///< index type

  constexpr static bool CONSIST_INT = sizeof(MKL_INT) == sizeof(index_type);
  ///< flag to check if \a index_type and MKL_INT are consistent
  constexpr static bool ONE_BASED = CrsType::ONE_BASED;  ///< index base

  /// \brief default constructor
  ///
  /// Will will first call \a pardisoinit with matrix type 11, i.e. real
  /// nonsymmetric systems. Then, we set \a _iparm[0] != 0 to indicate pardiso
  /// using user parameters. There are several parameters we need to manually
  /// set: 1) iparm[34] is for index base system, 0 for Fortran, 1 for C. 2)
  /// iparm[26] will be set to 1 for debug build. 3) iparm[27] indicates the
  /// input data type, 0 for \a double, 1 for \a float. 4) iparm[5] enforces to
  /// be 1 to overwrite the solution directly in \a b.
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
    _iparm[5] = 1;  // let solver write solution to b
  }

  Pardiso(const Pardiso&) = default;
  Pardiso(Pardiso&&)      = default;

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

  /// \brief check the error code return
  inline int info() const { return _error; }

  /// \brief query the total number of nonzeros in the LU patterns
  ///
  /// Based on MKL documentation, this information is stored in index 17 of the
  /// pardiso parameters
  inline size_type nnz() const { return _iparm[17]; }

  /// \brief empty checking
  inline bool empty() const { return _empty; }

  /// \brief move the matrix in, i.e. the input will be destroyed if it's lvalue
  /// \param[in,out] A input matrix, on output, it will be empty!
  inline void move_matrix(CrsType&& A) {
    _A     = std::move(A);
    _empty = false;
    _free();
    _ia = ensure_type_consistency<MKL_INT>(_A.row_start());
    _ja = ensure_type_consistency<MKL_INT>(_A.col_ind());
    _buf.resize(_A.nrows());
    psmilu_error_if(_buf.status() == DATA_UNDEF, "memory allocation failed");
  }

  /// \brief indicate pardiso to do analysis and factorization phases
  ///
  /// Analysis involves symbolic factorization, which will computes the sparsity
  /// pattern. After this stage, the total memory usage is known, and it's okay
  /// to go to next stage, which computes the numerical factorization.
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

  /// \brief solve with a single rhs
  /// \tparam ArrayType array type, should be \ref Array
  /// \param[in,out] x input as rhs and output as solution
  template <class ArrayType>
  inline void solve(ArrayType& x) {
    psmilu_error_if(_empty, "solver is empty!");
    psmilu_error_if(x.size() != _A.nrows(), "inconsistent sizes");
    _error         = 0;
    MKL_INT phase  = 33;  // solve
    MKL_INT maxfct = 1, nrhs = 1, n(_A.nrows());
    pardiso(_pt, &maxfct, &_mnum, &_mtype, &phase, &n, _A.vals().data(), _ia,
            _ja, nullptr, &nrhs, _iparm, &_msglvl, x.data(), _buf.data(),
            &_error);
    psmilu_warning_if(_error, "pardiso returned %d on phase 33", (int)_error);
  }

  /// \brief get the backend solver name
  inline static const char* backend() { return "MKL-PARDISO"; }

 private:
  void*   _pt[64];  ///< handle, should not touch
  MKL_INT _error;   ///< error code
  MKL_INT _mnum;    ///< factorization number should be 1
  MKL_INT _mtype;   ///< matrix type, should be 11, general systems
  bool    _empty;   ///< empty flag
  MKL_INT _msglvl;  ///< printing message flag, will be on in debug build

 protected:
  MKL_INT           _iparm[64];  ///< pardiso parameters
  CrsType           _A;          ///< input matrix
  MKL_INT*          _ia;         ///< pointer to the row_start
  MKL_INT*          _ja;         ///< pointer to the col_ind
  Array<value_type> _buf;        ///< buffer used in solving phase

  /// \brief safely free the workspace if \b needed
  inline void _free() {
    if (!CONSIST_INT) {
      if (_ia) delete[] _ia;
      if (_ja) delete[] _ja;
    }
  }
};

}  // namespace psmilu

#endif  // _PSMILU_DIRECT_PARDISO_HPP