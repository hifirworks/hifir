///////////////////////////////////////////////////////////////////////////////
//                  This file is part of the HIFIR library                   //
///////////////////////////////////////////////////////////////////////////////

/*!
 * \file hif/sparse_direct/mumps.hpp
 * \brief Interface to MUMPS >= 5.1.2 for last level sparse direct solver
 * \author Qiao Chen

\verbatim
Copyright (C) 2021 NumGeom Group at Stony Brook University

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
\endverbatim

 */

#ifndef _HIF_SPARSEDIRECT_MUMPS_HPP
#define _HIF_SPARSEDIRECT_MUMPS_HPP

#include <algorithm>
#include <tuple>
#include <type_traits>

#include <dmumps_c.h>  // double precision
#include <smumps_c.h>  // single precision

#include "hif/ds/Array.hpp"
#include "hif/utils/log.hpp"

namespace hif {
namespace internal {

/*!
 * \addtogroup sparse
 * @{
 */

/// \class MumpsStructTrait
/// \brief trait to select \a ?MUMPS_STRUCT_C
/// \tparam ScalarType scalar type used, e.g. \a double
/// \note Currently, we only support real matrices
template <typename ScalarType>
class MumpsStructTrait {
 public:
  using struct_type = typename std::conditional<
      std::is_same<ScalarType, DMUMPS_COMPLEX>::value, DMUMPS_STRUC_C,
      typename std::conditional<std::is_same<ScalarType, SMUMPS_COMPLEX>::value,
                                SMUMPS_STRUC_C, void>::type>::type;
  ///< type to \a ?MUMPS_STRUCT_C

  static_assert(!std::is_void<struct_type>::value, "unsupported scalar type");

  /// \brief overload mumps interface for \a double
  inline static void call_mumps(DMUMPS_STRUC_C &job) { dmumps_c(&job); }

  /// \brief overload mumps interface for \a float
  inline static void call_mumps(SMUMPS_STRUC_C &job) { smumps_c(&job); }
};

/// \brief convert \ref CRS to \a assembly \a format
/// \tparam ValueType value type
/// \tparam CrsType crs matrix type
/// \param[in] A input matrix
/// \return \a tuple of row, column indices, and values
///
/// HIF uses the assembly format in MUMPS, which is aka coordinate format
/// or IJV format. It contains three nnz arrays, where the row and column
/// indices are stored in pair. In addition, MUMPS uses Fortran index.
template <class ValueType, class CrsType>
inline std::tuple<Array<MUMPS_INT>, Array<MUMPS_INT>, Array<ValueType>>
convert2ijv(const CrsType &A) {
  static_assert(CrsType::ROW_MAJOR, "must be CRS");

  using size_type  = typename CrsType::size_type;
  using value_type = ValueType;

  const size_type n = A.nrows();
  hif_error_if(n != A.ncols(), "must be squared matrix");
  Array<MUMPS_INT>  rows, cols;
  Array<value_type> vs;
  const size_type   nnz = A.nnz();
  if (n) {
    hif_error_if(!nnz, "cannot access matrix of zeros");
    rows.resize(nnz);
    hif_error_if(rows.status() == DATA_UNDEF, "memory allocation failed");
    cols.resize(nnz);
    hif_error_if(cols.status() == DATA_UNDEF, "memory allocation failed");
    vs.resize(nnz);
    hif_error_if(vs.status() == DATA_UNDEF, "memory allocation failed");
    size_type idx(0);
    for (size_type i(0); i < n; ++i) {
      auto            v_itr = A.val_cbegin(i);
      const MUMPS_INT row(i + 1);
      for (auto itr = A.col_ind_cbegin(i), last = A.col_ind_cend(i);
           itr != last; ++itr, ++v_itr, ++idx) {
        rows[idx] = row;
        cols[idx] = *itr + 1;
        vs[idx]   = *v_itr;
      }
    }
    hif_assert(idx == nnz, "fatal");
  }
  return std::make_tuple(rows, cols, vs);
}

/*!
 * @}
 */

}  // namespace internal

/// \class MUMPS
/// \brief interface for using MUMPS solvers
/// \tparam ScalarType scalar type used, e.g. \a double
/// \ingroup sparse
template <class ScalarType>
class MUMPS {
  typedef internal::MumpsStructTrait<ScalarType> _trait;  ///< trait type
 public:
  typedef ScalarType scalar_type;  ///< scalar

  /// \brief check backend
  inline static const char *backend() { return "MUMPS"; }

  /// \brief constructor
  /// \param[in] verbose enable verbose printing, default is \a true
  /// \param[in] threads threads used, default is 1
  ///
  /// In constructor, we initialize the solver with default options, then
  /// we modify some utility configurations based on runtime or compilation
  /// options.
  explicit MUMPS(const bool verbose = true, int threads = 1) {
    _handle.par          = 1;        // host has data
    _handle.sym          = 0;        // general systems
    _handle.job          = -1;       // initialization stage
    _handle.comm_fortran = -987654;  // communicator for MPI build
    _trait::call_mumps(_handle);
    set_info(verbose, threads);
    // apply at most 3 iterative refinement
    _handle.icntl[9] = 3;
  }

  /// \brief set information streaming and threads
  /// \param[in] blr BLR feature, default is 0 (disbled)
  /// \param[in] blr_tol absolute tolerance of blr, default is 0 (full-rank)
  /// \param[in] threads threads used, default is 1
  /// \note See mumps user manual on control parameters ICNTL(35) and CNTL(7)
  inline void set_info(const int blr = 0, const double blr_tol = 0.0,
                       int threads = 1) const {
    if (true) _handle.icntl[0] = _handle.icntl[2] = 0;
    // query the warning
    if (!warn_flag()) _handle.icntl[3] = 1;  // only error
    threads           = threads <= 0 ? 1 : threads;
    _handle.icntl[15] = threads;
    _handle.icntl[34] = blr < 0 ? 0 : blr;
    _handle.cntl[6]   = blr_tol < 0.0 ? 0.0 : blr_tol;
  }

  /// \brief destructor, free out all internal memory used by MUMPS
  ~MUMPS() {
    _handle.job = -2;
    _trait::call_mumps(_handle);
  }

  /// \brief check empty
  inline bool empty() const { return _handle.job < 0; }

  /// \brief check nnz
  ///
  /// Please refer to MUMPS documentation of \a INFOG
  inline std::size_t nnz() const {
    if (_handle.job <= 0) return 0;
    MUMPS_INT nz = _handle.infog[8];  // value entries
    if (nz >= 0) return nz;
    nz = -nz;
    return std::size_t(1000000ul) * nz;
  }

  /// \brief check info return
  inline int info() const { return _handle.infog[0]; }

  /// \brief factorization
  /// \tparam CrsType input matrix in CRS type, see \ref CRS
  /// \param[in] A input \ref CRS matrix
  template <class CrsType>
  inline void factorize(const CrsType &A) const {
    // convert to ijv
    std::tie(_rows, _cols, _vs) = internal::convert2ijv<scalar_type>(A);

    _handle.job = 4;  // symbolic + factorization
    _handle.n   = A.nrows();
    _handle.nnz = _rows.size();
    _handle.irn = _rows.data();
    _handle.jcn = _cols.data();
    _handle.a   = _vs.data();

    _handle.infog[0] = 0;

    _trait::call_mumps(_handle);

    hif_error_if(info() != 0, "MUMPS did not succeed, INFO(1)=%d, INFO(2)=%d",
                 _handle.infog[0], _handle.infog[1]);
  }

  /// \brief solve with rhs inplace
  /// \tparam ArrayType array type, see \ref Array
  /// \param[in,out] x rhs on input and solution on output
  /// \param[in] tran (optional) tranpose flag, default is false
  template <class ArrayType>
  inline void solve(ArrayType &x, const bool tran = false) const {
    static constexpr bool SAME_TYPE =
        std::is_same<typename ArrayType::value_type, scalar_type>::value;
    hif_assert(_handle.job > 0, "invalid solver stage");
    hif_assert(x.size() == _handle.n, "mismatched sizes");
    if (!SAME_TYPE) {
      _x.resize(x.size());
      std::copy(x.cbegin(), x.cend(), _x.begin());
    }
    _handle.job      = 3;
    _handle.rhs      = SAME_TYPE ? (scalar_type *)x.data() : _x.data();
    _handle.infog[0] = 0;
    _handle.icntl[8] = tran ? 2 : 1;  // transpose
    _trait::call_mumps(_handle);
    if (!SAME_TYPE) std::copy(_x.cbegin(), _x.cend(), x.begin());
  }

 protected:
  mutable typename _trait::struct_type _handle;  ///< job handle/structure
  mutable Array<MUMPS_INT>             _rows;    ///< row indices
  mutable Array<MUMPS_INT>             _cols;    ///< column indices
  mutable Array<ScalarType>            _vs;      ///< values
  mutable Array<ScalarType>            _x;       ///< solution buffer
};

}  // namespace hif

#endif  // _HIF_SPARSEDIRECT_MUMPS_HPP