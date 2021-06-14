///////////////////////////////////////////////////////////////////////////////
//                  This file is part of the HIFIR library                   //
///////////////////////////////////////////////////////////////////////////////

/*!
 * \file hif/arch/mkl_trsv.hpp
 * \brief MKL optimized triangular solve
 * \author Qiao Chen

\verbatim
Copyright (C) 2021 NumGeom Group at Stony Brook University

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
\endverbatim

 */

#ifndef _HIF_ARCH_MKLTRSV_HPP
#define _HIF_ARCH_MKLTRSV_HPP

#include <algorithm>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>
#ifndef DOXYGEN_SHOULD_SKIP_THIS
#  include <mkl_spblas.h>
#endif  // DOXYGEN_SHOULD_SKIP_THIS
#include "hif/ds/Array.hpp"
#include "hif/utils/log.hpp"

// NOTE: customized for HIF, we only use unit diagonal CRS, in addition,
// we need not to worry about transpose operation

namespace hif {
namespace internal {

inline const char *mkl_status_name(const sparse_status_t status) {
  switch (status) {
    case SPARSE_STATUS_SUCCESS:
      return "SPARSE_STATUS_SUCCESS";
    case SPARSE_STATUS_NOT_INITIALIZED:
      return "SPARSE_STATUS_NOT_INITIALIZED";
    case SPARSE_STATUS_ALLOC_FAILED:
      return "SPARSE_STATUS_ALLOC_FAILED";
    case SPARSE_STATUS_INVALID_VALUE:
      return "SPARSE_STATUS_INVALID_VALUE";
    case SPARSE_STATUS_EXECUTION_FAILED:
      return "SPARSE_STATUS_EXECUTION_FAILED";
    case SPARSE_STATUS_INTERNAL_ERROR:
      return "SPARSE_STATUS_INTERNAL_ERROR";
    case SPARSE_STATUS_NOT_SUPPORTED:
      return "SPARSE_STATUS_NOT_SUPPORTED";
  }
  return "MKL_STATUS_UNKOWN";
}

}  // namespace internal

/// \class MKL_SpTrSolver
/// \brief Triangular solver with MKL enhanced on CPU
/// \tparam ValueType value type, either \a double or \a float
/// \tparam IndexType index type, better to be MKL_INT (int)
/// \ingroup mt
template <class ValueType, class IndexType>
class MKL_SpTrSolver {
  static_assert(std::is_floating_point<ValueType>::value &&
                    sizeof(ValueType) <= 8u,
                "must be double/float");

 public:
  using index_type      = IndexType;                        ///< index type
  using value_type      = ValueType;                        ///< value type
  using index_array     = Array<index_type>;                ///< array type
  using value_array     = Array<value_type>;                ///< value array
  using mkl_index_array = std::vector<MKL_INT>;             ///< mkl array type
  using size_type       = typename index_array::size_type;  ///< size type

  /// \brief default constructor
  MKL_SpTrSolver() : _handle(nullptr) {}

  MKL_SpTrSolver(MKL_SpTrSolver &&)      = default;
  MKL_SpTrSolver(const MKL_SpTrSolver &) = default;

  /// \brief destructor
  ~MKL_SpTrSolver() {
    if (_handle) mkl_sparse_destroy(_handle);
    _handle = nullptr;
  }

  /// \brief setup
  /// \param[in] rowptr row pointer
  /// \param[in] colind column indices
  /// \param[in] vals values
  inline void setup(const index_array &rowptr, const index_array &colind,
                    const value_array &vals) {
    _create_mkl_sparse(rowptr, colind, vals.data());
  }

  /// \brief solve as strict lower
  /// \tparam RhsType right-hand size type
  /// \param[in,out] y input and output of right-hand size and solution, resp
  template <class RhsType>
  inline void solve_as_strict_lower(RhsType &y) const {
    // copy to work space
    std::copy_n(&y[0], _work.size(), _work.begin());
    _solve_lower(_work.data(), &y[0]);
  }

  /// \brief solve as strict upper
  /// \tparam RhsType right-hand size type
  /// \param[in,out] y input and output of right-hand size and solution, resp
  template <class RhsType>
  inline void solve_as_strict_upper(RhsType &y) const {
    // copy to work space
    std::copy_n(&y[0], _work.size(), _work.begin());
    _solve_upper(_work.data(), &y[0]);
  }

  // interface compatibility
  inline size_type nrows() const { return _work.size(); }
  inline size_type ncols() const { return _work.size(); }
  inline bool      empty() const {
    if (!_handle) return true;
    return !_work.size();
  }

  /// \brief optimize the solver with hints
  /// \tparam IsUpper flag indicating upper or lower cases
  /// \param[in] expected_calls expected calls, default is max in \a MKL_INT
  template <bool IsUpper>
  inline void optimize(
      const MKL_INT expected_calls = std::numeric_limits<MKL_INT>::max()) {
    // pre-tabulated constant flags
    static const sparse_operation_t no_tran    = SPARSE_OPERATION_NON_TRANSPOSE;
    static const matrix_descr       upper_desc = {.type =
                                                SPARSE_MATRIX_TYPE_TRIANGULAR,
                                            .mode = SPARSE_FILL_MODE_UPPER,
                                            .diag = SPARSE_DIAG_UNIT},
                              lower_desc       = {
                                  .type = SPARSE_MATRIX_TYPE_TRIANGULAR,
                                  .mode = SPARSE_FILL_MODE_LOWER,
                                  .diag = SPARSE_DIAG_UNIT};
    const auto hint_status = mkl_sparse_set_sv_hint(
        _handle, no_tran, IsUpper ? upper_desc : lower_desc, expected_calls);
    hif_error_if(hint_status != SPARSE_STATUS_SUCCESS,
                 "MKL spblas returned error %s",
                 internal::mkl_status_name(hint_status));
    // optimize
    const auto opt_status = mkl_sparse_optimize(_handle);
    hif_error_if(opt_status != SPARSE_STATUS_SUCCESS,
                 "MKL spblas returned error %s",
                 internal::mkl_status_name(opt_status));
  }

 protected:
  /// \brief initialize workspace
  /// \param[in] rowptr external row pointer array
  /// \param[in] colind external column index array
  /// \note This routine will make shallow copies if \a index_type and MKL_INT
  ///       are compatible
  inline void _init_workspace(const index_array &rowptr,
                              const index_array &colind) {
    const auto n = rowptr.size() - 1;
    if (sizeof(index_type) == sizeof(MKL_INT)) {
      _rs_ptr = (MKL_INT *)rowptr.data();
      _re_ptr = _rs_ptr + 1;
      _ci_ptr = (MKL_INT *)colind.data();
    } else {
      // make a copy
      hif_info("copied!");
      _row_start.resize(n);
      std::copy_n(rowptr.cbegin(), n, _row_start.begin());
      _row_end.resize(n);
      std::copy_n(rowptr.cbegin() + 1, n, _row_end.begin());
      _col_ind.resize(colind.size());
      std::copy(colind.cbegin(), colind.cend(), _col_ind.begin());
      _rs_ptr = _row_start.data();
      _re_ptr = _row_end.data();
      _ci_ptr = _col_ind.data();
    }
    _work.resize(n);
  }

  /// \brief create for double precision
  /// \param[in] rowptr external row pointer array
  /// \param[in] colind external column index array
  /// \param[in] val external address to the value array
  inline void _create_mkl_sparse(const index_array &rowptr,
                                 const index_array &colind, const double *val) {
    this->~MKL_SpTrSolver();
    _init_workspace(rowptr, colind);
    const auto n = rowptr.size() - 1;
    const auto status =
        mkl_sparse_d_create_csr(&_handle, SPARSE_INDEX_BASE_ZERO, n, n, _rs_ptr,
                                _re_ptr, _ci_ptr, (double *)val);
    hif_error_if(status != SPARSE_STATUS_SUCCESS,
                 "MKL spblas returned error %s",
                 internal::mkl_status_name(status));
  }

  /// \brief create for single precision
  /// \param[in] rowptr external row pointer array
  /// \param[in] colind external column index array
  /// \param[in] val external address to the value array
  inline void _create_mkl_sparse(const index_array &rowptr,
                                 const index_array &colind, const float *val) {
    this->~MKL_SpTrSolver();
    _init_workspace(rowptr, colind);
    const auto n = rowptr.size() - 1;
    const auto status =
        mkl_sparse_s_create_csr(&_handle, SPARSE_INDEX_BASE_ZERO, n, n, _rs_ptr,
                                _re_ptr, _ci_ptr, (float *)val);
    hif_error_if(status != SPARSE_STATUS_SUCCESS,
                 "MKL spblas returned error %s",
                 internal::mkl_status_name(status));
  }

  /// \brief perform triangular upper solve as double precision
  /// \param[in] x right-hand size vector
  /// \param[in] y solution vector
  inline void _solve_upper(const double *x, double *y) const {
    // pre-tabulated constant flags
    static const sparse_operation_t no_tran    = SPARSE_OPERATION_NON_TRANSPOSE;
    static const matrix_descr       upper_desc = {
        .type = SPARSE_MATRIX_TYPE_TRIANGULAR,
        .mode = SPARSE_FILL_MODE_UPPER,
        .diag = SPARSE_DIAG_UNIT};
    const auto solve_status =
        mkl_sparse_d_trsv(no_tran, 1.0, _handle, upper_desc, x, y);
    hif_error_if(solve_status != SPARSE_STATUS_SUCCESS,
                 "MKL spblas returned error %s",
                 internal::mkl_status_name(solve_status));
  }

  /// \brief perform triangular upper solve as single precision
  /// \param[in] x right-hand size vector
  /// \param[in] y solution vector
  inline void _solve_upper(const float *x, float *y) const {
    // pre-tabulated constant flags
    static const sparse_operation_t no_tran    = SPARSE_OPERATION_NON_TRANSPOSE;
    static const matrix_descr       upper_desc = {
        .type = SPARSE_MATRIX_TYPE_TRIANGULAR,
        .mode = SPARSE_FILL_MODE_UPPER,
        .diag = SPARSE_DIAG_UNIT};
    const auto solve_status =
        mkl_sparse_s_trsv(no_tran, 1.0f, _handle, upper_desc, x, y);
    hif_error_if(solve_status != SPARSE_STATUS_SUCCESS,
                 "MKL spblas returned error %s",
                 internal::mkl_status_name(solve_status));
  }

  /// \brief perform triangular lower solve as double precision
  /// \param[in] x right-hand size vector
  /// \param[in] y solution vector
  inline void _solve_lower(const double *x, double *y) const {
    // pre-tabulated constant flags
    static const sparse_operation_t no_tran    = SPARSE_OPERATION_NON_TRANSPOSE;
    static const matrix_descr       lower_desc = {
        .type = SPARSE_MATRIX_TYPE_TRIANGULAR,
        .mode = SPARSE_FILL_MODE_LOWER,
        .diag = SPARSE_DIAG_UNIT};
    const auto solve_status =
        mkl_sparse_d_trsv(no_tran, 1.0, _handle, lower_desc, x, y);
    hif_error_if(solve_status != SPARSE_STATUS_SUCCESS,
                 "MKL spblas returned error %s",
                 internal::mkl_status_name(solve_status));
  }

  /// \brief perform triangular lower solve as single precision
  /// \param[in] x right-hand size vector
  /// \param[in] y solution vector
  inline void _solve_lower(const float *x, float *y) const {
    // pre-tabulated constant flags
    static const sparse_operation_t no_tran    = SPARSE_OPERATION_NON_TRANSPOSE;
    static const matrix_descr       lower_desc = {
        .type = SPARSE_MATRIX_TYPE_TRIANGULAR,
        .mode = SPARSE_FILL_MODE_LOWER,
        .diag = SPARSE_DIAG_UNIT};
    const auto solve_status =
        mkl_sparse_s_trsv(no_tran, 1.0f, _handle, lower_desc, x, y);
    hif_error_if(solve_status != SPARSE_STATUS_SUCCESS,
                 "MKL spblas returned error %s",
                 internal::mkl_status_name(solve_status));
  }

 protected:
  sparse_matrix_t                 _handle;     ///< internal handle
  mkl_index_array                 _row_start;  ///< starting row
  mkl_index_array                 _row_end;    ///< ending row
  mkl_index_array                 _col_ind;    ///< column index
  MKL_INT *                       _rs_ptr, *_re_ptr, *_ci_ptr;
  mutable std::vector<value_type> _work;  ///< work space

  // NOTE: the arrays are used to handle index type that is not MKL_INT
};

}  // namespace hif

#endif  // _HIF_ARCH_MKLTRSV_HPP
