///////////////////////////////////////////////////////////////////////////////
//                  This file is part of the HIFIR library                   //
///////////////////////////////////////////////////////////////////////////////

/*!
 * \file hif/alg/PivotCrout.hpp
 * \brief Implementation of modified \a Crout update in deferred fashion
 *        by using augmented compressed data structures with pivoting
 * \author Qiao Chen

\verbatim
Copyright (C) 2020 NumGeom Group at Stony Brook University

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

#ifndef _HIF_ALG_PIVOTCROUT_HPP
#define _HIF_ALG_PIVOTCROUT_HPP

#include <algorithm>
#include <cstddef>
#include <type_traits>
#include <utility>

#include "hif/alg/Crout.hpp"
#include "hif/utils/common.hpp"

namespace hif {

/// \class PivotCrout
/// \brief Crout update in deferred fashion with pivoting
/// \ingroup crout
///
/// The idea of this class is to implement Crout kernels that wrap around a
/// common factor---step. In this case, the class Crout is as simple (in terms
/// of data size) as an integer. It also supports implicitly casting to integer
/// as well as increment operations (both suffix and prefix versions). The whole
/// disign can be demonstrate by the following C++ pseudocode:
///
/// \code{.cpp}
/// for (Crout crout; crout < m; ++crout) {
///   crout.do_work1(...);
///   crout.do_work2(...);
///   ...
/// }
/// \endcode
class PivotCrout : public Crout {
 public:
  using size_type = Crout::size_type;  ///< size type

  /// \brief compute diagonal entry
  /// \tparam ScaleArray row/column scaling diagonal matrix type, see \ref Array
  /// \tparam CcsType ccs format, see \ref CCS
  /// \tparam PermType permutation matrix type, see \ref Array
  /// \tparam AugCcsType ccs format, see \ref AugCCS
  /// \tparam PosArray position array type, see \ref Array
  /// \tparam DiagType array used for storing diagonal array, see \ref Array
  /// \tparam AugCrsType crs format, see \ref AugCRS
  /// \param[in] s row scaling matrix from preprocessing
  /// \param[in] ccs_A input matrix in ccs scheme
  /// \param[in] t column scaling matrix from preprocessing
  /// \param[in] p_inv row inverse permutation (with tail indices)
  /// \param[in] qk permutated column index (with deferring)
  /// \param[in] L lower part of decomposition
  /// \param[in] L_start position array storing the starting locations of \a L
  /// \param[in] d diagonal entries
  /// \param[in] U augmented U matrix
  /// \return diagonal entry value
  /// \sa compute_ut, compute_l
  ///
  /// This routine computes the current diagonal entry of \f$\mathbf{D}\f$
  /// Mathematically, this routine is to compute:
  ///
  /// \f[
  ///   d_{k}=\hat{A}[p_{k},q_{k}]-\mathbf{L}_{k,1:k-1}\mathbf{D}_{k-1}
  ///       \mathbf{U}_{1:k-1,k}
  /// \f]
  template <class ScaleArray, class CcsType, class PermType, class AugCcsType,
            class PosArray, class DiagType, class AugCrsType>
  inline typename DiagType::value_type compute_dk(
      const ScaleArray &s, const CcsType &ccs_A, const ScaleArray &t,
      const PermType &p_inv, const size_type qk, const AugCcsType &L,
      const PosArray &L_start, const DiagType &d, const AugCrsType &U) const {
    using value_type = typename DiagType::value_type;
    using index_type = typename AugCrsType::index_type;

    value_type dk(0);
    // load diagonal from the coefficient matrix A
    do {
      const size_type defer_thres = deferred_step();
      // qk is c index
      auto       v_itr = ccs_A.val_cbegin(qk);
      auto       i_itr = ccs_A.row_ind_cbegin(qk);
      const auto t_qk  = t[qk];
      for (auto last = ccs_A.row_ind_cend(qk); i_itr != last;
           ++i_itr, ++v_itr) {
        const auto      A_idx = *i_itr;
        const size_type idx   = p_inv[A_idx];
        // push to the sparse vector only if its in range _step+1:n
        if (idx == defer_thres) {
          dk = s[A_idx] * *v_itr * t_qk;  // load diagonal entry
          break;
        }
      }
    } while (false);

    // compute the LDU part
    if (_step) {
      // get the deferred column handle
      index_type aug_id = U.start_col_id(deferred_step());
      while (!U.is_nil(aug_id)) {
        // get the row index
        const size_type row_idx = U.row_idx(aug_id);
        hif_assert(row_idx < _step,
                   "compute_ut row index %zd should not exceed step %zd for U",
                   row_idx, _step);
        hif_assert(row_idx < L_start.size(), "%zd exceeds the L_start size",
                   row_idx);
        // compute d*U
        const auto du = d[row_idx] * U.val_from_col_id(aug_id);
        // get the starting position from L_start
        auto L_v_itr = L.val_cbegin(row_idx) + L_start[row_idx];
        auto L_i_itr = L.row_ind_cbegin(row_idx) + L_start[row_idx];
        if (L_i_itr != L.row_ind_cend(row_idx) &&
            size_type(*L_i_itr) == deferred_step())
          dk -= du * *L_v_itr;  // diagonal part
        // advance augmented handle
        aug_id = U.next_col_id(aug_id);
      }  // while
    }
    return dk;
  }

  /// \brief dual of the \ref compute_l, i.e. computing the row of U
  /// \tparam ScaleArray row scaling diagonal matrix type, see \ref Array
  /// \tparam CrsType crs format, see \ref CRS
  /// \tparam PermType permutation matrix type, see \ref Array
  /// \tparam AugCcsType ccs format, see \ref AugCCS
  /// \tparam DiagType diagonal matrix type, see \ref Array
  /// \tparam AugCrsType crs format, see \ref AugCRS
  /// \tparam PosArray position array type, see \ref Array
  /// \tparam SpVecType work array for current row vector, see \ref SparseVector
  /// \param[in] s row scaling matrix from preprocessing
  /// \param[in] crs_A input matrix in crs format
  /// \param[in] t column scaling matrix from preprocessing
  /// \param[in] pk permuted row index (with deferring)
  /// \param[in] q_inv inverse column permutation (with tails indices)
  /// \param[in] L lower part
  /// \param[in] d diagonal vector
  /// \param[in] U upper part
  /// \param[in] U_start leading row positions of current \ref _step
  /// \param[in,out] ut upon input, it contains preloaded row in A via
  ///                   load_arow; upon output, it contains the factor of
  ///                   \f$\mathbf{u}_k^T\f$.
  /// \sa compute_l
  ///
  /// This routine computes the current row vector of \f$\mathbf{U}\f$
  /// (w/o diagonal scaling). Mathematically, this routine is to compute:
  ///
  /// \f[
  ///   \mathbf{u}_{k}^{T}=
  ///     \hat{\mathbf{A}}[p_{k},\mathbf{q}_{k+1:n}]-
  ///     \mathbf{L}_{k,1:k-1}\mathbf{D}_{k-1}
  ///       \mathbf{U}_{1:k-1,k+1:n}
  /// \f]
  ///
  /// It's worth noting that, conceptally, the formula above is nothing but
  /// a vector matrix operation. However, standard implementation won't give
  /// good performance (especially with consideration of cache performance),
  /// this is because \f$\mathbf{L}\f$ is stored in column major whereas
  /// row major for \f$\mathbf{U}\f$. Therefore, the actual implementation
  /// is in the fashion that loops through \f$\mathbf{U}\f$ in row major
  /// while keeping \f$\mathbf{L}\f$ as much static as possible.
  ///
  /// Regarding computation cost, it takes
  /// \f$\mathcal{O}(\cup_j \textrm{nnz}(\mathbf{U}_{j,:}))\f$, where
  /// \f$l_{kj}\neq 0\f$ for computing the vector matrix operation.
  template <class ScaleArray, class CrsType, class PermType, class AugCcsType,
            class DiagType, class AugCrsType, class PosArray, class SpVecType>
  void compute_ut(const ScaleArray &s, const CrsType &crs_A,
                  const ScaleArray &t, const size_type pk,
                  const PermType &q_inv, const AugCcsType &L, const DiagType &d,
                  const AugCrsType &U, const PosArray &U_start,
                  SpVecType &ut) const {
    // compilation checking
    using index_type = typename PosArray::value_type;

    // reset sparse buffer
    ut.reset_counter();

    // first load the A row
    _load_arow(s, crs_A, t, pk, q_inv, ut);

    // if not first step
    if (_step) {
      // get the starting row ID with deferring
      index_type aug_id = L.start_row_id(deferred_step());
      while (!L.is_nil(aug_id)) {
        // get the column index
        const size_type col_idx = L.col_idx(aug_id);
        hif_assert(
            col_idx < _step,
            "compute_ut column index %zd should not exceed step %zd for L",
            col_idx, _step);
        hif_assert(col_idx < U_start.size(), "%zd exceeds the U_start size",
                   col_idx);
        // compute L*d
        const auto ld = L.val_from_row_id(aug_id) * d[col_idx];
        // get the starting position from U_start
        auto U_v_itr  = U.val_cbegin(col_idx) + U_start[col_idx];
        auto U_i_itr  = U.col_ind_cbegin(col_idx) + U_start[col_idx],
             U_i_last = U.col_ind_cend(col_idx);
        if (U_i_itr != U_i_last && size_type(*U_i_itr) == deferred_step()) {
          // advance to offsets
          ++U_i_itr;
          ++U_v_itr;
        }
        // for loop to go thru all entries in U
        for (; U_i_itr != U_i_last; ++U_i_itr, ++U_v_itr) {
          const auto idx = *U_i_itr;
          hif_assert(idx > deferred_step(),
                     "U index %zd in computing ut should greater than step "
                     "%zd(defers:%zd)",
                     idx, _step, _defers);
          if (ut.push_back(idx, _step))
            ut.vals()[idx] = -ld * *U_v_itr;
          else
            ut.vals()[idx] -= ld * *U_v_itr;
        }
        // advance augmented handle
        aug_id = L.next_row_id(aug_id);
      }  // while
    }
  }

  /// \brief compute the column vector of L of current step
  /// \tparam ScaleArray row/column scaling diagonal matrix type, see \ref Array
  /// \tparam CcsType ccs format, see \ref CCS
  /// \tparam PermType permutation matrix type, see \ref Array
  /// \tparam AugCcsType ccs format, see \ref AugCCS
  /// \tparam PosArray position array type, see \ref Array
  /// \tparam DiagType array used for storing diagonal array, see \ref Array
  /// \tparam AugCrsType crs format, see \ref AugCRS
  /// \tparam SpVecType sparse vector for storing l, see \ref SparseVector
  /// \param[in] s row scaling matrix from preprocessing
  /// \param[in] ccs_A input matrix in ccs scheme
  /// \param[in] t column scaling matrix from preprocessing
  /// \param[in] p_inv row inverse permutation (with tail indices)
  /// \param[in] qk permutated column index (with deferring)
  /// \param[in] L lower part of decomposition
  /// \param[in] L_start position array storing the starting locations of \a L
  /// \param[in] d diagonal entries
  /// \param[in] U augmented U matrix
  /// \param[in,out] l upon input, it contains preloaded column in A via
  ///                  load_acol; upon output, it contains the factor of
  ///                  \f$\mathbf{\ell}_k\f$.
  /// \sa compute_ut
  ///
  /// This routine computes the current column vector of \f$\mathbf{L}\f$
  /// (w/o diagonal scaling). Mathematically, this routine is to compute:
  ///
  /// \f[
  ///   \mathbf{\ell}_{k}=
  ///     \hat{\mathbf{A}}[\mathbf{p}_{k+1:n},q_{k}]-
  ///     \mathbf{L}_{k+1:n,1:k-1}\mathbf{D}_{k-1}
  ///       \mathbf{U}_{1:k-1,k}
  /// \f]
  ///
  /// It's worth noting that, conceptally, the formula above is nothing but
  /// a matrix vector operation. However, standard implementation won't give
  /// good performance (especially with consideration of cache performance),
  /// this is because \f$\mathbf{L}\f$ is stored in column major whereas
  /// row major for \f$\mathbf{U}\f$. Therefore, the actual implementation
  /// is in the fashion that loops through \f$\mathbf{L}\f$ in column major
  /// while keeping \f$\mathbf{U}\f$ as much static as possible.
  ///
  /// Regarding the complexity cost, it takes
  /// \f$\mathcal{O}(\cup_i \textrm{nnz}(\mathbf{L}_{:,i}))\f$, where
  /// \f$u_{ik}\neq 0\f$ for computing the matrix vector operation.
  template <class ScaleArray, class CcsType, class PermType, class AugCcsType,
            class PosArray, class DiagType, class AugCrsType, class SpVecType>
  void compute_l(const ScaleArray &s, const CcsType &ccs_A, const ScaleArray &t,
                 const PermType &p_inv, const size_type qk, const AugCcsType &L,
                 const PosArray &L_start, const DiagType &d,
                 const AugCrsType &U, SpVecType &l) const {
    // compilation checking
    using index_type              = typename PosArray::value_type;
    static constexpr bool IS_SYMM = false;  // never symmetric for pivoting code

    // clear sparse counter
    l.reset_counter();

    // load A column
    _load_acol<IS_SYMM>(s, ccs_A, t, p_inv, qk, size_type(0), l);

    // if not first step
    if (_step) {
      // get the deferred column handle
      index_type aug_id = U.start_col_id(deferred_step());
      while (!U.is_nil(aug_id)) {
        // get the row index
        const size_type row_idx = U.row_idx(aug_id);
        hif_assert(row_idx < _step,
                   "compute_ut row index %zd should not exceed step %zd for U",
                   row_idx, _step);
        hif_assert(row_idx < L_start.size(), "%zd exceeds the L_start size",
                   row_idx);
        // compute d*U
        const auto du = d[row_idx] * U.val_from_col_id(aug_id);
        // get the starting position from L_start
        auto L_v_itr  = L.val_cbegin(row_idx) + L_start[row_idx];
        auto L_i_itr  = L.row_ind_cbegin(row_idx) + L_start[row_idx],
             L_i_last = L.row_ind_cend(row_idx);
        if (L_i_itr != L_i_last && size_type(*L_i_itr) == deferred_step()) {
          ++L_i_itr;
          ++L_v_itr;
        }
        for (; L_i_itr != L_i_last; ++L_i_itr, ++L_v_itr) {
          // convert to c index
          const auto idx = *L_i_itr;
          hif_assert(idx > deferred_step(),
                     "L index %zd in computing l should greater than step "
                     "%zd(defers:%zd)",
                     idx, _step, _defers);
          // compute this entry, if index does not exist, assign new value to
          // -L*d*u, else, -= L*d*u
          if (l.push_back(idx, _step))
            l.vals()[idx] = -du * *L_v_itr;
          else
            l.vals()[idx] -= *L_v_itr * du;
        }
        // advance augmented handle
        aug_id = U.next_col_id(aug_id);
      }  // while
    }
  }

  /// \brief compress L and U and update their corresponding starting positions
  /// \tparam CsType either \ref AugCRS or \ref AugCCS
  /// \tparam PosArray position array type, see \ref Array
  /// \param[in,out] T strictly lower (L) or upper (U) matrices
  /// \param[in,out] start local positions
  /// \note This routine essentially compresses deferred_step to _step
  /// \note All other routines should be called before this one, and
  ///       deferred_step therein are conceptually the current Crout step
  template <class CsType, class PosArray>
  inline void update_compress(CsType &T, PosArray &start) const {
    using index_type = typename PosArray::value_type;

    // get the starting augmented ID
    index_type aug_id = T.start_aug_id(deferred_step());
    while (!T.is_nil(aug_id)) {
      // get corresponding primary index, i.e., row index in CRS while column
      // index for CCS
      const index_type primary_idx = T.primary_idx(aug_id);
      auto             itr = T.ind_begin(primary_idx) + start[primary_idx];
      *itr -= _defers;       // compress
      ++start[primary_idx];  // increment next position
      // advance augment handle
      aug_id = T.next_aug_id(aug_id);
    }
    // need to update newly added entry at the end
    start[_step]                = 0;
    T.secondary_start()[_step]  = T.secondary_start()[deferred_step()];
    T.secondary_end()[_step]    = T.secondary_end()[deferred_step()];
    T.secondary_counts()[_step] = T.secondary_counts()[deferred_step()];
  }

  /// \brief estimate the inverse norm with augmented ds
  /// \tparam AugCsType Augmented compressed storage, see \ref AugCRS or
  ///                   \ref AugCCS
  /// \tparam KappaArray Inverse norm array type, see \ref Array
  /// \param[in] T Augmented compressed storage input
  /// \param[in,out] kappa history of inverse norms
  /// \param[in] entry (optional) concatenate \a entry to T_{k-1} for computing
  ///                  the absolute conditioning of leading block of T, default
  ///                  value is k, i.e., \a _step
  template <class AugCsType, class KappaArray>
  inline bool update_kappa(const AugCsType &T, KappaArray &kappa,
                           const size_type entry = 0u) {
    using value_type          = typename AugCsType::value_type;
    using index_type          = typename AugCsType::index_type;
    using size_type           = typename AugCsType::size_type;
    constexpr static bool one = true, neg_one = false;

    if (!_step) {
      kappa[0] = value_type(1);
      return one;
    }

    // we need to loop through all entries in row step
    value_type sum(0);

    // start augmented ID
    index_type aug_id = T.start_aug_id(entry ? entry : deferred_step());
    while (!T.is_nil(aug_id)) {
      const index_type primary_idx = T.primary_idx(aug_id);
      hif_assert((size_type)primary_idx < kappa.size(),
                 "%zd exceeds the solution size", (size_type)primary_idx);
      hif_assert((size_type)primary_idx < _step,
                 "the matrix U should only contain the strict upper case");
      sum += kappa[primary_idx] * T.val_from_aug_id(aug_id);
      // advance augment handle
      aug_id = T.next_aug_id(aug_id);
    }
    const value_type k1 = value_type(1) - sum, k2 = value_type(-1) - sum;
    if (std::abs(k1) < std::abs(k2)) {
      kappa[_step] = k2;
      return neg_one;
    }
    kappa[_step] = k1;
    return one;
  }

  /// \brief defer an secondary entry, i.e., rows in CCS and columns in CRS
  /// \tparam CsType either \ref AugCRS (U) or \ref AugCCS (L)
  /// \param[in] to_idx deferred destination
  /// \param[in,out] T lower or upper factors
  /// \ingroup defer
  template <class CsType>
  inline void defer_entry(const size_type to_idx, CsType &T) const {
    T.defer_entry(deferred_step(), to_idx);
  }

  /// \brief interchagen with pivot for pivoting with Crout LU
  /// \tparam AugCsType Augmented compressed data types, see \ref AugCRS or
  ///         \ref AugCCS
  /// \tparam PermType Permutation array type, see \ref BiPermMatrix
  /// \tparam InvPermType Inverse permutation array type
  /// \tparam DiagArrayType Diagonal array type, see \ref Array
  /// \tparam SpVecType Array type for storing current row or column of A
  /// \param[in] pivot Pivoting position
  /// \param[in,out] T Either lower or upper factors from Crout LU and step k
  /// \param[in,out] p Permutation array
  /// \param[in,out] p_inv Inverse permutation mapping of \a p
  /// \param[in,out] d diagonal array
  /// \param[in,out] a_k k-th row or column of A (scaled and permuted)
  template <class AugCsType, class PermType, class InvPermType,
            class DiagArrayType, class SpVecType>
  inline void interchange(const size_type pivot, AugCsType &T, PermType &p,
                          InvPermType &p_inv, DiagArrayType &d,
                          SpVecType &a_k) const {
    // swap permutation
    std::swap(p[deferred_step()], p[pivot]);
    std::swap(p_inv[p[deferred_step()]], p_inv[p[pivot]]);
    // interchanage in the L or U factors
    T.interchange_entries(deferred_step(), pivot);
    // swap current diagonal with the pivot entry in k-th row or column of A
    std::swap(d[deferred_step()], a_k[pivot]);
  }

  /// \brief Apply inverse-based rook pivoting
  /// \tparam ScaleArray Row/column scaling array, see \ref Array
  /// \tparam PermType Permutation array type
  /// \tparam PermType2 Inverse permutation array type
  /// \tparam CcsType Ccs storage type, see \ref CCS
  /// \tparam CrsType Crs storage type, see \ref CRS
  /// \tparam CroutInfoStreamer Crout update information streamer
  /// \tparam PosArray Integer array type for storing position information
  /// \tparam PivotUnaryOpL Unary predictor for finding pivot in L
  /// \tparam PivotUnaryOpU Unary predictor for finding pivot in U
  /// \tparam AugCcsType Augmented ccs storage type, see \ref AugCCS
  /// \tparam DiagType Diagonal array type, see \ref Array
  /// \tparam AugCrsType Augmented crs storage type, see \ref AugCRS
  /// \tparam SpVecType Sparse vector workspace, see \ref SparseVector
  /// \param[in] s row scaling vector
  /// \param[in] t column scaling vector
  /// \param[in] A_ccs input coefficient matrix in CCS
  /// \param[in] A_crs input coefficient matrix in CRS
  /// \param[in] m leading dimension size
  /// \param[in] gamma thresholding for rook pivoting
  /// \param[in] max_steps maximum number of pivoting steps
  /// \param[in] Crout_info information streamer (empty for non-verbose mode)
  /// \param[in] L_start starting positions in columns for \a L
  /// \param[in] U_start starting positions in rows for \a U
  /// \param[in] op_l unary operator for finding a pivot in L
  /// \param[in] op_ut unary operator for finding a pivot in U
  /// \param[in,out] p row permutation array, will be updated due to pivoting
  /// \param[in,out] p_inv inverse of row permutation, will be updated
  /// \param[in,out] q column permutation array, will be updated due to pivoting
  /// \param[in,out] q_inv inverse of column permutation, will be updated
  /// \param[in,out] L lower factor, will be permuted if necessary
  /// \param[in,out] d diagonal factor (vector), will be updated for current
  /// step \param[in,out] U upper factor, will be permuted if ncessary
  /// \param[out] l sparse vector work space for computing \f$\mathbf{\ell}_k\f$
  /// \param[out] ut sparse vector work space for computing \f$\mathbf{u}_k^T\f$
  /// \return number of pivots in row and column swappings.
  /// \sa interchange
  ///
  /// Regarding \a op_l and \a op_u, we find a pivot in L if and only if it
  /// satisfies op_l condition, i.e., op_l(pivot_l)==true.
  template <class ScaleArray, class PermType, class PermType2, class CcsType,
            class CrsType, class CroutInfoStreamer, class PosArray,
            class PivotUnaryOpL, class PivotUnaryOpU, class AugCcsType,
            class DiagType, class AugCrsType, class SpVecType>
  inline std::pair<int, int> apply_thres_pivot(
      const ScaleArray &s, const ScaleArray &t, const CcsType &A_ccs,
      const CrsType &A_crs, const size_type m, const double gamma,
      const int max_steps, const CroutInfoStreamer &Crout_info,
      const PosArray &L_start, const PosArray &U_start,
      const PivotUnaryOpL &op_l, const PivotUnaryOpU &op_ut, PermType &p,
      PermType2 &p_inv, PermType &q, PermType2 &q_inv, AugCcsType &L,
      DiagType &d, AugCrsType &U, SpVecType &l, SpVecType &ut) const {
    static_assert(!CcsType::ROW_MAJOR, "must be CCS");
    static_assert(CrsType::ROW_MAJOR, "must be CRS");
    using index_type = typename CcsType::index_type;

    int        pivot_step(0), col_pivots(0), row_pivots(0);
    const auto k = deferred_step();  // current step (with gap considerred)
    for (; pivot_step < max_steps; ++pivot_step) {
      bool l_pivot = false;

      //--------------
      // row pivoting
      //--------------

      // update Crout step for L_k
      // important! we need to reset dense flags in l
      if (pivot_step) l.restore_cur_state();
      compute_l(s, A_ccs, t, p_inv, q[k], L, L_start, d, U, l);
      // sort indices based on magnitude
      l.sort_indices([&](const index_type a, const index_type b) {
        return std::abs(l[a]) > std::abs(l[b]);
      });
      // find pivot
      const auto pivot_col = l.find_if(op_l);
      // compute diagonal
      d[k] = compute_dk(s, A_ccs, t, p_inv, q[k], L, L_start, d, U);
      if (pivot_col != std::numeric_limits<size_type>::max()) {
        hif_assert(pivot_col > k && pivot_col < m,
                   "wrong pivot range (pvt=%zd), (k,m)=(%zd,%zd)", pivot_col, k,
                   m);
        if (std::abs(d[k]) <
            gamma * std::max(std::abs(l[pivot_col]), std::abs(d[k]))) {
          l_pivot = true;
          Crout_info(
              "  interchanging rows current=%zd (deferrals=%zd) and pivot=%zd",
              k, _defers, size_type(pivot_col));
          // column pivoting occurs
          ++col_pivots;  // increment counter
          // swap
          interchange(pivot_col, L, p, p_inv, d, l);
        } else if (pivot_step > 0)
          break;  // give row pivot a chance at first step
      } else if (pivot_step > 0)
        break;

      //-----------------
      // column pivoting
      //-----------------

      // update Crout step for U_k^T
      // important! we need to reset dense flags in ut
      if (pivot_step) ut.restore_cur_state();
      compute_ut(s, A_crs, t, p[k], q_inv, L, d, U, U_start, ut);
      // sort indices based on magnitude
      ut.sort_indices([&](const index_type a, const index_type b) {
        return std::abs(ut[a]) > std::abs(ut[b]);
      });
      // find pivot
      const auto pivot_row = ut.find_if(op_ut);
      // compute diagonal if necessary
      if (l_pivot)
        d[k] = compute_dk(s, A_ccs, t, p_inv, q[k], L, L_start, d, U);
      if (pivot_row != std::numeric_limits<size_type>::max()) {
        hif_assert(pivot_row > k && pivot_row < m,
                   "wrong pivot range (pvt=%zd), (k,m)=(%zd,%zd)", pivot_row, k,
                   m);
        // d may be updated
        if (std::abs(d[k]) <
            gamma * std::max(std::abs(ut[pivot_row]), std::abs(d[k]))) {
          Crout_info(
              "  interchanging cols current=%zd (deferrals=%zd) and pivot=%zd",
              k, _defers, size_type(pivot_row));
          // row pivoting occurs
          ++row_pivots;
          // swap
          interchange(pivot_row, U, q, q_inv, d, ut);
        } else
          break;
      } else
        break;
    }
    if (pivot_step > max_steps)
      hif_warning("  could not satisfy pivoting requirement in %d iterations",
                  max_steps);
    // Crout_info("  could not satisfy pivoting requirement in %d iterations",
    //            max_steps);
    return std::make_pair(col_pivots, row_pivots);
  }
};
}  // namespace hif

#endif  // _HIF_ALG_PIVOTCROUT_HPP