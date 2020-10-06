///////////////////////////////////////////////////////////////////////////////
//                This file is part of HILUCSI project                       //
///////////////////////////////////////////////////////////////////////////////

/*!
 * \file hilucsi/alg/PivotCrout.hpp
 * \brief Implementation of modified \a Crout update in deferred fashion
 *        by using augmented compressed data structures with pivoting
 * \author Qiao Chen

\verbatim
Copyright (C) 2020 NumGeom Group at Stony Brook University

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

#ifndef _HILUCSI_ALG_PIVOTCROUT_HPP
#define _HILUCSI_ALG_PIVOTCROUT_HPP

#include <algorithm>
#include <cstddef>
#include <type_traits>

#include "hilucsi/alg/Crout.hpp"

namespace hilucsi {

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

  /// \brief dual of the \ref compute_l, i.e. computing the row of U
  /// \tparam ScaleArray row scaling diagonal matrix type, see \ref Array
  /// \tparam CrsType crs format, see \ref CRS
  /// \tparam PermType permutation matrix type, see \ref Array
  /// \tparam PosArray position array type, see \ref Array
  /// \tparam DiagType diagonal matrix type, see \ref Array
  /// \tparam AugCcsType ccs format, see \ref AugCCS
  /// \tparam AugCrsType crs format, see \ref AugCRS
  /// \tparam SpVecType work array for current row vector, see \ref SparseVector
  /// \param[in] s row scaling matrix from preprocessing
  /// \param[in] crs_A input matrix in crs scheme
  /// \param[in] t column scaling matrix from preprocessing
  /// \param[in] pk permutated row index
  /// \param[in] q column inverse permutation (deferred)
  /// \param[in] L lower part
  /// \param[in] d diagonal vector
  /// \param[in] U upper part
  /// \param[in] U_start leading row positions of current \ref _step
  /// \param[out] ut current row vector of U
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
  /// Regarding the complexity cost, it takes
  /// \f$\mathcal{O}(\textrm{nnz}(\mathbf{A}_{p_k,:}))\f$ to first load the
  /// row of A to \f$\mathbf{u}_k^T\f$, and takes
  /// \f$\mathcal{O}(\cup_j \textrm{nnz}(\mathbf{U}_{j,:}))\f$, where
  /// \f$l_{kj}\neq 0\f$ for computing the vector matrix operation.
  template <class ScaleArray, class CrsType, class PermType, class PosArray,
            class DiagType, class AugCcsType, class AugCrsType, class SpVecType>
  void compute_ut(const ScaleArray &s, const CrsType &crs_A,
                  const ScaleArray &t, const size_type pk, const PermType &q,
                  const AugCcsType &L, const DiagType &d, const AugCrsType &U,
                  const PosArray &U_start, SpVecType &ut) const {
    // compilation checking
    static_assert(CrsType::ROW_MAJOR, "input A must be CRS for loading ut");
    using index_type                = typename PosArray::value_type;
    constexpr static index_type nil = static_cast<index_type>(-1);

    // reset sparse buffer
    ut.reset_counter();

    // first load the A row
    _load_A2ut(s, crs_A, t, pk, q, ut);

    // if not first step
    if (_step) {
      // get the starting row ID with deferring
      index_type aug_id = L.start_row_id(deferred_step());
      while (!L.is_nil(aug_id)) {
        // get the column index
        const size_type col_idx = L.col_idx(aug_id);
        hilucsi_assert(
            col_idx < _step,
            "compute_ut column index %zd should not exceed step %zd for L",
            col_idx, _step);
        hilucsi_assert(col_idx < U_start.size(), "%zd exceeds the U_start size",
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
          hilucsi_assert(idx > deferred_step(),
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
  /// \tparam PosArray position array type, see \ref Array
  /// \tparam AugCcsType ccs format, see \ref AugCCS
  /// \tparam DiagType array used for storing diagonal array, see \ref Array
  /// \tparam AugCrsType crs format, see \ref AugCRS
  /// \tparam SpVecType sparse vector for storing l, see \ref SparseVector
  /// \param[in] s row scaling matrix from preprocessing
  /// \param[in] ccs_A input matrix in ccs scheme
  /// \param[in] t column scaling matrix from preprocessing
  /// \param[in] p row inverse permutation matrix (deferred)
  /// \param[in] qk permutated column index
  /// \param[in] m leading block size
  /// \param[in] L lower part of decomposition
  /// \param[in] L_start position array storing the starting locations of \a L
  /// \param[in] d diagonal entries
  /// \param[in] U augmented U matrix
  /// \param[out] l column vector of L at current \ref _step
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
  /// \f$\mathcal{O}(\textrm{nnz}(\mathbf{A}_{:,q_k}))\f$ to first load the
  /// column of A to \f$\mathbf{\ell}_k\f$, and takes
  /// \f$\mathcal{O}(\cup_i \textrm{nnz}(\mathbf{L}_{:,i}))\f$, where
  /// \f$u_{ik}\neq 0\f$ for computing the matrix vector operation.
  template <class ScaleArray, class CcsType, class PermType, class PosArray,
            class AugCcsType, class DiagType, class AugCrsType, class SpVecType>
  void compute_l(const ScaleArray &s, const CcsType &ccs_A, const ScaleArray &t,
                 const PermType &p, const size_type qk, const size_type m,
                 const AugCcsType &L, const PosArray &L_start,
                 const DiagType &d, const AugCrsType &U, SpVecType &l) const {
    // compilation checking
    static_assert(!CcsType::ROW_MAJOR, "input A must be CCS for loading l");
    using index_type                = typename PosArray::value_type;
    constexpr static index_type nil = static_cast<index_type>(-1);

    // clear sparse counter
    l.reset_counter();

    // load A column
    _load_A2l<false>(s, ccs_A, t, p, qk, m, l);

    // if not first step
    if (_step) {
      // get the deferred column handle
      index_type aug_id = U.start_col_id(deferred_step());
      while (!U.is_nil(aug_id)) {
        // get the row index
        const size_type row_idx = U.row_idx(aug_id);
        hilucsi_assert(
            row_idx < _step,
            "compute_ut row index %zd should not exceed step %zd for U",
            row_idx, _step);
        hilucsi_assert(row_idx < L_start.size(), "%zd exceeds the L_start size",
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
          hilucsi_assert(idx > deferred_step(),
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
    using index_type                = typename PosArray::value_type;
    constexpr static index_type nil = static_cast<index_type>(-1);

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
    start[_step] = 0;
  }

  /// \brief estimate the inverse norm with augmented ds
  /// \tparam AugCsType Augmented compressed storage, see \ref AugCRS or
  ///                   \ref AugCCS
  /// \tparam KappaArray Inverse norm array type, see \ref Array
  /// \param[in] T Augmented compressed storage input
  /// \param[in,out] kappa history of inverse norms
  template <class AugCsType, class KappaArray>
  inline bool update_kappa(const AugCsType &T, KappaArray &kappa) {
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
    index_type aug_id = T.start_aug_id(deferred_step());
    while (!T.is_nil(aug_id)) {
      const index_type primary_idx = T.primary_idx(aug_id);
      hilucsi_assert((size_type)primary_idx < kappa.size(),
                     "%zd exceeds the solution size", (size_type)primary_idx);
      hilucsi_assert((size_type)primary_idx < _step,
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
};
}  // namespace hilucsi

#endif  // _HILUCSI_ALG_PIVOTCROUT_HPP