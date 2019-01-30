//@HEADER
//----------------------------------------------------------------------------
//                Copyright (C) 2019 The PSMILU AUTHORS
//----------------------------------------------------------------------------
//@HEADER

/// \file psmilu_Schur.hpp
/// \brief Routines for computing Schur complements for both C and H versions
/// \authors Qiao,

#ifndef _PSMILU_SCHUR_HPP
#define _PSMILU_SCHUR_HPP

#include "psmilu_SparseVec.hpp"
#include "psmilu_log.hpp"
#include "psmilu_utils.hpp"

namespace psmilu {

/// \brief compute the C-version of Schur complement
/// \tparam CrsType crs matrix used for input, see \ref CRS
/// \tparam PermType permutation vector type, see \ref BiPermMatrix
/// \tparam L_AugCcsType ccs augmented for lower part, see \ref AugCCS
/// \tparam DiagType diagonal vector type, see \ref Array
/// \tparam U_CrsType augmented crs for upper part, see \ref AugCRS
/// \tparam U_StartType starting position type for U, see \ref Array
/// \param[in] A input crs matrix
/// \param[in] p row permutation vector
/// \param[in] q column permutation vector
/// \param[in] m leading block size of current level
/// \param[in] n input matrix A size
/// \param[in] L lower part after \ref Crout update
/// \param[in] d diagonal entry that contains the leading block
/// \param[in] U upper part after \ref Crout update
/// \param[in] U_start staring positions for rows in U
/// \param[in] SC Schur complement of C version
/// \ingroup schur
///
/// Mathematically, this is to compute:
///
/// \f[
///   \boldsymbol{S}_C=\boldsymbol{A}[\boldsymbol{p}_{m+1:n},\boldsymbol{q}_{m+1:n}]
///     -\boldsymbol{L}_E\textrm{diag}(\boldsymbol{d_B})\boldsymbol{U}_F
/// \f]
///
/// Where \f$\boldsymbol{L}_E=\boldsymbol{L}_{m+1:n,:}\f$,
/// \f$\boldsymbol{U}_F=\boldsymbol{U}_{:,m+1:n}\f$. Notice that both
/// \f$\boldsymbol{L}\f$ and \f$\boldsymbol{U}\f$ are lower and upper parts
/// \b after computing Crout updates. Therefore, they are rectangle matrices
/// if \f$m<n\f$. Accessing \f$\boldsymbol{U}_{:,m+1:n}\f$ can be efficiently
/// achieved with \a U_start vector, which is given after Crout updates.
///
/// This computation involves sparse matrix multiplication, which is done in
/// three steps: 1) determine the total number of nonzeros, 2) determine the
/// index pattern (symbolic computation), and 3) fill in the numerical values.
///
/// \note We allocate size n-m \ref SparseVector as buffer.
template <class CrsType, class PermType, class L_AugCcsType, class DiagType,
          class U_CrsType, class U_StartType>
inline void compute_Schur_C(const CrsType &A, const PermType &p,
                            const PermType &                  q,
                            const typename CrsType::size_type m,
                            const typename CrsType::size_type n,
                            const L_AugCcsType &L, const DiagType &d,
                            const U_CrsType &U, const U_StartType &U_start,
                            CrsType &SC) {
  using index_type                = typename CrsType::index_type;
  using value_type                = typename CrsType::value_type;
  using size_type                 = typename CrsType::size_type;
  constexpr static bool ONE_BASED = CrsType::ONE_BASED;
  using sparse_vec_type = SparseVector<value_type, index_type, ONE_BASED>;
  static_assert(!(ONE_BASED ^ L_AugCcsType::ONE_BASED),
                "inconsistent index base");
  static_assert(!(ONE_BASED ^ U_CrsType::ONE_BASED), "inconsistent index base");
  static_assert(CrsType::ROW_MAJOR, "input A and output SC must be row major");
  static_assert(U_CrsType::ROW_MAJOR, "input upper part must be row major");
  static_assert(!L_AugCcsType::ROW_MAJOR,
                "input lower part must be column major");

  if (m == n) return;
  psmilu_assert(m < n, "m %zd should less than n %zd", m, n);

  // helper routine for index conversion
  const auto c_idx = [](const size_type i) -> size_type {
    return to_c_idx<size_type, ONE_BASED>(i);
  };

  // get the offset size
  const size_type N = n - m;
  // create sparse vector buffer
  sparse_vec_type buf(N);
  SC.resize(N, N);
  auto &row_start = SC.row_start();
  row_start.resize(N + 1);
  row_start.front() = ONE_BASED;
  auto U_ind_first  = U.col_ind().cbegin();

  //------------------------------
  // step 1, determine buffer size
  //------------------------------

  for (size_type i = 0u; i < N; ++i) {
    // reset buffer counter
    buf.reset_counter();

    // get the row index in global system
    index_type aug_id = L.start_row_id(i + m);
    // low thru all entries in this row
    while (!L.is_nil(aug_id)) {
      // get the corresponding column index
      const size_type j = L.col_idx(aug_id);
#ifdef PSMILU_SCHUR_C_USE_FULL
      // mainly for unit testing
      if (j >= m) break;
#else
      psmilu_assert(j < m, "invalid L matrix");
#endif  // PSMILU_SCHUR_C_FULL
      auto first = U_ind_first + U_start[j];
      auto last  = U.col_ind_cend(j);
      // for each nonzero entries in U_{j,:}, push/register the column
      // index in SC size.
      // NOTE that *itr-m is C index is *itr is C. or Fortran if *itr is F
      for (auto itr = first; itr != last; ++itr) {
        psmilu_assert(c_idx(*itr) >= m, "%zd should point to offset of U",
                      c_idx(*itr));
        buf.push_back(*itr - m, i);
      }
      aug_id = L.next_row_id(aug_id);  // advance augment handle
    }
    // get the row index in A
    const size_type pi = p[m + i];
    // since A is permutated, we must loop through all entries in this row
    for (auto itr = A.col_ind_cbegin(pi), last = A.col_ind_cend(pi);
         itr != last; ++itr) {
      const size_type inv_q = q.inv(c_idx(*itr));
      if (inv_q >= m) buf.push_back(inv_q + ONE_BASED - m, i);
    }
    // assemble row_start
    row_start[i + 1] = row_start[i] + buf.size();
  }

  //-----------------------------------
  // step 2, fill in the column indices
  //-----------------------------------

  // NOTE that we reserve spaces for both values and column indices
  SC.reserve(row_start[N]);
  auto &col_ind = SC.col_ind();
  col_ind.resize(0);  // just reset the internal counter, O(1) for Array
  for (size_type i = 0u; i < N; ++i) {
    buf.reset_counter();  // reset counter

    // NOTE that in order to reuse the sparse vector w/o resetting all dense
    // tags, we must use a unique tag
    const size_type uniq_tag = i + N;
    // get row in global
    index_type aug_id = L.start_row_id(i + m);
    // loop thru all entries in this row
    while (!L.is_nil(aug_id)) {
      const size_type j = L.col_idx(aug_id);
#ifdef PSMILU_SCHUR_C_USE_FULL
      // mainly for unit testing
      if (j >= m) break;
#endif  // PSMILU_SCHUR_C_FULL
      auto first = U_ind_first + U_start[j];
      auto last  = U.col_ind_cend(j);
      for (auto itr = first; itr != last; ++itr)
        buf.push_back(*itr - m, uniq_tag);
      aug_id = L.next_row_id(aug_id);
    }
    // same thing as in step 1
    const size_type pi = p[m + i];
    for (auto itr = A.col_ind_cbegin(pi), last = A.col_ind_cend(pi);
         itr != last; ++itr) {
      const size_type inv_q = q.inv(c_idx(*itr));
      if (inv_q >= m) buf.push_back(inv_q + ONE_BASED - m, uniq_tag);
    }

    // NOTE that we must enforce the indices are sorted, because SC will
    // potentially be the input for next level. During the construction of
    // column indices, we didn't maintain a sorted list, thus we sort here
    buf.sort_indices();
    size_type j = col_ind.size();    // current start index
    col_ind.resize(j + buf.size());  // resize
    for (auto ind_itr = buf.inds().cbegin(); j < col_ind.size(); ++j, ++ind_itr)
      col_ind[j] = *ind_itr;
  }

  psmilu_assert(col_ind.size() == (size_type)row_start[N], "fatal error!");

  //---------------------------
  // step 3, fill in the values
  //---------------------------

  auto &buf_vals    = buf.vals();  // value array in sparse vector
  auto &vals        = SC.vals();  // value array in SC, memory already allocated
  auto  U_val_first = U.vals().cbegin();
  vals.resize(row_start[N]);  // This just reset the size counter for Array
  for (size_type i = 0u; i < N; ++i) {
    // 1, zero out all values in SC_i, based on the indices we just built
    // in step 2
    auto sc_ind_first = SC.col_ind_cbegin(i);
    for (auto itr = sc_ind_first, ind_last = SC.col_ind_cend(i);
         itr != ind_last; ++itr) {
      psmilu_assert(c_idx(*itr) < N, "%zd exceed value size", c_idx(*itr));
      buf_vals[c_idx(*itr)] = 0.0;
    }

    // 2, load the corresponding row in A to the buf_vals
    const size_type pi        = p[m + i];
    auto            a_val_itr = A.val_cbegin(pi);
    for (auto itr = A.col_ind_cbegin(pi), last = A.col_ind_cend(pi);
         itr != last; ++itr, ++a_val_itr) {
      const size_type inv_q = q.inv(c_idx(*itr));
      if (inv_q >= m) buf_vals[inv_q - m] = *a_val_itr;
    }

    // 3. compute -L*D*U, store it to the value buffer in sparse vector
    index_type aug_id = L.start_row_id(i + m);
    while (!L.is_nil(aug_id)) {
      const size_type j = L.col_idx(aug_id);
#ifdef PSMILU_SCHUR_C_USE_FULL
      // mainly for unit testing
      if (j >= m) break;
#endif  // PSMILU_SCHUR_C_FULL
      auto             first     = U_ind_first + U_start[j];
      auto             last      = U.col_ind_cend(j);
      auto             u_val_itr = U_val_first + U_start[j];
      const value_type ld        = L.val_from_row_id(aug_id) * d[j];  // L*D
      for (auto itr = first; itr != last; ++itr, ++u_val_itr)
        buf_vals[c_idx(*itr - m)] -= ld * *u_val_itr;
      aug_id = L.next_row_id(aug_id);
    }

    // 4. assign values directly to SC based on the column indices
    auto sc_val_itr = SC.val_begin(i);
    for (auto itr = sc_ind_first, ind_last = SC.col_ind_cend(i);
         itr != ind_last; ++itr, ++sc_val_itr)
      *sc_val_itr = buf_vals[c_idx(*itr)];
  }
}
}  // namespace psmilu

#endif  // _PSMILU_SCHUR_HPP
