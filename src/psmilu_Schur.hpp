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

#include "psmilu_Array.hpp"
#include "psmilu_SparseVec.hpp"
#include "psmilu_log.hpp"
#include "psmilu_utils.hpp"

namespace psmilu {

/// \brief compute the C-version of Schur complement
/// \tparam LeftDiagType row scaling vector type, see \ref Array
/// \tparam CrsType crs matrix used for input, see \ref CRS
/// \tparam RightDiagType column scaling vector type, see \ref Array
/// \tparam PermType permutation vector type, see \ref BiPermMatrix
/// \tparam L_AugCcsType ccs augmented for lower part, see \ref AugCCS
/// \tparam DiagType diagonal vector type, see \ref Array
/// \tparam U_CrsType augmented crs for upper part, see \ref AugCRS
/// \tparam U_StartType starting position type for U, see \ref Array
/// \param[in] s row scaling vector
/// \param[in] A input crs matrix
/// \param[in] t column scaling vector
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
/// \sa compute_Schur_H
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
template <class LeftDiagType, class CrsType, class RightDiagType,
          class PermType, class L_AugCcsType, class DiagType, class U_CrsType,
          class U_StartType>
inline void compute_Schur_C(const LeftDiagType &s, const CrsType &A,
                            const RightDiagType &t, const PermType &p,
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
#ifdef PSMILU_SCHUR_USE_FULL
      // mainly for unit testing
      if (j >= m) break;
#else
      psmilu_assert(j < m, "invalid L matrix");
#endif  // PSMILU_SCHUR_FULL
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

  if (!row_start[N]) {
    psmilu_warning("Schur (C version) complement becomes empty!");
    return;
  }

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
#ifdef PSMILU_SCHUR_USE_FULL
      // mainly for unit testing
      if (j >= m) break;
#endif  // PSMILU_SCHUR_FULL
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
    const auto      s_pi      = s[pi];  // row scaling
    for (auto itr = A.col_ind_cbegin(pi), last = A.col_ind_cend(pi);
         itr != last; ++itr, ++a_val_itr) {
      const size_type A_idx = c_idx(*itr);
      const size_type inv_q = q.inv(c_idx(A_idx));
      // load and scale
      if (inv_q >= m) buf_vals[inv_q - m] = s_pi * *a_val_itr * t[A_idx];
    }

    // 3. compute -L*D*U, store it to the value buffer in sparse vector
    index_type aug_id = L.start_row_id(i + m);
    while (!L.is_nil(aug_id)) {
      const size_type j = L.col_idx(aug_id);
#ifdef PSMILU_SCHUR_USE_FULL
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

namespace internal {

/// \brief compute the first part of H version, i.e. \f$\boldsymbol{T}_E\f$
/// \tparam L_AugCcsType augmented ccs type for L, see \ref AugCCS
/// \tparam L_StartType starting position after Crout update for L
/// \tparam L_B_CcsType lower part type, see \ref CCS
/// \tparam LeftDiagType row scaling vector type, see \ref Array
/// \tparam CcsType input and output matrix type, see \ref CCS
/// \tparam RightDiagType column scaling type, see \ref Array
/// \tparam PermType permutation matrix type, see \ref BiPermMatrix
/// \tparam DiagType diagonal vector type, see \ref Array
/// \tparam U_B_CcsType upper part type, see \ref CCS
/// \param[in] L_E whole lower part after \ref Crout update
/// \param[in] L_start position array after Crout update for lower part
/// \param[in] L_B lower part of leading block
/// \param[in] s row scaling vector
/// \param[in] A input matrix in CCS format
/// \param[in] t column scaling vector
/// \param[in] p row permutation matrix
/// \param[in] q column permutation matrix
/// \param[in] d diagonal entries (permutated)
/// \param[in] U_B upper part of leading block
/// \param[out] T_E computed first part of H version
/// \ingroup schur
/// \sa compute_Schur_H_T_F
template <class L_AugCcsType, class L_StartType, class L_B_CcsType,
          class LeftDiagType, class CcsType, class RightDiagType,
          class PermType, class DiagType, class U_B_CcsType>
inline void compute_Schur_H_T_E(const L_AugCcsType &L_E,
                                const L_StartType & L_start,
                                const L_B_CcsType &L_B, const LeftDiagType &s,
                                const CcsType &A, const RightDiagType &t,
                                const PermType &p, const PermType &q,
                                const DiagType &d, const U_B_CcsType &U_B,
                                CcsType &T_E) {
  using size_type                 = typename CcsType::size_type;
  using index_type                = typename CcsType::index_type;
  using value_type                = typename CcsType::value_type;
  constexpr static bool ONE_BASED = CcsType::ONE_BASED;
  using sp_vec_type = SparseVector<value_type, index_type, ONE_BASED>;
  using extractor   = internal::SpVInternalExtractor<sp_vec_type>;
  const auto c_idx  = [](size_type i) {
    return to_c_idx<size_type, ONE_BASED>(i);
  };
  const auto ori_idx = [](size_type i) {
    return to_ori_idx<size_type, ONE_BASED>(i);
  };

  // get the size of system
  // m is the leading block size and n is the leftover size
  const size_type m = L_B.nrows();
  psmilu_assert(m == L_B.ncols(), "must be squared system for L_B");
  psmilu_assert(L_E.nrows() >= m, "invalid size!");
  const size_type n = L_E.nrows() - m;

  if (!n) return;

  //---------------------------------------------------------------------------
  // regarding computing the first part, there are to steps:
  //  1. compute X=inv(L)B-DU, and
  //  2. compute L_E*X
  // each of these two steps requires symbolic and numerical computations. And
  // for symbolic computation, we first determine the total # of nonzeros, and
  // then compute fill in the indices. Regarding numerical computation, we just
  // need to fill in the values given that the indices (patterns) are known.
  //---------------------------------------------------------------------------

  //---------------------------------------------------------------------------
  // Compute T_E2 = inv(L) * B - D * U
  //---------------------------------------------------------------------------

  // the following kernal is to determine the symbolic kernel for loading a
  // column vector from B. Notice that heavy data is captured by reference
  // col: column index
  // tag: unique tag for sparse vector to ensure the index is unique
  // v (out): output sparse vector, on output, the size is number of nonzeros
  // in B(:,col)
  // return:
  //  the leading index of first nonzero in B(:,col)
  // It's worth noting that B = (P*S*A*T*Q)(1:m,1:m)
  const auto load_A_col_symbolic = [&, m](const size_type col,
                                          const size_type tag,
                                          sp_vec_type &   v) -> size_type {
    psmilu_assert(col < m, "column %zd exceeds size bound %zd", col, m);
    // reset the sparse vector internal counter
    v.reset_counter();
    // get the row index start from permutated index
    auto      itr = A.row_ind_cbegin(q[col]);
    size_type min_idx(m);
    for (auto last = A.row_ind_cend(q[col]); itr != last; ++itr) {
      // from A index to B index, using inverse permutation mapping
      const size_type p_inv = p.inv(c_idx(*itr));
      if (p_inv < m) {
        // got one, push back
        v.push_back(ori_idx(p_inv), tag);
        // update min
        if (min_idx > p_inv) min_idx = p_inv;
      }
    }
    return min_idx;
  };

  // create the sparse vector. Note that in general m > n, but we should not
  // assume this, thus we create a buffer with size of max(m,n). Be aware that
  // size n is needed for the second step.
  sp_vec_type buf(std::max(m, n));
  // we need to direct track the dense tags
  // Notice the d_tags is captured as reference inside the following lambda
  const auto &d_tags = static_cast<const extractor &>(buf).dense_tags();

  // the following lambda is the kernel for solving the forward sub in a
  // symbolic fashion. Since the input L_B is in column major, the loop order
  // is slight different from textbook. Also, be aware that L_B is unit diagonal
  // with implicit diagonal entries.
  //
  // Algorithm: forward sub with unit diagonal
  //  for j=1:m
  //    for i=j+1:m
  //      x(i)-=x(j)*L(i,j)
  //
  // The alg above is for dense arithmetic, for sparse, it is
  //
  // Algorithm: forward sub with unit diagonal (sparse version)
  //  for j=first_nz_index:m
  //    if x(j) == 0: continue
  //    for i=j+1:m and L(i,j) != 0
  //      x(i)-=x(j)*L(i,j)
  const auto solve_col_symbolic =
      [&, m](const size_type tag, const size_type start_index, sp_vec_type &v) {
        // loop from the first nonzero index
        for (size_type j = start_index; j < m; ++j) {
          // by comparing the tags, we know if or not the current entry is
          // zero.
          // NOTE the d_tags is dynamically changing!
          if (static_cast<size_type>(d_tags[j]) != tag) continue;
          // x_j is nz, then for each nz in L(:,j), we known there must be an
          // entry in the solution
          auto last = L_B.row_ind_cend(j);
          auto info = find_sorted(L_B.row_ind_cbegin(j), last, ori_idx(j + 1));
          for (auto itr = info.second; itr != last; ++itr)
            v.push_back(*itr, tag);  // NOTE this may update d_tags
        }
      };

  // the following lambda is the kernel for updating the symbolic pattern for
  // subtracting D*U from inv(L)*B
  const auto subtract_du_symbolic =
      [&, m](const size_type col, const size_type tag, sp_vec_type &v) {
        std::for_each(U_B.row_ind_cbegin(col), U_B.row_ind_cend(col),
                      [&, tag](const index_type i) { v.push_back(i, tag); });
      };

  // create T_E2=inv(L)*B-D*U
  CcsType T_E2(m, m);
  // get the column start array
  auto &col_start = T_E2.col_start();
  col_start.resize(m + 1);
  psmilu_error_if(col_start.status() == DATA_UNDEF,
                  "memory allocation failed!");
  col_start.front() = ONE_BASED;  // set the first entry to be 0 or 1

  // step 1, determine the number of nonzeros
  for (size_type col = 0u; col < m; ++col) {
    // load rhs b=B(:,col) and determine the start index
    const size_type start_index = load_A_col_symbolic(col, col, buf);
    // compute inv(L)*b
    solve_col_symbolic(col, start_index, buf);
    // subtract inv(L)*b-(D*U)(:,col)
    subtract_du_symbolic(col, col, buf);
    // update the col start array by using the size in sparse vector
    col_start[col + 1] = col_start[col] + buf.size();
  }

  // check if empty
  if (!col_start[m]) {
    psmilu_warning(
        "computing H version Schur complement for inv(L)*B-D*U yields an empty "
        "matrix!");
    // NOTE that even for empty matrix, in order to make it valid, we need to
    // allocate the column start array in T_E to ensure it's safe to access
    T_E.resize(n, m);
    T_E.col_start().resize(m + 1);
    std::fill_n(T_E.col_start().begin(), m + 1, ONE_BASED);
    return;
  }

  // reserve the nnz array storage
  T_E2.reserve(col_start[m]);
  auto &row_ind = T_E2.row_ind();
  auto &vals    = T_E2.vals();
  psmilu_error_if(row_ind.status() == DATA_UNDEF || vals.status() == DATA_UNDEF,
                  "memory allocation failed!");
  // for Array, this is just restting the internal size counter
  row_ind.resize(col_start[m]);
  auto      i_itr     = row_ind.begin();
  size_type tag_start = m;
  // redo the loop above, this time, we also push back the indices
  for (size_type col = 0u; col < m; ++col) {
    // IMPORTANT, make a unique tag!
    const size_type tag         = col + tag_start;
    const size_type start_index = load_A_col_symbolic(col, tag, buf);
    solve_col_symbolic(tag, start_index, buf);
    subtract_du_symbolic(col, tag, buf);
    // TODO do we really need to sort the indices??
    buf.sort_indices();
    i_itr =
        std::copy(buf.inds().cbegin(), buf.inds().cbegin() + buf.size(), i_itr);
  }

  // The symbolic part is done! For computing X=inv(L)*B-D*U

  // the following kernel to do the numerical value loading for rhs; just like
  // the kernel for symbolic, but adding codes for assigning numerical values.
  // col: column index
  // tag: unique sparse tag
  // v (output):
  //    on output, it contains the numerical values in the dense value array
  const auto load_A_col_num = [&, m](const size_type col, const size_type tag,
                                     sp_vec_type &v) {
    v.reset_counter();                                // reset counter
    const size_type q_col = q[col];                   // permutated column index
    auto            itr   = A.row_ind_cbegin(q_col);  // index iterator
    auto            v_itr = A.val_cbegin(q_col);      // value iterator
    const auto      t_q   = t[q_col];  // column scaling at this column
    auto &          vals  = v.vals();  // reference to dense value array
    size_type       min_idx(m);
    // same as the symbolic
    for (auto last = A.row_ind_cend(q[col]); itr != last; ++itr, ++v_itr) {
      const size_type j     = c_idx(*itr);
      const size_type p_inv = p.inv(j);
      if (p_inv < m) {
        v.push_back(ori_idx(p_inv), tag);
        vals[j] = s[j] * *v_itr * t_q;  // assign value her
        if (min_idx > p_inv) min_idx = p_inv;
      }
    }
    return min_idx;
  };

  // the following kernel is for solving forward sub, with the algorithm
  // described in the symbolic kernel.
  const auto solve_col_num = [&, m](const size_type tag,
                                    const size_type start_index,
                                    sp_vec_type &   v) {
    auto &vals = v.vals();  // get reference to dense value
    for (size_type j = start_index; j < m; ++j) {
      if (static_cast<size_type>(d_tags[j]) != tag) continue;
      const auto x_j  = vals[j];
      auto       last = L_B.row_ind_cend(j);
      auto info  = find_sorted(L_B.row_ind_cbegin(j), last, ori_idx(j + 1));
      auto v_itr = L_B.val_cbegin(j) + (info.second - L_B.row_ind_cbegin(j));
      // for each nz in L(:,j), we push the index back
      // if a new value is added to the sparse vector (return true), then
      // we assign -L(*itr,j)*x_j, ow subtract L(*itr,j)
      for (auto itr = info.second; itr != last; ++itr, ++v_itr)
        v.push_back(*itr, tag) ? vals[c_idx(*itr)] = -*v_itr * x_j
                               : vals[c_idx(*itr)] -= *v_itr * x_j;
    }
  };

  // numerical kernel for subtracting (D*U)(:,col)
  const auto subtract_du_num = [&, m](const size_type col, const size_type tag,
                                      sp_vec_type &v) {
    auto &vals  = v.vals();
    auto  itr   = U_B.row_ind_cbegin(col);
    auto  v_itr = U_B.val_cbegin(col);
    for (auto last = U_B.row_ind_cend(col); itr != last; ++itr, ++v_itr) {
      const size_type j                = c_idx(*itr);
      v.push_back(*itr, tag) ? vals[j] = -d[j] * *v_itr
                             : vals[j] -= d[j] * *v_itr;
    }
  };

  // resize the value array in T_E, O(1) operation for Array
  vals.resize(col_start[m]);
  auto v_itr = vals.begin();
  tag_start += m;
  // get the buf value reference
  auto &buf_vals = buf.vals();
  for (size_type col = 0u; col < m; ++col) {
    // get the unique tag!
    const size_type tag         = col + tag_start;
    const size_type start_index = load_A_col_num(col, tag, buf);
    solve_col_num(tag, start_index, buf);
    subtract_du_num(col, tag, buf);
    // since col_start is already built, the following accessing is valid
    auto itr = T_E2.row_ind_cbegin(col);
    // NOTE that we just loop through the index parttern and directly access
    // the values.
    for (auto last = T_E2.row_ind_cend(col); itr != last; ++itr, ++v_itr)
      *v_itr = buf_vals[c_idx(*itr)];
  }

  //---------------------------------------------------------------------------
  // FINISHED: Compute T_E2 = inv(L) * B - D * U
  //---------------------------------------------------------------------------

  //---------------------------------------------------------------------------
  // Compute T_E = L_E * T_E2
  //
  // This part is nothing but a sparse sparse multiplication with both left-
  // and right- hand sides to be column major storage. Since the return matrix
  // is also column major, we use the following looping order for matrix
  // multiplication
  //
  // for j=1:m
  //  for k=1:m
  //    for i=1:n
  //      T_E(i,j)+=L_E(i,k)*T_E2(k,j)
  //---------------------------------------------------------------------------

  // reshape the matrix shape
  T_E.resize(n, m);
  auto &o_col_start = T_E.col_start();
  o_col_start.resize(m + 1);
  psmilu_error_if(o_col_start.status() == DATA_UNDEF,
                  "memory allocation failed");
  o_col_start.front() = ONE_BASED;
  tag_start += m;
  auto start_pos = L_E.row_ind().cbegin();
  // loop begins, for each column (j)
  // This the first step in symbolic computation, to determine the nnz in the
  // T_E matrix
  for (size_type col = 0u; col < m; ++col) {
    const size_type tag = col + tag_start;
    buf.reset_counter();  // reset buffer size counter
    auto rhs_itr = T_E2.row_ind_cbegin(col);
    // for each nz in T_E2(:,col)
    for (auto rhs_last = T_E2.row_ind_cend(col); rhs_itr != rhs_last;
         ++rhs_itr) {
      // we use the starting position array to, virtually, split L_E from
      // the whole L from Crout update
      const size_type k       = c_idx(*rhs_itr);
      auto            lhs_itr = start_pos + L_start[k];
      // for each nz in L_E(:,k)
      for (auto lhs_last = L_E.row_ind_cend(k); lhs_itr != lhs_last; ++lhs_itr)
        buf.push_back(*lhs_itr - m, tag);
    }
    o_col_start[col + 1] = o_col_start[col] + buf.size();
  }

  if (!o_col_start[m]) {
    psmilu_warning("H version of Schur complement has an empty T_E part!");
    return;
  }

  tag_start += m;
  T_E.reserve(o_col_start[m]);
  auto &o_row_ind = T_E.row_ind();
  auto &o_vals    = T_E.vals();
  psmilu_error_if(
      o_row_ind.status() == DATA_UNDEF || o_vals.status() == DATA_UNDEF,
      "memory allocation failed!");
  o_row_ind.resize(o_col_start[m]);
  auto o_itr = o_row_ind.begin();
  // redo the loop above to fill in the indices, aka symbolic pattern
  for (size_type col = 0u; col < m; ++col) {
    const size_type tag = col + tag_start;
    buf.reset_counter();
    auto rhs_itr = T_E2.row_ind_cbegin(col);
    for (auto rhs_last = T_E2.row_ind_cend(col); rhs_itr != rhs_last;
         ++rhs_itr) {
      const size_type j       = c_idx(*rhs_itr);
      auto            lhs_itr = start_pos + L_start[j];
      for (auto lhs_last = L_E.row_ind_cend(j); lhs_itr != lhs_last; ++lhs_itr)
        buf.push_back(*lhs_itr - m, tag);
    }
    // sort indices and push the index pattern to the row_ind array
    buf.sort_indices();
    o_itr =
        std::copy(buf.inds().cbegin(), buf.inds().cbegin() + buf.size(), o_itr);
  }

  // finally, we compute the numerical values of L_E*T_E2

  o_vals.resize(o_col_start[m]);
  auto o_v_itr     = o_vals.begin();
  auto start_v_pos = L_E.vals().cbegin();
  for (size_type col = 0u; col < m; ++col) {
    // first, from the index pattern, we first assign the corresponding entries
    // in buffer value to be zero, O(nnz(T_E(:,col)))
    for (auto itr = T_E.row_ind_cbegin(col), last = T_E.row_ind_cend(col);
         itr != last; ++itr)
      buf_vals[c_idx(*itr)] = 0;
    auto rhs_itr   = T_E2.row_ind_cbegin(col);
    auto rhs_v_itr = T_E2.val_cbegin(col);
    // same loop mechanism but for numerical value this time
    for (auto rhs_last = T_E2.row_ind_cend(col); rhs_itr != rhs_last;
         ++rhs_itr, ++rhs_v_itr) {
      const size_type j         = c_idx(*rhs_itr);
      auto            lhs_itr   = start_pos + L_start[j];
      auto            lhs_v_itr = start_v_pos + L_start[j];
      const auto      temp      = *rhs_v_itr;  // break strong reference
      for (auto lhs_last = L_E.row_ind_cend(j); lhs_itr != lhs_last;
           ++lhs_itr, ++lhs_v_itr)
        buf_vals[c_idx(*lhs_itr - m)] += *lhs_v_itr * temp;
    }
    // finally, with the index pattern, we directly assign the values in
    // the dense buffer to our compressed value array.
    for (auto itr = T_E.row_ind_cbegin(col), last = T_E.row_ind_cend(col);
         itr != last; ++itr, ++o_v_itr)
      *o_v_itr = buf_vals[c_idx(*itr)];
  }

  //---------------------------------------------------------------------------
  // FINISHED: Compute T_E = L_E * T_E2
  //---------------------------------------------------------------------------
}

/// \brief compute the second part, \f$\boldsymbol{T}_F\f$, of H version
/// \tparam U_B_CcsType upper part type, see \ref CCS
/// \tparam U_AugCrsType augmented crs type for upper part, see \ref AugCRS
/// \tparam CcsType input and output matrix type, see \ref CCS
/// \param[in] U_B upper part of leading block
/// \param[in] U_F whole upper part after Crout update
/// \param[out] T_F the output part
/// \ingroup schur
/// \sa compute_Schur_H_T_E
template <class U_B_CcsType, class U_AugCrsType, class CcsType>
inline void compute_Schur_H_T_F(const U_B_CcsType &U_B, const U_AugCrsType &U_F,
                                CcsType &T_F) {
  using size_type  = typename CcsType::size_type;
  using value_type = typename CcsType::value_type;
  using index_type = typename CcsType::index_type;
  using iterator   = std::reverse_iterator<decltype(U_B.row_ind_cbegin(0))>;
  constexpr static bool ONE_BASED = CcsType::ONE_BASED;
  using sp_vec_type = SparseVector<value_type, index_type, ONE_BASED>;
  using extractor   = internal::SpVInternalExtractor<sp_vec_type>;
  const auto c_idx  = [](const size_type i) {
    return to_c_idx<size_type, ONE_BASED>(i);
  };
  const auto ori_idx = [](const size_type i) {
    return to_ori_idx<size_type, ONE_BASED>(i);
  };

  const size_type m = U_B.nrows();
  psmilu_assert(m == U_B.ncols(), "U should be squared matrix");
  const size_type n = U_F.ncols() - m;

  sp_vec_type buf(m);  // create buffer

  // Computing the second part, i.e. T_F, is nothing but performing a serial
  // of backward sub. Since U is stored in column major, the loop is slightly
  // different than the version in textbook.
  //
  // Algorithm: backward sub with unit diagonal
  //  for j=n:1:-1
  //    for i=j-1:1:-1
  //      x(i)-=x(j)*U(i,j)
  //
  // The sparse version reads:
  //
  // Algorithm: backward sub with unit diagonal (sparse version)
  //  for j=last_nz:1:-1
  //    if x(j) == 0: continue
  //    for i=j-1:1:-1 and U(i,j) != 0
  //      x(i)-=x(j)*U(i,j)
  //
  // As usual, we need to do symbolic computation followed by numerical fillins

  T_F.resize(m, n);
  auto &col_start = T_F.col_start();
  col_start.resize(n + 1);
  psmilu_error_if(col_start.status() == DATA_UNDEF, "memory allocation failed");
  col_start.front() = ONE_BASED;

  // symbolically loading the rhs, i.e. U_F(:, col)
  // since U_F is virtually splitted from the whole U from Crout update by
  // utilizing the augmented data structure with starting index m (leading
  // block size), loading the values requires using augmented API
  const auto load_U_col_symbolic = [&, m](const size_type col,
                                          const size_type tag, sp_vec_type &v) {
    index_type aug_id = U_F.start_col_id(col + m);  // get the start aug handle
    v.reset_counter();                              // reset counter in buffer
    while (!U_F.is_nil(aug_id)) {
      // get the row index (C-based)
      const size_type j = U_F.row_idx(aug_id);
#ifdef PSMILU_SCHUR_USE_FULL
      if (j >= m) break;
#else
      psmilu_assert(j < m, "invalid U_F matrix");
#endif
      // push back to buffer
      v.push_back(ori_idx(j), tag);
      aug_id = U_F.next_col_id(aug_id);  // advance handle
    }
  };

  // extract the dense tags, which is used in backward sub
  const auto &d_tags = static_cast<const extractor &>(buf).dense_tags();

  // symbolic kernel for solving inv(U)*b, where b=U_F(:,col)
  const auto solve_col_symbolic =
      [&](const size_type tag, const size_type start_index, sp_vec_type &v) {
        // note for first entry, no new entries will be added
        for (size_type j = start_index; j != 0u; --j) {
          if (static_cast<size_type>(d_tags[j]) != tag) continue;
          // find the last entry
          auto info = find_sorted(U_B.row_ind_cbegin(j), U_B.row_ind_cend(j),
                                  ori_idx(j - 1));
          // IMPORTANT! since find_sorted uses lower_bound, which means if we
          // actually find an entry with value j-1, then it will be skipped in
          // reserve_iterator! Therefore, we need to advance the iterator if we
          // encounter this.
          if (info.first) ++info.second;
          auto last = iterator(U_B.row_ind_cbegin(j));
          auto itr  = iterator(info.second);
          for (; itr != last; ++itr) v.push_back(*itr, tag);
        }
      };

  for (size_type col = 0u; col < n; ++col) {
    load_U_col_symbolic(col, col, buf);
    // NOTE the last index is just the index value of the last entry in buffer
    // since we know that buf is reset in loading U and augmented loop ensures
    // the resulting index array is sorted.
    if (buf.size())
      solve_col_symbolic(col, c_idx(buf.c_idx(buf.size() - 1)), buf);
    col_start[col + 1] = col_start[col] + buf.size();  // build col_start
  }

  if (!col_start[n]) {
    psmilu_warning("H version of Schur complement has an empty T_F part!");
    return;
  }

  // reserve the nnz arrays
  T_F.reserve(col_start[n]);
  auto &row_ind = T_F.row_ind();
  auto &vals    = T_F.vals();
  psmilu_error_if(row_ind.status() == DATA_UNDEF || vals.status() == DATA_UNDEF,
                  "memory allocation failed!");
  row_ind.reserve(col_start[n]);
  auto      i_itr     = row_ind.begin();
  size_type tag_start = n;
  // same loop but for indices
  for (size_type col = 0u; col < n; ++col) {
    const size_type tag = col + tag_start;
    load_U_col_symbolic(col, tag, buf);
    psmilu_assert(
        std::is_sorted(buf.inds().cbegin(), buf.inds().cbegin() + buf.size()),
        "after loading U, the indices should be sorted");
    if (buf.size())
      solve_col_symbolic(tag, c_idx(buf.c_idx(buf.size() - 1)), buf);
    buf.sort_indices();
    i_itr =
        std::copy(buf.inds().cbegin(), buf.inds().cbegin() + buf.size(), i_itr);
  }

  // kernel to load numerical data from U_F(:,col)
  const auto load_U_col_num = [&, m](const size_type col, const size_type tag,
                                     sp_vec_type &v) {
    auto &     buf_vals = v.vals();
    index_type aug_id   = U_F.start_col_id(col + m);
    v.reset_counter();
    while (!U_F.is_nil(aug_id)) {
      const size_type j = U_F.row_idx(aug_id);
#ifdef PSMILU_SCHUR_USE_FULL
      if (j >= m) break;
#endif
      v.push_back(ori_idx(j), tag);
      buf_vals[j] = U_F.val_from_col_id(aug_id);
      aug_id      = U_F.next_col_id(aug_id);
    }
  };

  // kernel to solve for inv(U)*b numerically
  const auto solve_col_num = [&](const size_type tag,
                                 const size_type start_index, sp_vec_type &v) {
    using v_iterator = std::reverse_iterator<decltype(U_B.val_cbegin(0))>;
    auto &buf_vals   = v.vals();
    for (size_type j = start_index; j != 0u; --j) {
      if (static_cast<size_type>(d_tags[j]) != tag) continue;
      const auto x_j  = buf_vals[j];
      auto       info = find_sorted(U_B.row_ind_cbegin(j), U_B.row_ind_cend(j),
                              ori_idx(j - 1));
      if (info.first) ++info.second;  // increment if *info.second==j-1
      auto v_itr =
          v_iterator(U_B.val_cbegin(j) + (info.second - U_B.row_ind_cbegin(j)));
      auto last = iterator(U_B.row_ind_cbegin(j));
      auto itr  = iterator(info.second);
      for (; itr != last; ++itr, ++v_itr)
        v.push_back(*itr, tag) ? buf_vals[c_idx(*itr)] = -*v_itr * x_j
                               : buf_vals[c_idx(*itr)] -= *v_itr * x_j;
    }
  };

  // finally, fetch values directly from the dense value buffer
  vals.resize(col_start[n]);
  auto v_itr = vals.begin();
  tag_start += n;
  const auto &buf_vals = buf.vals();
  for (size_type col = 0u; col < n; ++col) {
    const size_type tag = col + tag_start;
    load_U_col_num(col, tag, buf);
    if (buf.size()) solve_col_num(tag, buf.c_idx(buf.size() - 1), buf);
    auto itr = T_F.row_ind_cbegin(col);
    for (auto last = T_F.row_ind_cend(col); itr != last; ++itr, ++v_itr)
      *v_itr = buf_vals[c_idx(*itr)];
  }
}

}  // namespace internal

/// \brief compute H version of Schur complement
/// \tparam L_AugCcsType augmented ccs type for L, see \ref AugCCS
/// \tparam L_StartType starting position after Crout update for L
/// \tparam L_B_CcsType lower part type, see \ref CCS
/// \tparam LeftDiagType row scaling vector type, see \ref Array
/// \tparam CcsType input and output matrix type, see \ref CCS
/// \tparam RightDiagType column scaling type, see \ref Array
/// \tparam PermType permutation matrix type, see \ref BiPermMatrix
/// \tparam DiagType diagonal vector type, see \ref Array
/// \tparam U_B_CcsType upper part type, see \ref CCS
/// \tparam U_AugCrsType augmented crs type for upper part, see \ref AugCRS
/// \tparam DenseType dense matrix for C version, see \ref DenseMatrix
/// \param[in] L_E whole lower part after \ref Crout update
/// \param[in] L_start position array after Crout update for lower part
/// \param[in] L_B lower part of leading block
/// \param[in] s row scaling vector
/// \param[in] A input matrix in CCS format
/// \param[in] t column scaling vector
/// \param[in] p row permutation matrix
/// \param[in] q column permutation matrix
/// \param[in] d diagonal entries (permutated)
/// \param[in] U_B upper part of leading block
/// \param[in] U_F whole upper part after Crout update
/// \param[in,out] HC C version of Schur complement in dense, H version upon
///                 output
/// \ingroup schur
/// \sa compute_Schur_C
/// \note Both \a L_B and \a U_B are matrices with size m by m, where m is the
///       leading block size. However, on the other side, both \a L_E and \a U_F
///       are the complete augmented parts, where only the offsets are accessed.
///
/// Computing the H version of Schur complement is one of the most complicated
/// algorithms in PSMILU library. The analytical formula reads:
///
/// \f[
///   \boldsymbol{H}_C=\boldsymbol{S}_C+\boldsymbol{T}_E\boldsymbol{T}_F
/// \f]
///
/// where
///
/// \f[
///   \boldsymbol{T}_E=\boldsymbol{L}_E\left(\boldsymbol{L}_B^{-1}
///     \hat{\boldsymbol{B}}-\boldsymbol{D}_B\boldsymbol{U}_B\right)
/// \f]
///
/// and
///
/// \f[
///   \boldsymbol{T}_F=\boldsymbol{U}_B^{-1}\boldsymbol{U}_F
/// \f]
///
/// It's worth noting that \f$\hat{\boldsymbol{B}}\f$ is nothing but just
/// \f$\left(\boldsymbol{S}\boldsymbol{A}\boldsymbol{T}\right)_ {\boldsymbol
/// {p}_{1:m},\boldsymbol{q}_{1:m}}\f$
///
/// At high level, the H version is computed with two steps, where the first
/// step is to form \f$\boldsymbol{T}_E\f$ while \f$\boldsymbol{T}_F\f$ for the
/// second step. Furthermore, \f$\boldsymbol{T}_E\f$ is computed in two substeps
/// 1) compute \f$\boldsymbol{T}_E^{*}=\boldsymbol{L}_B^{-1}\hat{\boldsymbol{B}}
/// -\boldsymbol{D}_B\boldsymbol{U}_B\f$, and 2) compute \f$\boldsymbol{L}_E
/// \boldsymbol{T}_E^{*}\f$.
///
/// For detailed implementation, each of those steps are computed with a unified
/// procedure, i.e. symbolic and numerical computations. The symbolic step is to
/// determine the sparse pattern, which involves computing the exact total
/// number of nonzeros and filling the indices; once the symbolic step is done,
/// we can easily fill in the numerical values given all indices are known; the
/// "easyness" is achieved with a dense vector buffer for values, i.e. using
/// \ref SparseVector.
///
/// As regards the time complexity, it's hard (or even impossible) to give a
/// precise bound, but easy to estimate the worst case. Assuming all inputs are
/// dense, then the complexity is bounded by \f$\mathcal{O}(m^3)\f$, where
/// \f$m\f$ is the leading block size, which should be larger than the Schur
/// complement size. The above bound is derived from solving
/// \f$\boldsymbol{L}_B^{-1}\hat{\boldsymbol{B}}\f$. In order to derive the
/// best case (which is impossible to have in practice), we need to analyze the
/// complexity of solving forward/backward substitution, e.g.
/// \f$\boldsymbol{L}^{-1}\boldsymbol{b}\f$ (forward), where
/// \f$\boldsymbol{b}\f$ is a \b sparse vector. In this case, the complexity is
/// determined by the \b first nonzero entry in \f$\boldsymbol{b}\f$, thus the
/// most efficient time is bounded by
/// \f$\mathcal{O}(\textrm{nnz}(b)\textrm{nnz}(\cup_{j,b_j\neq 0}L))\f$, which
/// is constant given the assumption of constant values of \f$\textrm{nnz}(b)\f$
/// and \f$\textrm{nnz}(L(:,i))\f$. Therefore, the best case for solving
/// \f$\boldsymbol{L}_B^{-1}\hat{\boldsymbol{B}}\f$ is in
/// \f$\mathcal{O}(\textrm{nnz}(b)\textrm{nnz}(\cup_{j,b_j\neq 0}L)m)\f$, which
/// is linear. Similar argument can be drew for backward substitution; the only
/// difference, in this case, is to find the last nonzero entry. Now, notice
/// that computing sparse matrix multiplication costs
/// \f$\mathcal{O}(\textrm{nnz}^2m)\f$, where \f$\textrm{nnz}\f$ is some
/// averaged number of nonzeros; if this is bounded by constant, then the
/// complexity is linear. Therefore, in best case scenario, computing the H
/// version requires linear complexity in terms of the system size. \b However,
/// this is just too good to be true, if we use a randomized analysis procedure,
/// we can see that on average, the complexity is cubic.
///
/// With the consideration of the H version computation complexity, we only
/// apply it on the last level \b and \f$m\sim\match{O}(N^{1/3})\f$, where
/// \f$N\f$ is the global system size, i.e. input size on first level. This
/// ensures that computing H version of Schur complement won't destroy our
/// overall time complexity analysis.
template <class L_AugCcsType, class L_StartType, class L_B_CcsType,
          class LeftDiagType, class CcsType, class RightDiagType,
          class PermType, class DiagType, class U_B_CcsType, class U_AugCrsType,
          class DenseType>
inline void compute_Schur_H(
    const L_AugCcsType &L_E, const L_StartType &L_start, const L_B_CcsType &L_B,
    const LeftDiagType &s, const CcsType &A, const RightDiagType &t,
    const PermType &p, const PermType &q, const DiagType &d,
    const U_B_CcsType &U_B, const U_AugCrsType &U_F,
    DenseType &HC
#ifdef PSMILU_UNIT_TESTING
    // the following two parameters are exposed for unit testing
    ,
    CcsType &T_E, CcsType &T_F
#endif
) {
  // static checking
  static_assert(!L_AugCcsType::ROW_MAJOR, "L_E should be CCS type");
  static_assert(!L_B_CcsType::ROW_MAJOR, "L_B should be CCS type");
  static_assert(!CcsType::ROW_MAJOR, "A should be CCS type");
  static_assert(!U_B_CcsType::ROW_MAJOR, "U_B should be CCS type");
  static_assert(U_AugCrsType::ROW_MAJOR, "U_F should be CRS type");
  static_assert(!(L_AugCcsType::ONE_BASED ^ L_B_CcsType::ONE_BASED),
                "inconsistent index system");
  static_assert(!(L_AugCcsType::ONE_BASED ^ CcsType::ONE_BASED),
                "inconsistent index system");
  static_assert(!(L_AugCcsType::ONE_BASED ^ U_B_CcsType::ONE_BASED),
                "inconsistent index system");
  static_assert(!(L_AugCcsType::ONE_BASED ^ U_AugCrsType::ONE_BASED),
                "inconsistent index system");
  using size_type                 = typename CcsType::size_type;
  constexpr static bool ONE_BASED = CcsType::ONE_BASED;
  const auto            c_idx     = [](size_type i) {
    return to_c_idx<size_type, ONE_BASED>(i);
  };

  psmilu_warning_if(HC.nrows() != HC.ncols(),
                    "dense matrix HC is not squared!");
  if (!HC.nrows()) return;
#ifndef PSMILU_UNIT_TESTING
  CcsType T_E;
#endif
  // compute the T_E part
  internal::compute_Schur_H_T_E(L_E, L_start, L_B, s, A, t, p, q, d, U_B, T_E);
#ifndef PSMILU_UNIT_TESTING
  CcsType T_F;
#endif
  // compute the T_F part
  internal::compute_Schur_H_T_F(U_B, U_F, T_F);

  psmilu_assert(HC.nrows() == T_E.nrows(),
                "T_E and HC should have same row size");
  psmilu_assert(HC.ncols() == T_F.ncols(),
                "T_F and HC should have same column size");

  // mm operation with all A, B and C column major. Therefore, regarding the
  // kernel order, i.e. C(i,j)+=A(i,k)*B(k,j), i goes fastest, following by k,
  // finally j.
  const size_type M = HC.ncols();
  for (size_type col = 0u; col < M; ++col) {
    auto rhs_v_itr   = T_F.val_cbegin(col);  // get rhs value iterator
    auto dense_v_itr = HC.col_begin(col);    // get the dense column iterator
    for (auto rhs_itr  = T_F.row_ind_cbegin(col),
              rhs_last = T_F.row_ind_cend(col);
         rhs_itr != rhs_last; ++rhs_itr, ++rhs_v_itr) {
      const size_type k         = c_idx(*rhs_itr);  // get k
      const auto      temp      = *rhs_v_itr;       // copy value to local temp
      auto            lhs_v_itr = T_E.val_cbegin(k);
      for (auto lhs_itr = T_E.row_ind_cbegin(k), lhs_last = T_E.row_ind_cend(k);
           lhs_itr != lhs_last; ++lhs_itr, ++lhs_v_itr) {
        psmilu_assert(c_idx(*lhs_itr) < HC.nrows(),
                      "%zd exceeds the SC matrix row size", c_idx(*lhs_itr));
        // mm kernel, c_ij+=a_ik*b_kj
        dense_v_itr[c_idx(*lhs_itr)] += *lhs_v_itr * temp;
      }
    }
  }
}

}  // namespace psmilu

#endif  // _PSMILU_SCHUR_HPP
