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
  const size_type m = L_B.nrows();
  psmilu_assert(m == L_B.ncols(), "must be squared system for L_B");
  psmilu_assert(L_E.nrows() >= m, "invalid size!");
  const size_type n = L_E.nrows() - m;

  const auto load_A_col_symbolic = [&, m](const size_type col,
                                          const size_type tag,
                                          sp_vec_type &   v) -> size_type {
    psmilu_assert(col < m, "column %zd exceeds size bound %zd", col, m);
    v.reset_counter();
    auto      itr = A.row_ind_cbegin(q[col]);
    size_type min_idx(m);
    for (auto last = A.row_ind_cend(q[col]); itr != last; ++itr) {
      const size_type p_inv = p.inv(c_idx(*itr));
      if (p_inv < m) {
        v.push_back(ori_idx(p_inv), tag);
        if (min_idx > p_inv) min_idx = p_inv;
      }
    }
    return min_idx;
  };

  sp_vec_type buf(std::max(m, n));
  // we need to direct track the dense tags
  const auto &d_tags = static_cast<const extractor &>(buf).dense_tags();

  const auto solve_col_symbolic =
      [&, m](const size_type tag, const size_type start_index, sp_vec_type &v) {
        for (size_type j = start_index; j < m; ++j) {
          if (static_cast<size_type>(d_tags[j]) != tag) continue;
          auto last = L_B.row_ind_cend(j);
          auto info = find_sorted(L_B.row_ind_cbegin(j), last, ori_idx(j + 1));
          for (auto itr = info.second; itr != last; ++itr)
            v.push_back(*itr, tag);
        }
      };

  const auto subtract_du_symbolic =
      [&, m](const size_type col, const size_type tag, sp_vec_type &v) {
        std::for_each(U_B.row_ind_cbegin(col), U_B.row_ind_cend(col),
                      [&, tag](const index_type i) { v.push_back(i, tag); });
      };

  CcsType T_E2(m, m);
  auto &  col_start = T_E2.col_start();
  col_start.resize(m + 1);
  psmilu_error_if(col_start.status() == DATA_UNDEF,
                  "memory allocation failed!");
  col_start.front() = ONE_BASED;

  for (size_type col = 0u; col < m; ++col) {
    const size_type start_index = load_A_col_symbolic(col, col, buf);
    solve_col_symbolic(col, start_index, buf);
    subtract_du_symbolic(col, col, buf);
    col_start[col + 1] = col_start[col] + buf.size();
  }

  if (!col_start[m]) {
    psmilu_warning(
        "computing H version Schur complement for inv(L)*B-D*U yields an empty "
        "matrix!");
    T_E.resize(n, m);
    T_E.col_start().resize(m + 1);
    std::fill_n(T_E.col_start().begin(), m + 1, ONE_BASED);
    return;
  }

  T_E2.reserve(col_start[m]);
  auto &row_ind = T_E2.row_ind();
  auto &vals    = T_E2.vals();
  psmilu_error_if(row_ind.status() == DATA_UNDEF || vals.status() == DATA_UNDEF,
                  "memory allocation failed!");
  row_ind.resize(col_start[m]);
  auto      i_itr     = row_ind.begin();
  size_type tag_start = m;
  for (size_type col = 0u; col < m; ++col) {
    const size_type tag         = col + tag_start;
    const size_type start_index = load_A_col_symbolic(col, tag, buf);
    solve_col_symbolic(tag, start_index, buf);
    subtract_du_symbolic(col, tag, buf);
    buf.sort_indices();
    i_itr =
        std::copy(buf.inds().cbegin(), buf.inds().cbegin() + buf.size(), i_itr);
  }

  const auto load_A_col_num = [&, m](const size_type col, const size_type tag,
                                     sp_vec_type &v) {
    v.reset_counter();
    const size_type q_col = q[col];
    auto            itr   = A.row_ind_cbegin(q_col);
    auto            v_itr = A.val_cbegin(q_col);
    const auto      t_q   = t[q_col];
    auto &          vals  = v.vals();
    size_type       min_idx(m);
    for (auto last = A.row_ind_cend(q[col]); itr != last; ++itr, ++v_itr) {
      const size_type j     = c_idx(*itr);
      const size_type p_inv = p.inv(j);
      if (p_inv < m) {
        v.push_back(ori_idx(p_inv), tag);
        vals[j] = s[j] * *v_itr * t_q;
        if (min_idx > p_inv) min_idx = p_inv;
      }
    }
    return min_idx;
  };

  const auto solve_col_num = [&, m](const size_type tag,
                                    const size_type start_index,
                                    sp_vec_type &   v) {
    auto &vals = v.vals();
    for (size_type j = start_index; j < m; ++j) {
      if (static_cast<size_type>(d_tags[j]) != tag) continue;
      const auto x_j  = vals[j];
      auto       last = L_B.row_ind_cend(j);
      auto info  = find_sorted(L_B.row_ind_cbegin(j), last, ori_idx(j + 1));
      auto v_itr = L_B.val_cbegin(j) + (info.second - L_B.row_ind_cbegin(j));
      for (auto itr = info.second; itr != last; ++itr, ++v_itr)
        v.push_back(*itr, tag) ? vals[c_idx(*itr)] = -*v_itr * x_j
                               : vals[c_idx(*itr)] -= *v_itr * x_j;
    }
  };

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

  vals.resize(col_start[m]);
  auto v_itr = vals.begin();
  tag_start += m;
  auto &buf_vals = buf.vals();
  for (size_type col = 0u; col < m; ++col) {
    const size_type tag         = col + tag_start;
    const size_type start_index = load_A_col_num(col, tag, buf);
    solve_col_num(tag, start_index, buf);
    subtract_du_num(col, tag, buf);
    auto itr = T_E2.row_ind_cbegin(col);
    for (auto last = T_E2.row_ind_cend(col); itr != last; ++itr, ++v_itr)
      *v_itr = buf_vals[c_idx(*itr)];
  }

  T_E.resize(n, m);
  auto &o_col_start = T_E.col_start();
  o_col_start.resize(m + 1);
  psmilu_error_if(o_col_start.status() == DATA_UNDEF,
                  "memory allocation failed");
  o_col_start.front() = ONE_BASED;
  tag_start += m;
  auto start_pos = L_E.row_ind().cbegin();
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
    buf.sort_indices();
    o_itr =
        std::copy(buf.inds().cbegin(), buf.inds().cbegin() + buf.size(), o_itr);
  }

  o_vals.resize(o_col_start[m]);
  auto o_v_itr     = o_vals.begin();
  auto start_v_pos = L_E.vals().cbegin();
  for (size_type col = 0u; col < m; ++col) {
    for (auto itr = T_E.row_ind_cbegin(col), last = T_E.row_ind_cend(col);
         itr != last; ++itr)
      buf_vals[c_idx(*itr)] = 0;
    auto rhs_itr   = T_E2.row_ind_cbegin(col);
    auto rhs_v_itr = T_E2.val_cbegin(col);
    for (auto rhs_last = T_E2.row_ind_cend(col); rhs_itr != rhs_last;
         ++rhs_itr, ++rhs_v_itr) {
      const size_type j         = c_idx(*rhs_itr);
      auto            lhs_itr   = start_pos + L_start[j];
      auto            lhs_v_itr = start_v_pos + L_start[j];
      const auto      temp      = *rhs_v_itr;
      for (auto lhs_last = L_E.row_ind_cend(j); lhs_itr != lhs_last;
           ++lhs_itr, ++lhs_v_itr)
        buf_vals[c_idx(*lhs_itr - m)] += *lhs_v_itr * temp;
    }
    for (auto itr = T_E.row_ind_cbegin(col), last = T_E.row_ind_cend(col);
         itr != last; ++itr, ++o_v_itr)
      *o_v_itr = buf_vals[c_idx(*itr)];
  }
}

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

  sp_vec_type buf(m);

  T_F.resize(m, n);
  auto &col_start = T_F.col_start();
  col_start.resize(n + 1);
  psmilu_error_if(col_start.status() == DATA_UNDEF, "memory allocation failed");
  col_start.front() = ONE_BASED;

  const auto load_U_col_symbolic = [&, m](const size_type col,
                                          const size_type tag, sp_vec_type &v) {
    index_type aug_id = U_F.start_col_id(col + m);
    v.reset_counter();
    while (!U_F.is_nil(aug_id)) {
      const size_type j = U_F.row_idx(aug_id);
#ifdef PSMILU_SCHUR_USE_FULL
      if (j >= m) break;
#else
      psmilu_assert(j < m, "invalid U_F matrix");
#endif
      v.push_back(ori_idx(j), tag);
      aug_id = U_F.next_col_id(aug_id);
    }
  };

  const auto &d_tags = static_cast<const extractor &>(buf).dense_tags();

  const auto solve_col_symbolic =
      [&](const size_type tag, const size_type start_index, sp_vec_type &v) {
        // note for first entry, no new entries will be added
        for (size_type j = start_index; j != 0u; --j) {
          if (static_cast<size_type>(d_tags[j]) != tag) continue;
          auto info = find_sorted(U_B.row_ind_cbegin(j), U_B.row_ind_cend(j),
                                  ori_idx(j - 1));
          if (info.first) ++info.second;
          auto last = iterator(U_B.row_ind_cbegin(j));
          auto itr  = iterator(info.second);
          for (; itr != last; ++itr) v.push_back(*itr, tag);
        }
      };

  for (size_type col = 0u; col < n; ++col) {
    load_U_col_symbolic(col, col, buf);
    if (buf.size())
      solve_col_symbolic(col, c_idx(buf.c_idx(buf.size() - 1)), buf);
    col_start[col + 1] = col_start[col] + buf.size();
  }

  if (!col_start[n]) {
    psmilu_warning("H version of Schur complement has an empty T_F part!");
    return;
  }

  T_F.reserve(col_start[n]);
  auto &row_ind = T_F.row_ind();
  auto &vals    = T_F.vals();
  psmilu_error_if(row_ind.status() == DATA_UNDEF || vals.status() == DATA_UNDEF,
                  "memory allocation failed!");
  row_ind.reserve(col_start[n]);
  auto      i_itr     = row_ind.begin();
  size_type tag_start = n;
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

  const auto solve_col_num = [&](const size_type tag,
                                 const size_type start_index, sp_vec_type &v) {
    using v_iterator = std::reverse_iterator<decltype(U_B.val_cbegin(0))>;
    auto &buf_vals   = v.vals();
    for (size_type j = start_index; j != 0u; --j) {
      if (static_cast<size_type>(d_tags[j]) != tag) continue;
      const auto x_j  = buf_vals[j];
      auto       info = find_sorted(U_B.row_ind_cbegin(j), U_B.row_ind_cend(j),
                              ori_idx(j - 1));
      if (info.first) ++info.second;
      auto v_itr =
          v_iterator(U_B.val_cbegin(j) + (info.second - U_B.row_ind_cbegin(j)));
      auto last = iterator(U_B.row_ind_cbegin(j));
      auto itr  = iterator(info.second);
      for (; itr != last; ++itr, ++v_itr)
        v.push_back(*itr, tag) ? buf_vals[c_idx(*itr)] = -*v_itr * x_j
                               : buf_vals[c_idx(*itr)] -= *v_itr * x_j;
    }
  };

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

template <class L_AugCcsType, class L_StartType, class L_B_CcsType,
          class LeftDiagType, class CcsType, class RightDiagType,
          class PermType, class DiagType, class U_B_CcsType, class U_AugCrsType,
          class DenseType>
inline void compute_Schur_H(const L_AugCcsType &L_E, const L_StartType &L_start,
                            const L_B_CcsType &L_B, const LeftDiagType &s,
                            const CcsType &A, const RightDiagType &t,
                            const PermType &p, const PermType &q,
                            const DiagType &d, const U_B_CcsType &U_B,
                            const U_AugCrsType &U_F, DenseType &HC
#ifdef PSMILU_UNIT_TESTING
                            ,
                            CcsType &T_E, CcsType &T_F
#endif
) {
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
#ifndef PSMILU_UNIT_TESTING
  CcsType T_E;
#endif
  internal::compute_Schur_H_T_E(L_E, L_start, L_B, s, A, t, p, q, d, U_B, T_E);
#ifndef PSMILU_UNIT_TESTING
  CcsType T_F;
#endif
  internal::compute_Schur_H_T_F(U_B, U_F, T_F);

  psmilu_assert(HC.nrows() == T_E.nrows(),
                "T_E and HC should have same row size");
  psmilu_assert(HC.ncols() == T_F.ncols(),
                "T_F and HC should have same column size");

  const size_type M = HC.ncols();
  for (size_type col = 0u; col < M; ++col) {
    auto rhs_v_itr   = T_F.val_cbegin(col);
    auto dense_v_itr = HC.col_begin(col);
    for (auto rhs_itr  = T_F.row_ind_cbegin(col),
              rhs_last = T_F.row_ind_cend(col);
         rhs_itr != rhs_last; ++rhs_itr, ++rhs_v_itr) {
      const size_type j         = c_idx(*rhs_itr);
      const auto      temp      = *rhs_v_itr;
      auto            lhs_v_itr = T_E.val_cbegin(j);
      for (auto lhs_itr = T_E.row_ind_cbegin(j), lhs_last = T_E.row_ind_cend(j);
           lhs_itr != lhs_last; ++lhs_itr, ++lhs_v_itr) {
        psmilu_assert(c_idx(*lhs_itr) < HC.nrows(),
                      "%zd exceeds the SC matrix row size", c_idx(*lhs_itr));
        dense_v_itr[c_idx(*lhs_itr)] += *lhs_v_itr * temp;
      }
    }
  }
}

}  // namespace psmilu

#endif  // _PSMILU_SCHUR_HPP
