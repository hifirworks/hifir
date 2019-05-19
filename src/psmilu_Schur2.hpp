//@HEADER
//----------------------------------------------------------------------------
//                Copyright (C) 2019 The PSMILU AUTHORS
//----------------------------------------------------------------------------
//@HEADER

/// \file psmilu_Schur2.hpp
/// \brief Routines for computing Schur complements for both S and H versions
///        with drops
/// \authors Qiao,

#ifndef _PSMILU_SCHUR2_HPP
#define _PSMILU_SCHUR2_HPP

#include <algorithm>
#include <cmath>
#include <type_traits>

#include "psmilu_Array.hpp"
#include "psmilu_log.hpp"
#include "psmilu_utils.hpp"

namespace psmilu {
namespace internal {

/// \brief drop \a L_E matrix or \a U_F matrix
/// \tparam OneBased if or not is Fortran based index system
/// \tparam IntArray integer array, see \ref Array
/// \tparam ValueArray value array, see \ref Array
/// \tparam BufArray value buffer array type
/// \tparam IntBufArray integer buffer array type
/// \param[in] ref_indptr reference starting position array
/// \param[in] alpha drop space limiter threshold
/// \param[in,out] indptr in and out put index starting array
/// \param[in,out] indices in and out put index array
/// \param[in,out] vals in and out put values
/// \param[out] buf work space
/// \param[out] ibuf integer work space
/// \ingroup schur
template <bool OneBased, class IntArray, class ValueArray, class BufArray,
          class IntBufArray>
inline void drop_offsets_kernel(const IntArray &ref_indptr, const double alpha,
                                IntArray &indptr, IntArray &indices,
                                ValueArray &vals, BufArray &buf,
                                IntBufArray &ibuf) {
  using size_type  = typename IntArray::size_type;
  using index_type = typename IntArray::value_type;

  const size_type n = indptr.size() - 1;

  // loop starts here
  auto &gaps = ibuf;
  for (size_type i(0); i < n; ++i) {
    const size_type A_sz     = ref_indptr[i + 1] - ref_indptr[i],
                    sz_thres = std::ceil(alpha * A_sz),
                    nnz      = indptr[i + 1] - indptr[i];
    if (sz_thres >= nnz) {
      // if the threshold is no smaller than the local nnz
      gaps[i] = 0;
      continue;
    }
    gaps[i] = nnz - sz_thres;
    // fetch the value to the buffer
    const size_type first = indptr[i] - OneBased,
                    last  = indptr[i + 1] - OneBased;
    for (size_type j(first); j < last; ++j) buf[indices[j]] = vals[j];
    std::nth_element(
        indices.begin() + first, indices.begin() + first + sz_thres - 1,
        indices.begin() + last, [&](const index_type i, const index_type j) {
          return std::abs(buf[i]) > std::abs(buf[j]);
        });
    // sort
    std::sort(indices.begin() + first, indices.begin() + first + sz_thres);
    // fetch back the value
    for (size_type j(first); j < first + sz_thres; ++j)
      vals[j] = buf[indices[j]];
  }

  // compress
  auto i_itr = indices.begin();
  auto v_itr = vals.begin();
  auto prev  = indptr[0];
  for (size_type i(0); i < n; ++i) {
    const size_type first = prev - OneBased, last = indptr[i + 1] - OneBased;
    auto            itr_bak = i_itr;
    i_itr                   = std::copy(indices.cbegin() + first,
                      indices.cbegin() + last - gaps[i], i_itr);
    v_itr =
        std::copy(vals.cbegin() + first, vals.cbegin() + last - gaps[i], v_itr);
    prev          = indptr[i + 1];
    indptr[i + 1] = indptr[i] + (i_itr - itr_bak);
  }
  // need to resize the nnz arrays
  indices.resize(indptr[n] - OneBased);
  vals.resize(indptr[n] - OneBased);
}

}  // namespace internal

/// \brief drop \a L_E
/// \tparam CrsType crs storage type, see \ref CRS
/// \tparam BufArray value buffer type
/// \tparam IntBufArray integer buffer type
/// \param[in] ref_indptr reference starting array
/// \param[in] alpha drop space limiter threshold
/// \param[in,out] L_E in and out puts of L offset
/// \param[out] buf work space
/// \param[out] ibuf integer work space
/// \ingroup schur
/// \note buffers can be got from Crout work spaces
/// \sa drop_U_F
template <class CrsType, class BufArray, class IntBufArray>
inline void drop_L_E(const typename CrsType::iarray_type &ref_indptr,
                     const double alpha, CrsType &L_E, BufArray &buf,
                     IntBufArray &ibuf) {
  static_assert(CrsType::ROW_MAJOR, "must be CRS");

  if (alpha > 0)
    internal::drop_offsets_kernel<CrsType::ONE_BASED>(
        ref_indptr, alpha, L_E.row_start(), L_E.col_ind(), L_E.vals(), buf,
        ibuf);
  else {
    for (typename CrsType::size_type i(0); i < L_E.nrows(); ++i)
      L_E.row_start()[i + 1] = CrsType::ONE_BASED;
    L_E.col_ind().resize(0);
    L_E.vals().resize(0);
  }
#ifndef NDEBUG
  L_E.check_validity();
#endif
}

/// \brief drop \a U_F
/// \tparam CcsType ccs storage type, see \ref CCS
/// \tparam BufArray value buffer type
/// \tparam IntBufArray integer buffer type
/// \param[in] ref_indptr reference starting array
/// \param[in] alpha drop space limiter threshold
/// \param[in,out] U_F in and out puts of U offset
/// \param[out] buf work space
/// \param[out] ibuf integer work space
/// \ingroup schur
/// \note buffers can be got from Crout work spaces
/// \sa drop_L_E
template <class CcsType, class BufArray, class IntBufArray>
inline void drop_U_F(const typename CcsType::iarray_type &ref_indptr,
                     const double alpha, CcsType &U_F, BufArray &buf,
                     IntBufArray &ibuf) {
  static_assert(!CcsType::ROW_MAJOR, "must be CCS");

  if (alpha > 0)
    internal::drop_offsets_kernel<CcsType::ONE_BASED>(
        ref_indptr, alpha, U_F.col_start(), U_F.row_ind(), U_F.vals(), buf,
        ibuf);
  else {
    for (typename CcsType::size_type i(0); i < U_F.ncols(); ++i)
      U_F.col_start()[i + 1] = CcsType::ONE_BASED;
    U_F.row_ind().resize(0);
    U_F.vals().resize(0);
  }
#ifndef NDEBUG
  U_F.check_validity();
#endif
}

/// \brief compute the simple version of Schur complement
/// \tparam ScaleArray row/column scaling vector type, see \ref Array
/// \tparam CrsType crs matrix used for input, see \ref CRS
/// \tparam PermType permutation vector type, see \ref BiPermMatrix
/// \tparam DiagArray diagonal vector type, see \ref Array
/// \tparam SpVecType sparse vector buffer type, see \ref SparseVector
/// \param[in] s row scaling vector
/// \param[in] A input crs matrix
/// \param[in] t column scaling vector
/// \param[in] p row permutation vector
/// \param[in] q column permutation vector
/// \param[in] m leading block size of current level
/// \param[in] L_E lower part after \ref Crout update
/// \param[in] d diagonal entry that contains the leading block
/// \param[in] U_F upper part after \ref Crout update
/// \param[out] SC Schur complement of C version
/// \param[out] buf work space, can use directly passed in as from Crout bufs
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
template <class ScaleArray, class CrsType, class PermType, class DiagArray,
          class SpVecType>
inline void compute_Schur_simple(const ScaleArray &s, const CrsType &A,
                                 const ScaleArray &t, const PermType &p,
                                 const PermType &                  q,
                                 const typename CrsType::size_type m,
                                 const CrsType &L_E, const DiagArray &d,
                                 const CrsType &U_F, CrsType &SC,
                                 SpVecType &buf) {
  using size_type                 = typename CrsType::size_type;
  constexpr static bool ONE_BASED = CrsType::ONE_BASED;
  static_assert(CrsType::ROW_MAJOR, "must be CRS");

  const size_type n = A.nrows();
  if (m == n) return;

  // helper routine for index conversion
  const auto c_idx = [](const size_type i) -> size_type {
    return to_c_idx<size_type, ONE_BASED>(i);
  };

  // get the offset size
  const size_type N(n - m);
  SC.resize(N, N);
  auto &row_start = SC.row_start();
  row_start.resize(N + 1);
  row_start.front() = ONE_BASED;
  size_type tag(n);

  // step 1, determine the overall nnz
  for (size_type i(0); i < N; ++i, ++tag) {
    buf.reset_counter();  // reset buffer counter
    auto L_last = L_E.col_ind_cend(i);
    for (auto L_itr = L_E.col_ind_cbegin(i); L_itr != L_last; ++L_itr) {
      const auto idx = c_idx(*L_itr);
      // get the U iter
      auto U_last = U_F.col_ind_cend(idx);
      for (auto U_itr = U_F.col_ind_cbegin(idx); U_itr != U_last; ++U_itr)
        buf.push_back(*U_itr, tag);
    }
    // now add the C contribution
    const size_type pi = p[m + i];
    // since A is permutated, we must loop through all entries in this row
    for (auto itr = A.col_ind_cbegin(pi), last = A.col_ind_cend(pi);
         itr != last; ++itr) {
      const size_type inv_q = q.inv(c_idx(*itr));
      if (inv_q >= m) buf.push_back(inv_q + ONE_BASED - m, tag);
    }
    // assemble row_start
    row_start[i + 1] = row_start[i] + buf.size();
  }

  if (row_start[N] == ONE_BASED) {
    psmilu_warning("Schur (S version) complement becomes empty!");
    return;
  }

  // step 2, fill in the indices
  SC.reserve(row_start[N] - ONE_BASED);
  auto &col_ind = SC.col_ind();
  col_ind.resize(0);  // just reset the internal counter, O(1) for Array
  for (size_type i(0); i < N; ++i, ++tag) {
    buf.reset_counter();  // reset buffer counter
    auto L_last = L_E.col_ind_cend(i);
    for (auto L_itr = L_E.col_ind_cbegin(i); L_itr != L_last; ++L_itr) {
      const auto idx = c_idx(*L_itr);
      // get the U iter
      auto U_last = U_F.col_ind_cend(idx);
      for (auto U_itr = U_F.col_ind_cbegin(idx); U_itr != U_last; ++U_itr)
        buf.push_back(*U_itr, tag);
    }
    // now add the C contribution
    const size_type pi = p[m + i];
    // since A is permutated, we must loop through all entries in this row
    for (auto itr = A.col_ind_cbegin(pi), last = A.col_ind_cend(pi);
         itr != last; ++itr) {
      const size_type inv_q = q.inv(c_idx(*itr));
      if (inv_q >= m) buf.push_back(inv_q + ONE_BASED - m, tag);
    }
    // we need the ensure that the indices are sorted, because it will be the
    // input for next level
    buf.sort_indices();
    size_type j = col_ind.size();    // current start index
    col_ind.resize(j + buf.size());  // resize
    for (auto ind_itr = buf.inds().cbegin(); j < col_ind.size(); ++j, ++ind_itr)
      col_ind[j] = *ind_itr;
  }

  psmilu_assert(col_ind.size() + ONE_BASED == (size_type)row_start[N],
                "fatal error!");

  // step 3, fill in the numerical values
  auto &buf_vals = buf.vals();
  auto &vals     = SC.vals();
  vals.resize(col_ind.size());
  for (size_type i(0); i < N; ++i) {
    // first, assign values to zero
    auto sc_ind_first = SC.col_ind_cbegin(i);
    for (auto itr = sc_ind_first, ind_last = SC.col_ind_cend(i);
         itr != ind_last; ++itr) {
      psmilu_assert(c_idx(*itr) < N, "%zd exceed value size", c_idx(*itr));
      buf_vals[c_idx(*itr)] = 0.0;
    }

    // second, fetch C into the buffer
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

    // third, compute -L_E*D*U_F
    auto L_last  = L_E.col_ind_cend(i);
    auto L_v_itr = L_E.val_cbegin(i);
    for (auto L_itr = L_E.col_ind_cbegin(i); L_itr != L_last;
         ++L_itr, ++L_v_itr) {
      const auto idx = c_idx(*L_itr);
      const auto ld  = *L_v_itr * d[idx];
      // get the U iter
      auto U_last  = U_F.col_ind_cend(idx);
      auto U_v_itr = U_F.val_cbegin(idx);
      for (auto U_itr = U_F.col_ind_cbegin(idx); U_itr != U_last;
           ++U_itr, ++U_v_itr)
        buf_vals[c_idx(*U_itr)] -= ld * *U_v_itr;
    }

    // finally, assign values directly to SC
    auto sc_val_itr = SC.val_begin(i);
    for (auto itr = sc_ind_first, ind_last = SC.col_ind_cend(i);
         itr != ind_last; ++itr, ++sc_val_itr)
      *sc_val_itr = buf_vals[c_idx(*itr)];
  }
}

namespace internal {
/// \brief compute the first part of H version, i.e. \f$\boldsymbol{T}_E\f$
/// \tparam CrsType crs type, see \ref CRS
/// \tparam CcsType ccs type, see \ref CCS
/// \tparam ScaleArray scaling array type, see \ref Array
/// \tparam PermType permutation matrix type, see \ref BiPermMatrix
/// \tparam DiagType diagonal vector type, see \ref Array
/// \tparam SpVecType work space type, see \ref SparseVector
/// \param[in] L_E_ \a L_E part of partial ILU factorization
/// \param[in] L_B lower part of leading block
/// \param[in] s row scaling vector
/// \param[in] A input matrix in CCS format
/// \param[in] t column scaling vector
/// \param[in] p row permutation matrix
/// \param[in] q column permutation matrix
/// \param[in] d diagonal entries (permutated)
/// \param[in] U_B upper part of leading block
/// \param[out] T_E computed first part of H version
/// \param[in,out] buf work space
/// \param[in] start_tag starting dense tag value
/// \ingroup schur
/// \sa compute_Schur_hybrid_T_F
template <class CrsType, class CcsType, class ScaleArray, class PermType,
          class DiagType, class SpVecType>
inline void compute_Schur_hybrid_T_E(
    const CrsType &L_E_, const CcsType &L_B, const ScaleArray &s,
    const CcsType &A, const ScaleArray &t, const PermType &p, const PermType &q,
    const DiagType &d, const CcsType &U_B, CcsType &T_E, SpVecType &buf,
    const typename CcsType::size_type start_tag) {
  using size_type                 = typename CcsType::size_type;
  using index_type                = typename CcsType::index_type;
  constexpr static bool ONE_BASED = CcsType::ONE_BASED;
  using extractor                 = internal::SpVInternalExtractor<SpVecType>;

  // get the size of system
  // m is the leading block size and n is the leftover size
  const size_type m = L_B.nrows();
  psmilu_assert(m == L_B.ncols(), "must be squared system for L_B");
  const size_type n = L_E_.nrows();

  if (!n) return;

  const auto c_idx = [](size_type i) {
    return to_c_idx<size_type, ONE_BASED>(i);
  };
  const auto ori_idx = [](size_type i) {
    return to_ori_idx<size_type, ONE_BASED>(i);
  };

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
                                          SpVecType &     v) -> size_type {
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
      [&, m](const size_type tag, const size_type start_index, SpVecType &v) {
        // loop from the first nonzero index
        for (size_type j = start_index; j < m; ++j) {
          // by comparing the tags, we know if or not the current entry is
          // zero.
          // NOTE the d_tags is dynamically changing!
          if (static_cast<size_type>(d_tags[j]) != tag) continue;
          // x_j is nz, then for each nz in L(:,j), we known there must be an
          // entry in the solution
          auto last = L_B.row_ind_cend(j);
          for (auto itr = L_B.row_ind_cbegin(j); itr != last; ++itr)
            v.push_back(*itr, tag);  // NOTE this may update d_tags
        }
      };

  // the following lambda is the kernel for updating the symbolic pattern for
  // subtracting D*U from inv(L)*B
  const auto subtract_du_symbolic = [&, m](const size_type col,
                                           const size_type tag, SpVecType &v) {
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

  size_type tag(start_tag);

  // step 1, determine the number of nonzeros
  for (size_type col = 0u; col < m; ++col, ++tag) {
    // load rhs b=B(:,col) and determine the start index
    const size_type start_index = load_A_col_symbolic(col, tag, buf);
    // compute inv(L)*b
    solve_col_symbolic(tag, start_index, buf);
    // subtract inv(L)*b-(D*U)(:,col)
    subtract_du_symbolic(col, tag, buf);
    // update the col start array by using the size in sparse vector
    col_start[col + 1] = col_start[col] + buf.size();
  }

  // check if empty
  if (!(col_start[m] - ONE_BASED)) {
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
  T_E2.reserve(col_start[m] - ONE_BASED);
  auto &row_ind = T_E2.row_ind();
  auto &vals    = T_E2.vals();
  psmilu_error_if(row_ind.status() == DATA_UNDEF || vals.status() == DATA_UNDEF,
                  "memory allocation failed!");
  // for Array, this is just restting the internal size counter
  row_ind.resize(col_start[m] - ONE_BASED);
  auto i_itr = row_ind.begin();
  // redo the loop above, this time, we also push back the indices
  for (size_type col = 0u; col < m; ++col, ++tag) {
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
                                     SpVecType &v) {
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
        vals[p_inv] = s[j] * *v_itr * t_q;  // assign value her
        if (min_idx > p_inv) min_idx = p_inv;
      }
    }
    return min_idx;
  };

  // the following kernel is for solving forward sub, with the algorithm
  // described in the symbolic kernel.
  const auto solve_col_num = [&, m](const size_type tag,
                                    const size_type start_index, SpVecType &v) {
    auto &vals = v.vals();  // get reference to dense value
    for (size_type j = start_index; j < m; ++j) {
      if (static_cast<size_type>(d_tags[j]) != tag) continue;
      const auto x_j   = vals[j];
      auto       last  = L_B.row_ind_cend(j);
      auto       v_itr = L_B.val_cbegin(j);
      // for each nz in L(:,j), we push the index back
      // if a new value is added to the sparse vector (return true), then
      // we assign -L(*itr,j)*x_j, ow subtract L(*itr,j)
      for (auto itr = L_B.row_ind_cbegin(j); itr != last; ++itr, ++v_itr)
        v.push_back(*itr, tag) ? vals[c_idx(*itr)] = -*v_itr * x_j
                               : vals[c_idx(*itr)] -= *v_itr * x_j;
    }
  };

  // numerical kernel for subtracting (D*U)(:,col)
  const auto subtract_du_num = [&, m](const size_type col, const size_type tag,
                                      SpVecType &v) {
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
  vals.resize(col_start[m] - ONE_BASED);
  auto v_itr = vals.begin();
  // get the buf value reference
  auto &buf_vals = buf.vals();
  for (size_type col = 0u; col < m; ++col, ++tag) {
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
  // conver L_E_ to ccs
  CcsType L_E(L_E_);
  // loop begins, for each column (j)
  // This the first step in symbolic computation, to determine the nnz in the
  // T_E matrix
  for (size_type col = 0u; col < m; ++col, ++tag) {
    buf.reset_counter();  // reset buffer size counter
    auto rhs_itr = T_E2.row_ind_cbegin(col);
    // for each nz in T_E2(:,col)
    for (auto rhs_last = T_E2.row_ind_cend(col); rhs_itr != rhs_last;
         ++rhs_itr) {
      // we use the starting position array to, virtually, split L_E from
      // the whole L from Crout update
      const size_type k       = c_idx(*rhs_itr);
      auto            lhs_itr = L_E.row_ind_cbegin(k);
      // for each nz in L_E(:,k)
      for (auto lhs_last = L_E.row_ind_cend(k); lhs_itr != lhs_last; ++lhs_itr)
        buf.push_back(*lhs_itr, tag);
    }
    o_col_start[col + 1] = o_col_start[col] + buf.size();
  }

  if (!(o_col_start[m] - ONE_BASED)) {
    psmilu_warning("H version of Schur complement has an empty T_E part!");
    return;
  }

  T_E.reserve(o_col_start[m] - ONE_BASED);
  auto &o_row_ind = T_E.row_ind();
  auto &o_vals    = T_E.vals();
  psmilu_error_if(
      o_row_ind.status() == DATA_UNDEF || o_vals.status() == DATA_UNDEF,
      "memory allocation failed!");
  o_row_ind.resize(o_col_start[m] - ONE_BASED);
  auto o_itr = o_row_ind.begin();
  // redo the loop above to fill in the indices, aka symbolic pattern
  for (size_type col = 0u; col < m; ++col, ++tag) {
    buf.reset_counter();
    auto rhs_itr = T_E2.row_ind_cbegin(col);
    for (auto rhs_last = T_E2.row_ind_cend(col); rhs_itr != rhs_last;
         ++rhs_itr) {
      const size_type j       = c_idx(*rhs_itr);
      auto            lhs_itr = L_E.row_ind_cbegin(j);
      for (auto lhs_last = L_E.row_ind_cend(j); lhs_itr != lhs_last; ++lhs_itr)
        buf.push_back(*lhs_itr, tag);
    }
    // sort indices and push the index pattern to the row_ind array
    buf.sort_indices();
    o_itr =
        std::copy(buf.inds().cbegin(), buf.inds().cbegin() + buf.size(), o_itr);
  }

  // finally, we compute the numerical values of L_E*T_E2

  o_vals.resize(o_col_start[m] - ONE_BASED);
  auto o_v_itr = o_vals.begin();
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
      auto            lhs_itr   = L_E.row_ind_cbegin(j);
      auto            lhs_v_itr = L_E.val_cbegin(j);
      const auto      temp      = *rhs_v_itr;  // break strong reference
      for (auto lhs_last = L_E.row_ind_cend(j); lhs_itr != lhs_last;
           ++lhs_itr, ++lhs_v_itr)
        buf_vals[c_idx(*lhs_itr)] += *lhs_v_itr * temp;
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

/// \brief compute the second part of the H version
/// \tparam CcsType ccs type, see \ref CCS
/// \tparam SpVecType work space type, see \ref SparseVector
/// \param[in] U_B strict upper part
/// \param[in] U_F \a U_F part of the partial ILU factorization
/// \param[out] T_F second part
/// \param[in,out] buf work space
/// \param[in] start_tag starting dense tag value
/// \ingroup schur
/// \sa compute_Schur_hybrid_T_E
template <class CcsType, class SpVecType>
inline void compute_Schur_hybrid_T_F(
    const CcsType &U_B, const CcsType &U_F, CcsType &T_F, SpVecType &buf,
    const typename CcsType::size_type start_tag) {
  using size_type = typename CcsType::size_type;
  using iterator  = std::reverse_iterator<decltype(U_B.row_ind_cbegin(0))>;
  constexpr static bool ONE_BASED = CcsType::ONE_BASED;
  using sp_vec_type               = SpVecType;
  using extractor                 = internal::SpVInternalExtractor<sp_vec_type>;
  const auto c_idx                = [](const size_type i) {
    return to_c_idx<size_type, ONE_BASED>(i);
  };

  const size_type m = U_B.nrows();
  psmilu_assert(m == U_B.ncols(), "U should be squared matrix");
  const size_type n = U_F.ncols();

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
  size_type tag(start_tag);

  // symbolically loading the rhs, i.e. U_F(:, col)
  // since U_F is virtually splitted from the whole U from Crout update by
  // utilizing the augmented data structure with starting index m (leading
  // block size), loading the values requires using augmented API
  const auto load_U_col_symbolic = [&, m](const size_type col,
                                          const size_type tag, sp_vec_type &v) {
    v.reset_counter();  // reset counter in buffer
    for (auto itr = U_F.row_ind_cbegin(col), last = U_F.row_ind_cend(col);
         itr != last; ++itr)
      v.push_back(*itr, tag);
  };

  // extract the dense tags, which is used in backward sub
  const auto &d_tags = static_cast<const extractor &>(buf).dense_tags();

  // symbolic kernel for solving inv(U)*b, where b=U_F(:,col)
  const auto solve_col_symbolic =
      [&](const size_type tag, const size_type start_index, sp_vec_type &v) {
        // note for first entry, no new entries will be added
        for (size_type j = start_index; j != 0u; --j) {
          if (static_cast<size_type>(d_tags[j]) != tag) continue;
          if (!U_B.nnz_in_col(j)) continue;
          auto last = iterator(U_B.row_ind_cbegin(j));
          auto itr  = iterator(U_B.row_ind_cend(j));
          for (; itr != last; ++itr) v.push_back(*itr, tag);
        }
      };

  for (size_type col = 0u; col < n; ++col, ++tag) {
    load_U_col_symbolic(col, tag, buf);
    // NOTE the last index is just the index value of the last entry in buffer
    // since we know that buf is reset in loading U and augmented loop ensures
    // the resulting index array is sorted.
    if (buf.size())
      solve_col_symbolic(tag, c_idx(buf.c_idx(buf.size() - 1)), buf);
    col_start[col + 1] = col_start[col] + buf.size();  // build col_start
  }

  if (!(col_start[n] - ONE_BASED)) {
    psmilu_warning("H version of Schur complement has an empty T_F part!");
    return;
  }

  // reserve the nnz arrays
  T_F.reserve(col_start[n] - ONE_BASED);
  auto &row_ind = T_F.row_ind();
  auto &vals    = T_F.vals();
  psmilu_error_if(row_ind.status() == DATA_UNDEF || vals.status() == DATA_UNDEF,
                  "memory allocation failed!");
  row_ind.reserve(col_start[n] - ONE_BASED);
  auto i_itr = row_ind.begin();
  // same loop but for indices
  for (size_type col = 0u; col < n; ++col, ++tag) {
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
    v.reset_counter();
    auto v_itr = U_F.val_cbegin(col);
    for (auto itr = U_F.row_ind_cbegin(col), last = U_F.row_ind_cend(col);
         itr != last; ++itr, ++v_itr) {
      v.push_back(*itr, tag);
      v.vals()[c_idx(*itr)] = *v_itr;
    }
  };

  // kernel to solve for inv(U)*b numerically
  const auto solve_col_num = [&](const size_type tag,
                                 const size_type start_index, sp_vec_type &v) {
    using v_iterator = std::reverse_iterator<decltype(U_B.val_cbegin(0))>;
    auto &buf_vals   = v.vals();
    for (size_type j = start_index; j != 0u; --j) {
      if (static_cast<size_type>(d_tags[j]) != tag) continue;
      if (!U_B.nnz_in_col(j)) continue;
      const auto x_j   = buf_vals[j];
      auto       v_itr = v_iterator(U_B.val_cend(j));
      auto       last  = iterator(U_B.row_ind_cbegin(j));
      auto       itr   = iterator(U_B.row_ind_cend(j));
      for (; itr != last; ++itr, ++v_itr)
        v.push_back(*itr, tag) ? buf_vals[c_idx(*itr)] = -*v_itr * x_j
                               : buf_vals[c_idx(*itr)] -= *v_itr * x_j;
    }
  };

  // finally, fetch values directly from the dense value buffer
  vals.resize(col_start[n] - ONE_BASED);
  auto        v_itr    = vals.begin();
  const auto &buf_vals = buf.vals();
  for (size_type col = 0u; col < n; ++col, ++tag) {
    load_U_col_num(col, tag, buf);
    if (buf.size()) solve_col_num(tag, buf.c_idx(buf.size() - 1), buf);
    auto itr = T_F.row_ind_cbegin(col);
    for (auto last = T_F.row_ind_cend(col); itr != last; ++itr, ++v_itr)
      *v_itr = buf_vals[c_idx(*itr)];
  }
}

}  // namespace internal

/// \brief compute H version of Schur complement
/// \tparam CrsType crs type, see \ref CRS
/// \tparam CcsType ccs type, see \ref CCS
/// \tparam SchurType simple version of Schur complement
/// \tparam ScaleArray scaling array type, see \ref Array
/// \tparam PermType permutation matrix type, see \ref BiPermMatrix
/// \tparam DiagType diagonal vector type, see \ref Array
/// \tparam SpVecType work space type, see \ref SparseVector
/// \param[in] L_E \a L_E part of partial ILU factorization
/// \param[in] L_B lower part of leading block
/// \param[in] s row scaling vector
/// \param[in] A input matrix in CCS format
/// \param[in] t column scaling vector
/// \param[in] p row permutation matrix
/// \param[in] q column permutation matrix
/// \param[in] d diagonal entries (permutated)
/// \param[in] U_B upper part of leading block
/// \param[in] U_F \a U_F part of partial ILU factorization
/// \param[in] SC simple version of Schur complement
/// \param[out] buf work space
/// \return hybrid version of Schur complement
/// \ingroup schur
/// \sa compute_Schur_simple
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
template <class CrsType, class CcsType, class SchurType, class ScaleArray,
          class PermType, class DiagType, class SpVecType>
inline CcsType compute_Schur_hybrid(const CrsType &L_E, const CcsType &L_B,
                                    const ScaleArray &s, const CcsType &A,
                                    const ScaleArray &t, const PermType &p,
                                    const PermType &q, const DiagType &d,
                                    const CcsType &U_B, const CcsType &U_F,
                                    const SchurType &SC, SpVecType &buf) {
  static_assert(!CcsType::ROW_MAJOR, "Should be CCS type");
  static_assert(CrsType::ROW_MAJOR, "Should be CRS type");
  static_assert(!CcsType::ROW_MAJOR, "A should be CCS type");
  static_assert(!(CcsType::ONE_BASED ^ CrsType::ONE_BASED),
                "inconsistent index system");
  using size_type                 = typename CcsType::size_type;
  constexpr static bool ONE_BASED = CcsType::ONE_BASED;
  CcsType               HC;
  if (!SC.nrows()) return HC;
  const auto c_idx = [](size_type i) {
    return to_c_idx<size_type, ONE_BASED>(i);
  };

  CcsType T_E, T_F;
  // NOTE that we call compute TE first to ensure the dense tag
  internal::compute_Schur_hybrid_T_E(L_E, L_B, s, A, t, p, q, d, U_B, T_E, buf,
                                     A.nrows());
  // then compute TF
  internal::compute_Schur_hybrid_T_F(U_B, U_F, T_F, buf, A.nrows() * 2);
  // compute HC=SC+T_E*T_F
  size_type tag(A.nrows() * 3);
  // convert (might be shallow) to CCS
  const CcsType   S_C(SC);
  const size_type n = SC.nrows();
  HC.resize(n, n);
  auto &col_start = HC.col_start();
  col_start.resize(n + 1);
  psmilu_error_if(col_start.status() == DATA_UNDEF, "memory allocation failed");
  col_start.front() = ONE_BASED;

  // step 1, determine the overall nnz
  for (size_type i(0); i < n; ++i, ++tag) {
    buf.reset_counter();
    // load SC
    for (auto itr = S_C.row_ind_cbegin(i), last = S_C.row_ind_cend(i);
         itr != last; ++itr)
      buf.push_back(*itr, tag);
    // load column of T_E*T_F
    for (auto itr1 = T_F.row_ind_cbegin(i), last1 = T_F.row_ind_cend(i);
         itr1 != last1; ++itr1) {
      const size_type j = c_idx(*itr1);
      for (auto itr2 = T_E.row_ind_cbegin(j), last2 = T_E.row_ind_cend(j);
           itr2 != last2; ++itr2)
        buf.push_back(*itr2, tag);
    }
    col_start[i + 1] = col_start[i] + buf.size();
  }

  if (!(col_start[n] - ONE_BASED)) {
    std::fill_n(col_start.begin(), n, ONE_BASED);
    return HC;
  }

  auto &row_ind = HC.row_ind();
  auto &vals    = HC.vals();
  row_ind.resize(col_start[n] - ONE_BASED);
  vals.resize(col_start[n] - ONE_BASED);
  psmilu_error_if(row_ind.status() == DATA_UNDEF || vals.status() == DATA_UNDEF,
                  "memory allocation failed");
  auto i_itr = row_ind.begin();
  auto v_itr = vals.begin();

  // step 2 fill in the indices and values
  for (size_type i(0); i < n; ++i, ++tag) {
    buf.reset_counter();
    // load SC
    auto sc_v_itr = S_C.val_cbegin(i);
    for (auto itr = S_C.row_ind_cbegin(i), last = S_C.row_ind_cend(i);
         itr != last; ++itr, ++sc_v_itr) {
      buf.push_back(*itr, tag);
      buf.vals()[c_idx(*itr)] = *sc_v_itr;
    }
    // load column of T_E*T_F
    auto tf_v_itr = T_F.val_cbegin(i);
    for (auto itr1 = T_F.row_ind_cbegin(i), last1 = T_F.row_ind_cend(i);
         itr1 != last1; ++itr1, ++tf_v_itr) {
      const size_type j        = c_idx(*itr1);
      const auto      temp     = *tf_v_itr;
      auto            te_v_itr = T_E.val_cbegin(j);
      for (auto itr2 = T_E.row_ind_cbegin(j), last2 = T_E.row_ind_cend(j);
           itr2 != last2; ++itr2, ++te_v_itr)
        if (buf.push_back(*itr2, tag))
          buf.vals()[c_idx(*itr2)] = temp * *te_v_itr;
        else
          buf.vals()[c_idx(*itr2)] += temp * *te_v_itr;
    }
    // sort
    buf.sort_indices();
    const size_type nn = buf.size();
    i_itr = std::copy(buf.inds().cbegin(), buf.inds().cbegin() + nn, i_itr);
    for (size_type j(0); j < nn; ++j, ++v_itr) *v_itr = buf.val(j);
  }
  return HC;
}

}  // namespace psmilu

#endif  // _PSMILU_SCHUR2_HPP
