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
#include <type_traits>

#include "psmilu_log.hpp"
#include "psmilu_utils.hpp"

namespace psmilu {
namespace internal {

/// \brief drop \a L_E matrix or \a U_F matrix
/// \tparam PermType permutation type
/// \tparam OneBased if or not is Fortran based index system
/// \tparam IntArray integer array, see \ref Array
/// \tparam ValueArray value array, see \ref Array
/// \tparam BufArray value buffer array type
/// \tparam IntBufArray integer buffer array type
/// \param[in] A_indptr index starting position array or A
/// \param[in] m starting position of the original matrix A
/// \param[in] alpha drop space limiter threshold
/// \param[in,out] indptr in and out put index starting array
/// \param[in,out] indices in and out put index array
/// \param[in,out] vals in and out put values
/// \param[out] buf work space
/// \param[out] ibuf integer work space
/// \ingroup schur
template <bool OneBased, class PermType, class IntArray, class ValueArray,
          class BufArray, class IntBufArray>
inline void drop_offsets_kernel(const PermType &p, const IntArray &A_indptr,
                                const typename IntArray::size_type m,
                                const double alpha, IntArray &indptr,
                                IntArray &indices, ValueArray &vals,
                                BufArray &buf, IntBufArray &ibuf) {
  using size_type  = typename IntArray::size_type;
  using index_type = typename IntArray::value_type;

  const size_type n = indptr.size() - 1;

  // loop starts here
  auto &gaps = ibuf;
  for (size_type i(0); i < n; ++i) {
    const size_type A_sz     = A_indptr[p[i + m] + 1] - A_indptr[p[i + m]],
                    sz_thres = A_sz * alpha, nnz = indptr[i + 1] - indptr[i];
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
/// \tparam PermType permutation array type
/// \tparam CrsType crs storage type, see \ref CRS
/// \tparam BufArray value buffer type
/// \tparam IntBufArray integer buffer type
/// \param[in] p row permutation vector
/// \param[in] A input matrix
/// \param[in] m starting index of offsets
/// \param[in] alpha drop space limiter threshold
/// \param[in,out] L_E in and out puts of L offset
/// \param[out] buf work space
/// \param[out] ibuf integer work space
/// \ingroup schur
/// \note buffers can be got from Crout work spaces
/// \sa drop_U_F
template <class PermType, class CrsType, class BufArray, class IntBufArray>
inline void drop_L_E(const PermType &p, const CrsType &A,
                     const typename CrsType::size_type m, const double alpha,
                     CrsType &L_E, BufArray &buf, IntBufArray &ibuf) {
  static_assert(CrsType::ROW_MAJOR, "must be CRS");

  if (A.nrows() > m) {
    if (alpha > 0)
      internal::drop_offsets_kernel<CrsType::ONE_BASED>(
          p, A.row_start(), m, alpha, L_E.row_start(), L_E.col_ind(),
          L_E.vals(), buf, ibuf);
    else {
      for (typename CrsType::size_type i(0); i < L_E.nrows(); ++i)
        L_E.row_start()[i + 1] = CrsType::ONE_BASED;
      L_E.col_ind().resize(0);
      L_E.vals().resize(0);
    }
  }
#ifndef NDEBUG
  L_E.check_validity();
#endif
}

/// \brief drop \a U_F
/// \tparam PermType permutation array
/// \tparam CcsType ccs storage type, see \ref CCS
/// \tparam BufArray value buffer type
/// \tparam IntBufArray integer buffer type
/// \param[in] q column permutation vector
/// \param[in] A input matrix
/// \param[in] m starting index of offsets
/// \param[in] alpha drop space limiter threshold
/// \param[in,out] U_F in and out puts of U offset
/// \param[out] buf work space
/// \param[out] ibuf integer work space
/// \ingroup schur
/// \note buffers can be got from Crout work spaces
/// \sa drop_L_E
template <class PermType, class CcsType, class BufArray, class IntBufArray>
inline void drop_U_F(const PermType &q, const CcsType &A,
                     const typename CcsType::size_type m, const double alpha,
                     CcsType &U_F, BufArray &buf, IntBufArray &ibuf) {
  static_assert(!CcsType::ROW_MAJOR, "must be CCS");

  if (A.ncols() > m) {
    if (alpha > 0)
      internal::drop_offsets_kernel<CcsType::ONE_BASED>(
          q, A.col_start(), m, alpha, U_F.col_start(), U_F.row_ind(),
          U_F.vals(), buf, ibuf);
    else {
      for (typename CcsType::size_type i(0); i < U_F.ncols(); ++i)
        U_F.col_start()[i + 1] = CcsType::ONE_BASED;
      U_F.row_ind().resize(0);
      U_F.vals().resize(0);
    }
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
  using value_type                = typename CrsType::value_type;
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

}  // namespace psmilu

#endif  // _PSMILU_SCHUR2_HPP