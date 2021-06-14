///////////////////////////////////////////////////////////////////////////////
//                  This file is part of the HIFIR library                         //
///////////////////////////////////////////////////////////////////////////////

/*!
 * \file hif/alg/Schur.hpp
 * \brief Routines for computing Schur complements of simple version
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

#ifndef _HIF_ALG_SCHUR_HPP
#define _HIF_ALG_SCHUR_HPP

#include <algorithm>
#include <cmath>
#include <type_traits>

#include "hif/ds/Array.hpp"
#include "hif/ds/SparseVec.hpp"
#include "hif/utils/common.hpp"
#include "hif/utils/log.hpp"
#include "hif/utils/mt.hpp"

namespace hif {
namespace internal {

/// \brief drop \a L_E matrix or \a U_F matrix
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
template <class IntArray, class ValueArray, class BufArray, class IntBufArray>
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
    const size_type first = indptr[i], last = indptr[i + 1];
    for (size_type j(first); j < last; ++j) buf[indices[j]] = vals[j];
    std::nth_element(
        indices.begin() + first, indices.begin() + first + sz_thres - 1,
        indices.begin() + last, [&](const index_type ii, const index_type jj) {
          return std::abs(buf[ii]) > std::abs(buf[jj]);
        });
    // sort
    // std::sort(indices.begin() + first, indices.begin() + first + sz_thres);
    // fetch back the value
    for (size_type j(first); j < first + sz_thres; ++j)
      vals[j] = buf[indices[j]];
  }

  // compress
  auto i_itr = indices.begin();
  auto v_itr = vals.begin();
  auto prev  = indptr[0];
  for (size_type i(0); i < n; ++i) {
    const size_type first = prev, last = indptr[i + 1];
    auto            itr_bak = i_itr;
    i_itr                   = std::copy(indices.cbegin() + first,
                      indices.cbegin() + last - gaps[i], i_itr);
    v_itr =
        std::copy(vals.cbegin() + first, vals.cbegin() + last - gaps[i], v_itr);
    prev          = indptr[i + 1];
    indptr[i + 1] = indptr[i] + (i_itr - itr_bak);
  }
  // need to resize the nnz arrays
  indices.resize(indptr[n]);
  vals.resize(indptr[n]);
}

}  // namespace internal

/*!
 * \addtogroup schur
 * @{
 */

/// \brief drop \a L_E
/// \tparam CrsType crs storage type, see \ref CRS
/// \tparam BufArray value buffer type
/// \tparam IntBufArray integer buffer type
/// \param[in] ref_indptr reference starting array
/// \param[in] alpha drop space limiter threshold
/// \param[in,out] L_E in and out puts of L offset
/// \param[out] buf work space
/// \param[out] ibuf integer work space
/// \note buffers can be got from Crout work spaces
/// \sa drop_U_F
template <class CrsType, class BufArray, class IntBufArray>
inline void drop_L_E(const typename CrsType::iarray_type &ref_indptr,
                     const double alpha, CrsType &L_E, BufArray &buf,
                     IntBufArray &ibuf) {
  static_assert(CrsType::ROW_MAJOR, "must be CRS");

  if (alpha > 0)
    internal::drop_offsets_kernel(ref_indptr, alpha, L_E.row_start(),
                                  L_E.col_ind(), L_E.vals(), buf, ibuf);
  else {
    std::fill_n(L_E.row_start().begin(), L_E.nrows() + 1, 0);
    L_E.col_ind().resize(0);
    L_E.vals().resize(0);
  }
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
/// \note buffers can be got from Crout work spaces
/// \sa drop_L_E
template <class CcsType, class BufArray, class IntBufArray>
inline void drop_U_F(const typename CcsType::iarray_type &ref_indptr,
                     const double alpha, CcsType &U_F, BufArray &buf,
                     IntBufArray &ibuf) {
  static_assert(!CcsType::ROW_MAJOR, "must be CCS");

  if (alpha > 0)
    internal::drop_offsets_kernel(ref_indptr, alpha, U_F.col_start(),
                                  U_F.row_ind(), U_F.vals(), buf, ibuf);
  else {
    std::fill_n(U_F.col_start().begin(), U_F.ncols() + 1, 0);
    U_F.row_ind().resize(0);
    U_F.vals().resize(0);
  }
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
///
/// Mathematically, this is to compute:
///
/// \f[
///   \mathbf{S}_C=\mathbf{A}[\mathbf{p}_{m+1:n},\mathbf{q}_{m+1:n}]
///     -\mathbf{L}_E\textrm{diag}(\mathbf{d_B})\mathbf{U}_F
/// \f]
///
/// Where \f$\mathbf{L}_E=\mathbf{L}_{m+1:n,:}\f$,
/// \f$\mathbf{U}_F=\mathbf{U}_{:,m+1:n}\f$. Notice that both
/// \f$\mathbf{L}\f$ and \f$\mathbf{U}\f$ are lower and upper parts
/// \b after computing Crout updates. Therefore, they are rectangle matrices
/// if \f$m<n\f$. Accessing \f$\mathbf{U}_{:,m+1:n}\f$ can be efficiently
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
  using size_type        = typename CrsType::size_type;
  using value_type       = typename CrsType::value_type;
  using boost_value_type = typename ValueTypeMixedTrait<value_type>::boost_type;
  constexpr static bool BOOST_PRECISION =
      !std::is_same<boost_value_type, value_type>::value;
  using boost_scalar_type =
      typename ValueTypeTrait<boost_value_type>::value_type;
  static_assert(CrsType::ROW_MAJOR, "must be CRS");

  const size_type n = A.nrows();
  if (m == n) return;

  // get the offset size
  const size_type N(n - m);
  SC.resize(N, N);
  auto &row_start = SC.row_start();
  row_start.resize(N + 1);
  row_start.front() = 0;
  size_type tag(n);

  // step 1, determine the overall nnz
  for (size_type i(0); i < N; ++i, ++tag) {
    buf.reset_counter();  // reset buffer counter
    auto L_last = L_E.col_ind_cend(i);
    for (auto L_itr = L_E.col_ind_cbegin(i); L_itr != L_last; ++L_itr) {
      auto U_last = U_F.col_ind_cend(*L_itr);
      for (auto U_itr = U_F.col_ind_cbegin(*L_itr); U_itr != U_last; ++U_itr)
        buf.push_back(*U_itr, tag);
    }
    // now add the C contribution
    const size_type pi = p[m + i];
    // since A is permutated, we must loop through all entries in this row
    for (auto itr = A.col_ind_cbegin(pi), last = A.col_ind_cend(pi);
         itr != last; ++itr) {
      const size_type inv_q = q.inv(*itr);
      if (inv_q >= m) buf.push_back(inv_q - m, tag);
    }
    // assemble row_start
    row_start[i + 1] = row_start[i] + buf.size();
  }

  if (row_start[N] == 0) {
    hif_warning("Schur (S version) complement becomes empty!");
    return;
  }

  // step 2, fill in the indices
  SC.reserve(row_start[N]);
  auto &col_ind = SC.col_ind();
  col_ind.resize(0);  // just reset the internal counter, O(1) for Array
  for (size_type i(0); i < N; ++i, ++tag) {
    buf.reset_counter();  // reset buffer counter
    auto L_last = L_E.col_ind_cend(i);
    for (auto L_itr = L_E.col_ind_cbegin(i); L_itr != L_last; ++L_itr) {
      // get the U iter
      auto U_last = U_F.col_ind_cend(*L_itr);
      for (auto U_itr = U_F.col_ind_cbegin(*L_itr); U_itr != U_last; ++U_itr) {
        hif_assert(size_type(*U_itr) < N, "%zd exceed Schur dimension %zd",
                   size_type(*U_itr), N);
        buf.push_back(*U_itr, tag);
      }
    }
    // now add the C contribution
    const size_type pi = p[m + i];
    // since A is permutated, we must loop through all entries in this row
    for (auto itr = A.col_ind_cbegin(pi), last = A.col_ind_cend(pi);
         itr != last; ++itr) {
      const size_type inv_q = q.inv(*itr);
      if (inv_q >= m) {
        hif_assert(inv_q - m < N,
                   "%zd exceed Schur dimension %zd, bad index %zd in A",
                   inv_q - m, N, inv_q);
        buf.push_back(inv_q - m, tag);
      }
    }
    // we need the ensure that the indices are sorted, because it will be the
    // input for next level
    buf.sort_indices();
    size_type j = col_ind.size();    // current start index
    col_ind.resize(j + buf.size());  // resize
    for (auto ind_itr = buf.inds().cbegin(); j < col_ind.size(); ++j, ++ind_itr)
      col_ind[j] = *ind_itr;
  }

  hif_assert(col_ind.size() == (size_type)row_start[N], "fatal error!");

  // step 3, fill in the numerical values
  Array<boost_value_type> _buf;
  boost_value_type *      buf_vals;
  if (BOOST_PRECISION) {
    _buf.resize(N);
    hif_error_if(_buf.status() == DATA_UNDEF, "memory allocation failed");
    buf_vals = _buf.data();
  } else
    buf_vals = (boost_value_type *)buf.vals().data();
  auto &vals = SC.vals();
  vals.resize(col_ind.size());
  for (size_type i(0); i < N; ++i) {
    // first, assign values to zero
    auto sc_ind_first = SC.col_ind_cbegin(i);
    for (auto itr = sc_ind_first, ind_last = SC.col_ind_cend(i);
         itr != ind_last; ++itr) {
      hif_assert(size_type(*itr) < N, "%zd exceed Schur dimension %zd",
                 size_type(*itr), N);
      buf_vals[*itr] = 0.0;
    }

    // second, fetch C into the buffer
    const size_type         pi        = p[m + i];
    auto                    a_val_itr = A.val_cbegin(pi);
    const boost_scalar_type s_pi      = s[pi];  // row scaling
    for (auto itr = A.col_ind_cbegin(pi), last = A.col_ind_cend(pi);
         itr != last; ++itr, ++a_val_itr) {
      const size_type A_idx = *itr;
      const size_type inv_q = q.inv(A_idx);
      // load and scale
      if (inv_q >= m) buf_vals[inv_q - m] = s_pi * *a_val_itr * t[A_idx];
    }

    // third, compute -L_E*D*U_F
    auto L_last  = L_E.col_ind_cend(i);
    auto L_v_itr = L_E.val_cbegin(i);
    for (auto L_itr = L_E.col_ind_cbegin(i); L_itr != L_last;
         ++L_itr, ++L_v_itr) {
      const auto             idx = *L_itr;
      const boost_value_type ld  = *L_v_itr * d[idx];
      // get the U iter
      auto U_last  = U_F.col_ind_cend(idx);
      auto U_v_itr = U_F.val_cbegin(idx);
      for (auto U_itr = U_F.col_ind_cbegin(idx); U_itr != U_last;
           ++U_itr, ++U_v_itr)
        buf_vals[*U_itr] -= ld * *U_v_itr;
    }

    // finally, assign values directly to SC
    auto sc_val_itr = SC.val_begin(i);
    for (auto itr = sc_ind_first, ind_last = SC.col_ind_cend(i);
         itr != ind_last; ++itr, ++sc_val_itr)
      *sc_val_itr = buf_vals[*itr];
  }
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
/// \param[out] buf work space, can use directly passed in as from Crout bufs
/// \return simple version of Schur complement
template <class ScaleArray, class CrsType, class PermType, class DiagArray,
          class SpVecType>
inline CrsType compute_Schur_simple(const ScaleArray &s, const CrsType &A,
                                    const ScaleArray &t, const PermType &p,
                                    const PermType &                  q,
                                    const typename CrsType::size_type m,
                                    const CrsType &L_E, const DiagArray &d,
                                    const CrsType &U_F, SpVecType &buf) {
  CrsType SC;
  compute_Schur_simple(s, A, t, p, q, m, L_E, d, U_F, SC, buf);
  return SC;
}

/*!
 * @}
 */

namespace mt {

/*!
 * \addtogroup schur
 * @{
 */

/// \brief drop \a L_E and \a U_F
/// \tparam CrsType crs storage type, see \ref CRS
/// \tparam CcsType ccs storage type, see \ref CCS
/// \tparam BufArray value buffer type
/// \tparam IntBufArray integer buffer type
/// \param[in] ref_indptr_L reference starting array for \a L_E
/// \param[in] alpha_L drop space limiter threshold for \a L_E
/// \param[in] ref_indptr_U reference starting array for \a U_F
/// \param[in] alpha_U drop space limiter threshold for \a U_F
/// \param[in,out] L_E in and out puts of L offset
/// \param[in,out] U_F in and out puts of U offset
/// \param[out] buf_L work space for \a L_E
/// \param[out] ibuf_L integer work space for \a L_E
/// \param[out] buf_U work space for \a U_F
/// \param[out] ibuf_U workspace for \a U_F
/// \param[in] threads total threads used, must be at least 1, not alerted
/// \note buffers can be got from Crout work spaces
template <class CrsType, class CcsType, class BufArray, class IntBufArray>
inline void drop_L_E_and_U_F(const typename CrsType::iarray_type &ref_indptr_L,
                             const double                         alpha_L,
                             const typename CcsType::iarray_type &ref_indptr_U,
                             const double alpha_U, CrsType &L_E, CcsType &U_F,
                             BufArray &buf_L, IntBufArray &ibuf_L,
                             BufArray &buf_U, IntBufArray &ibuf_U,
                             const int threads) {
  using size_type  = typename CrsType::size_type;
  using index_type = typename CrsType::index_type;

  const size_type n = L_E.nrows();
  hif_assert(n == U_F.ncols(), "mismatched sizes between L_E and U_F");
  if (threads == 1 || n <= 200u) {
    drop_L_E(ref_indptr_L, alpha_L, L_E, buf_L, ibuf_L);
    drop_U_F(ref_indptr_U, alpha_U, U_F, buf_U, ibuf_U);
    return;
  }

  // parallel
  hif_assert(threads > 1, "thread number %d must be no smaller than 1",
             threads);

  if (threads <= 3) {
#pragma omp parallel num_threads(2) default(shared)
    do {
      const int thread = get_thread();
      if (thread)
        drop_L_E(ref_indptr_L, alpha_L, L_E, buf_L, ibuf_L);
      else
        drop_U_F(ref_indptr_U, alpha_U, U_F, buf_U, ibuf_U);
    } while (false);  // end of parallel region for 2
    return;
  }

  auto &gaps_L = ibuf_L, &gaps_U = ibuf_U;
#pragma omp parallel num_threads(threads) default(shared)
  do {
    // determine thread id and work of partition
    const int       thread    = get_thread();
    const auto      part      = uniform_partition(n, threads, thread);
    const size_type start_idx = part.first, poe_idx = part.second;

    // create additional work buffer
    BufArray buf_;
    if (thread > 1) buf_.resize(L_E.ncols());
    BufArray &buf = thread < 2 ? (thread ? buf_U : buf_L) : buf_;

    // actual loop begins
    for (size_type i(start_idx); i < poe_idx; ++i) {
      // L_E part
      const size_type sz_thres_L =
          std::ceil(alpha_L * (ref_indptr_L[i + 1] - ref_indptr_L[i]));
      if (sz_thres_L >= L_E.nnz_in_row(i))
        gaps_L[i] = 0;
      else {
        gaps_L[i] = L_E.nnz_in_row(i) - sz_thres_L;
        // fetch values to buffer
        const size_type first = L_E.row_start()[i],
                        last  = L_E.row_start()[i + 1];
        for (size_type j(first); j < last; ++j)
          buf[L_E.col_ind()[j]] = L_E.vals()[j];
        // quickselect
        std::nth_element(L_E.col_ind_begin(i),
                         L_E.col_ind().begin() + first + sz_thres_L - 1,
                         L_E.col_ind_end(i),
                         [&](const index_type ii, const index_type jj) {
                           return std::abs(buf[ii]) > std::abs(buf[jj]);
                         });
        // fetch back
        for (size_type j(first); j < first + sz_thres_L; ++j)
          L_E.vals()[j] = buf[L_E.col_ind()[j]];
      }

      // U_F part
      const size_type sz_thres_U =
          std::ceil(alpha_U * (ref_indptr_U[i + 1] - ref_indptr_U[i]));
      if (sz_thres_U >= U_F.nnz_in_col(i))
        gaps_U[i] = 0;
      else {
        gaps_U[i] = U_F.nnz_in_col(i) - sz_thres_U;
        // fetch values to buffer
        const size_type first = U_F.col_start()[i],
                        last  = U_F.col_start()[i + 1];
        for (size_type j(first); j < last; ++j)
          buf[U_F.row_ind()[j]] = U_F.vals()[j];
        // quickselect
        std::nth_element(U_F.row_ind_begin(i),
                         U_F.row_ind().begin() + first + sz_thres_U - 1,
                         U_F.row_ind_end(i),
                         [&](const index_type ii, const index_type jj) {
                           return std::abs(buf[ii]) > std::abs(buf[jj]);
                         });
        // fetch back
        for (size_type j(first); j < first + sz_thres_U; ++j)
          U_F.vals()[j] = buf[U_F.row_ind()[j]];
      }
    }

#pragma omp barrier

    // compress
    if (thread == 0) {
      // L part
      auto i_itr = L_E.col_ind().begin();
      auto v_itr = L_E.vals().begin();
      auto prev  = L_E.row_start()[0];
      for (size_type i(0); i < n; ++i) {
        const size_type first = prev, last = L_E.row_start()[i + 1];
        auto            itr_bak = i_itr;
        i_itr                   = std::copy(L_E.col_ind().cbegin() + first,
                          L_E.col_ind().cbegin() + last - gaps_L[i], i_itr);
        v_itr                   = std::copy(L_E.vals().cbegin() + first,
                          L_E.vals().cbegin() + last - gaps_L[i], v_itr);
        prev                    = L_E.row_start()[i + 1];
        L_E.row_start()[i + 1]  = L_E.row_start()[i] + (i_itr - itr_bak);
      }
      // resize nnz arrays
      L_E.col_ind().resize(L_E.nnz());
      L_E.vals().resize(L_E.nnz());
    } else if (thread == 1) {
      // U part
      auto i_itr = U_F.row_ind().begin();
      auto v_itr = U_F.vals().begin();
      auto prev  = U_F.col_start()[0];
      for (size_type i(0); i < n; ++i) {
        const size_type first = prev, last = U_F.col_start()[i + 1];
        auto            itr_bak = i_itr;
        i_itr                   = std::copy(U_F.row_ind().cbegin() + first,
                          U_F.row_ind().cbegin() + last - gaps_U[i], i_itr);
        v_itr                   = std::copy(U_F.vals().cbegin() + first,
                          U_F.vals().cbegin() + last - gaps_U[i], v_itr);
        prev                    = U_F.col_start()[i + 1];
        U_F.col_start()[i + 1]  = U_F.col_start()[i] + (i_itr - itr_bak);
      }
      // resize nnz arrays
      U_F.row_ind().resize(U_F.nnz());
      U_F.vals().resize(U_F.nnz());
    }
  } while (false);  // end of parallel region
}

/// \brief compute the simple version of Schur complement in MT setting
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
/// \param[out] buf0 work space, can use directly passed in as from Crout bufs
/// \param[in] threads number of threads used, must be at least 1, not alerted
/// \return simple version of Schur complement
template <class ScaleArray, class CrsType, class PermType, class DiagArray,
          class SpVecType>
inline CrsType compute_Schur_simple(const ScaleArray &s, const CrsType &A,
                                    const ScaleArray &t, const PermType &p,
                                    const PermType &                  q,
                                    const typename CrsType::size_type m,
                                    const CrsType &L_E, const DiagArray &d,
                                    const CrsType &U_F, SpVecType &buf0,
                                    const int threads) {
  if (threads == 1)
    return hif::compute_Schur_simple(s, A, t, p, q, m, L_E, d, U_F, buf0);

  hif_assert(threads > 1, "thread number %d must be no smaller than 1",
             threads);

  using size_type        = typename CrsType::size_type;
  using value_type       = typename CrsType::value_type;
  using boost_value_type = typename ValueTypeMixedTrait<value_type>::boost_type;
  using boost_sparse_vec_type =
      SparseVector<boost_value_type, typename CrsType::index_type>;
  using boost_scalar_type =
      typename ValueTypeTrait<boost_value_type>::value_type;

  const size_type n = A.nrows();
  if (m == n) return CrsType();

  const size_type N(n - m);
  // if running in parallel is efficient?
  if (N <= 200u || L_E.nnz() <= 10000u || U_F.nnz() <= 10000u)
    return hif::compute_Schur_simple(s, A, t, p, q, m, L_E, d, U_F, buf0);

  // parallel part
  CrsType SC;
  SC.resize(N, N);
  auto &row_start = SC.row_start();
  row_start.resize(N + 1);
  row_start.front() = 0;

#pragma omp parallel num_threads(threads) default(shared)
  do {
    const int       thread    = get_thread();
    const auto      part      = uniform_partition(N, threads, thread);
    const size_type start_idx = part.first, poe_idx = part.second;

    // create local sparse buffer
    boost_sparse_vec_type buf(N);
    // SpVecType buf_;
    // if (thread) buf_.resize(N);
    // SpVecType &buf = thread ? buf_ : buf0;
    size_type tag(n);

    // step 1, determine the overall nnz
    for (size_type i(start_idx); i < poe_idx; ++i, ++tag) {
      buf.reset_counter();  // reset buffer counter
      auto L_last = L_E.col_ind_cend(i);
      for (auto L_itr = L_E.col_ind_cbegin(i); L_itr != L_last; ++L_itr) {
        auto U_last = U_F.col_ind_cend(*L_itr);
        for (auto U_itr = U_F.col_ind_cbegin(*L_itr); U_itr != U_last; ++U_itr)
          buf.push_back(*U_itr, tag);
      }
      // now add the C contribution
      const size_type pi = p[m + i];
      // since A is permutated, we must loop through all entries in this row
      for (auto itr = A.col_ind_cbegin(pi), last = A.col_ind_cend(pi);
           itr != last; ++itr) {
        const size_type inv_q = q.inv(*itr);
        if (inv_q >= m) buf.push_back(inv_q - m, tag);
      }
      // assemble row_start
      // NOTE we only get the local value here
      row_start[i + 1] = buf.size();
    }

    // bottleneck

#pragma omp barrier
#pragma omp master
    do {
      for (size_type i(0); i < N; ++i) row_start[i + 1] += row_start[i];
      if (row_start[N]) {
        SC.reserve(row_start[N]);
        SC.col_ind().resize(row_start[N]);
        SC.vals().resize(row_start[N]);
      }
    } while (false);
#pragma omp barrier
    if (row_start[N] == 0) break;

    // step 2, fill in the indices
    for (size_type i(start_idx); i < poe_idx; ++i, ++tag) {
      buf.reset_counter();  // reset buffer counter
      auto L_last = L_E.col_ind_cend(i);
      for (auto L_itr = L_E.col_ind_cbegin(i); L_itr != L_last; ++L_itr) {
        // get the U iter
        auto U_last = U_F.col_ind_cend(*L_itr);
        for (auto U_itr = U_F.col_ind_cbegin(*L_itr); U_itr != U_last; ++U_itr)
          buf.push_back(*U_itr, tag);
      }
      // now add the C contribution
      const size_type pi = p[m + i];
      // since A is permutated, we must loop through all entries in this row
      for (auto itr = A.col_ind_cbegin(pi), last = A.col_ind_cend(pi);
           itr != last; ++itr) {
        const size_type inv_q = q.inv(*itr);
        if (inv_q >= m) buf.push_back(inv_q - m, tag);
      }
      // we need the ensure that the indices are sorted, because it will be the
      // input for next level
      buf.sort_indices();
      std::copy_n(buf.inds().cbegin(), buf.size(), SC.col_ind_begin(i));
    }

    // step 3, fill in the numerical values
    auto &buf_vals = buf.vals();
    for (size_type i(start_idx); i < poe_idx; ++i) {
      // first, assign values to zero
      auto sc_ind_first = SC.col_ind_cbegin(i);
      for (auto itr = sc_ind_first, ind_last = SC.col_ind_cend(i);
           itr != ind_last; ++itr)
        buf_vals[*itr] = 0.0;

      // second, fetch C into the buffer
      const size_type         pi        = p[m + i];
      auto                    a_val_itr = A.val_cbegin(pi);
      const boost_scalar_type s_pi      = s[pi];  // row scaling
      for (auto itr = A.col_ind_cbegin(pi), last = A.col_ind_cend(pi);
           itr != last; ++itr, ++a_val_itr) {
        const size_type A_idx = *itr;
        const size_type inv_q = q.inv(A_idx);
        // load and scale
        if (inv_q >= m) buf_vals[inv_q - m] = s_pi * *a_val_itr * t[A_idx];
      }

      // third, compute -L_E*D*U_F
      auto L_last  = L_E.col_ind_cend(i);
      auto L_v_itr = L_E.val_cbegin(i);
      for (auto L_itr = L_E.col_ind_cbegin(i); L_itr != L_last;
           ++L_itr, ++L_v_itr) {
        const auto             idx = *L_itr;
        const boost_value_type ld  = *L_v_itr * d[idx];
        // get the U iter
        auto U_last  = U_F.col_ind_cend(idx);
        auto U_v_itr = U_F.val_cbegin(idx);
        for (auto U_itr = U_F.col_ind_cbegin(idx); U_itr != U_last;
             ++U_itr, ++U_v_itr)
          buf_vals[*U_itr] -= ld * *U_v_itr;
      }

      // finally, assign values directly to SC
      auto sc_val_itr = SC.val_begin(i);
      for (auto itr = sc_ind_first, ind_last = SC.col_ind_cend(i);
           itr != ind_last; ++itr, ++sc_val_itr)
        *sc_val_itr = buf_vals[*itr];
    }
  } while (false);  // end of parallel region

  return SC;
}

/*!
 * @}
 */

}  // namespace mt
}  // namespace hif

#endif  // HIF_ALG_SCHUR_HPP
