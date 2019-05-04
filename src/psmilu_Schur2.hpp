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

#include "psmilu_Array.hpp"
#include "psmilu_SparseVec.hpp"
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
/// \param[in] A_indptr index starting position array or A
/// \param[in] m starting position of the original matrix A
/// \param[in] alpha drop space limiter threshold
/// \param[in,out] indptr in and out put index starting array
/// \param[in,out] indices in and out put index array
/// \param[in,out] vals in and out put values
/// \param[out] buf work space
/// \param[out] ibuf integer work space
/// \ingroup schur
template <bool OneBased, class IntArray, class ValueArray, class BufArray,
          class IntBufArray>
inline void drop_offsets_kernel(const IntArray &                   A_indptr,
                                const typename IntArray::size_type m,
                                const int alpha, IntArray &indptr,
                                IntArray &indices, ValueArray &vals,
                                BufArray &buf, IntBufArray &ibuf) {
  using size_type  = typename IntArray::size_type;
  using index_type = typename IntArray::value_type;

  const size_type n = indptr.size() - 1;

  // loop starts here
  auto &gaps = ibuf;
  for (size_type i(0); i < n; ++i) {
    const size_type A_sz     = A_indptr[i + m + 1] - A_indptr[i + m],
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
  if (gaps[n]) {
    auto i_itr = indices.begin();
    auto v_itr = vals.begin();
    auto prev  = indptr[0];
    for (size_type i(0); i < n; ++i) {
      const size_type first = prev - OneBased, last = indptr[i + 1] - OneBased;
      auto            itr_bak = i_itr;
      i_itr                   = std::copy(indices.cbegin() + first,
                        indices.cbegin() + last - gaps[i], i_itr);
      v_itr = std::copy(vals.cbegin() + first, vals.cbegin() + last - gaps[i],
                        v_itr);
      prev  = indptr[i + 1];
      indptr[i + 1] = indptr[i] + (i_itr - itr_bak);
    }
    // need to resize the nnz arrays
    indices.resize(indptr[n] - OneBased);
    vals.resize(indptr[n] - OneBased);
  }
}

}  // namespace internal

/// \brief drop \a L_E
/// \tparam CrsType crs storage type, see \ref CRS
/// \tparam BufArray value buffer type
/// \tparam IntBufArray integer buffer type
/// \param[in] A input matrix
/// \param[in] m starting index of offsets
/// \param[in] alpha drop space limiter threshold
/// \param[in,out] L_E in and out puts of L offset
/// \param[out] buf work space
/// \param[out] ibuf integer work space
/// \ingroup schur
/// \note buffers can be got from Crout work spaces
/// \sa drop_U_F
template <class CrsType, class BufArray, class IntBufArray>
inline void drop_L_E(const CrsType &A, const typename CrsType::size_type m,
                     const int alpha, CrsType &L_E, BufArray &buf,
                     IntBufArray &ibuf) {
  static_assert(CrsType::ROW_MAJOR, "must be CRS");

  if (A.nrows() > m) {
    if (alpha > 0)
      internal::drop_offsets_kernel<CrsType::ONE_BASED>(
          A.row_start(), m, alpha, L_E.row_start(), L_E.col_ind(), L_E.vals(),
          buf, ibuf);
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
/// \tparam CcsType ccs storage type, see \ref CCS
/// \tparam BufArray value buffer type
/// \tparam IntBufArray integer buffer type
/// \param[in] A input matrix
/// \param[in] m starting index of offsets
/// \param[in] alpha drop space limiter threshold
/// \param[in,out] U_F in and out puts of U offset
/// \param[out] buf work space
/// \param[out] ibuf integer work space
/// \ingroup schur
/// \note buffers can be got from Crout work spaces
/// \sa drop_L_E
template <class CcsType, class BufArray, class IntBufArray>
inline void drop_U_F(const CcsType &A, const typename CcsType::size_type m,
                     const int alpha, CcsType &U_F, BufArray &buf,
                     IntBufArray &ibuf) {
  static_assert(!CcsType::ROW_MAJOR, "must be CCS");

  if (A.ncols() > m) {
    if (alpha > 0)
      internal::drop_offsets_kernel<CcsType::ONE_BASED>(
          A.col_start(), m, alpha, U_F.col_start(), U_F.row_ind(), U_F.vals(),
          buf, ibuf);
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

}  // namespace psmilu

#endif  // _PSMILU_SCHUR2_HPP