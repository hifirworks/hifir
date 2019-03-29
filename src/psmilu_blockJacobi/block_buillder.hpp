//@HEADER
//----------------------------------------------------------------------------
//                Copyright (C) 2019 The PSMILU AUTHORS
//----------------------------------------------------------------------------
//@HEADER

/// \file psmilu_blockJacobi/block_builder.hpp
/// \brief Interface(s) for building Jacobi blocks
/// \authors Qiao,

#ifndef _PSMILU_BLOCKJACOBI_BLOCKBUILDER_HPP
#define _PSMILU_BLOCKJACOBI_BLOCKBUILDER_HPP

#include <algorithm>
#include <type_traits>

#include "psmilu_Array.hpp"
#include "psmilu_log.hpp"
#include "psmilu_utils.hpp"

namespace psmilu {
namespace bjacobi {

namespace internal {
template <bool OneBased, class IndexArray, class ValueArray>
inline void extract_block(const IndexArray &i_indptr,
                          const IndexArray &i_indices, const ValueArray &i_vals,
                          const typename IndexArray::size_type start,
                          const typename IndexArray::size_type end,
                          IndexArray &indptr, IndexArray &indices,
                          ValueArray &vals) {
  using size_type  = typename IndexArray::size_type;
  using index_type = typename IndexArray::value_type;

  const size_type n = end - start;
  indptr.resize(n + 1);
  psmilu_error_if(indptr.status() == DATA_UNDEF, "memory allocation failed");

  auto i_begin = i_indices.cbegin();  // get the starting iterator

  // in case we may need another array to store positions
  IndexArray sec_pos;
  int        flag = 0;
  if (start == 0u)
    flag = 1;
  else if (end + 1u == i_indptr.size())
    flag = -1;
  if (!flag) {
    sec_pos.resize(n);
    psmilu_error_if(sec_pos.status() == DATA_UNDEF, "memory allocation failed");
  }

  size_type         nnz(0);
  const index_type *start_pos(nullptr), *end_pos(nullptr);

  if (!flag) {
    for (size_type i(start); i < end; ++i) {
      auto first            = i_begin + i_indptr[i] - OneBased,
           last             = i_begin + i_indptr[i + 1] - OneBased;
      auto info1            = find_sorted(first, last, start + OneBased);
      indptr[i - start + 1] = info1.second - i_begin + OneBased;
      auto info2            = find_sorted(first, last, end + OneBased);
      sec_pos[i - start]    = info2.second - i_begin + OneBased;
      nnz += info2.second - info1.second;
    }
    // make the position
    start_pos = indptr.data() + 1;
    end_pos   = sec_pos.data();
  } else if (flag == 1) {
    for (size_type i = 0u; i < end; ++i) {
      auto first    = i_begin + i_indptr[i] - OneBased,
           last     = i_begin + i_indptr[i + 1] - OneBased;
      auto info2    = find_sorted(first, last, end + OneBased);
      indptr[i + 1] = info2.second - i_begin + OneBased;
      nnz += info2.second - first;
    }
    start_pos = i_indptr.data();
    end_pos   = indptr.data() + 1;
  } else {
    for (size_type i = start; i < end; ++i) {
      auto first            = i_begin + i_indptr[i] - OneBased,
           last             = i_begin + i_indptr[i + 1] - OneBased;
      auto info1            = find_sorted(first, last, start + OneBased);
      indptr[i - start + 1] = info1.second - i_begin + OneBased;
      nnz += last - info1.second;
    }
    start_pos = indptr.data() + 1;
    end_pos   = i_indptr.data() + 1 + start;
  }

  // reserve space for vals and indices
  vals.resize(nnz);
  indices.resize(nnz);
  psmilu_error_if(vals.status() == DATA_UNDEF || indices.status() == DATA_UNDEF,
                  "memory allocation failed");

  auto i_itr = indices.begin();
  auto v_itr = vals.begin();

  auto v_begin = i_vals.cbegin();

  // note that all positions start with second element
  indptr.front()     = OneBased;
  const size_type nn = end - start;
  for (size_type i(0); i < nn; ++i) {
    auto bak_itr              = i_itr;
    i_itr                     = std::copy(i_begin + start_pos[i] - OneBased,
                      i_begin + end_pos[i] - OneBased, i_itr);
    const size_type local_nnz = i_itr - bak_itr;
    // NOTE that OneBase is automatically satisfied
    std::for_each(bak_itr, i_itr, [=](index_type &j) { j -= start; });
    v_itr = std::copy(v_begin + start_pos[i] - OneBased,
                      v_begin + end_pos[i] - OneBased, v_itr);
    // safe to update indptr
    indptr[i + 1] = indptr[i] + local_nnz;
  }
  psmilu_assert((size_type)indptr[n] == nnz + OneBased, "fatal");
}
}  // namespace internal

template <class CsType, class T = CsType>
inline typename std::enable_if<CsType::ROW_MAJOR, T>::type simple_block_build(
    const CsType &A, const typename CsType::size_type start,
    const typename CsType::size_type end) {
  using size_type                 = typename CsType::size_type;
  constexpr static bool ONE_BASED = CsType::ONE_BASED;

  psmilu_error_if(end < start, "invalid block region [%zd,%zd)", start, end);
  const size_type n = A.nrows();
  psmilu_error_if(start > n || end > n,
                  "region exceeds the global system size %zd", n);

  CsType B;
  if (start == end) return B;
  if (start == 0u && end == n) return CsType(A, true);
  B.resize(end - start, end - start);
  internal::extract_block<ONE_BASED>(A.row_start(), A.col_ind(), A.vals(),
                                     start, end, B.row_start(), B.col_ind(),
                                     B.vals());
  return B;
}

template <class CsType, class T = CsType>
inline typename std::enable_if<!CsType::ROW_MAJOR, T>::type simple_block_build(
    const CsType &A, const typename CsType::size_type start,
    const typename CsType::size_type end) {
  using size_type                 = typename CsType::size_type;
  constexpr static bool ONE_BASED = CsType::ONE_BASED;

  psmilu_error_if(end < start, "invalid block region [%zd,%zd)", start, end);
  const size_type n = A.nrows();
  psmilu_error_if(start > n || end > n,
                  "region exceeds the global system size %zd", n);

  CsType B;
  if (start == end) return B;
  if (start == 0u && end == n) return CsType(A, true);
  B.resize(end - start, end - start);
  internal::extract_block<ONE_BASED>(A.col_start(), A.row_ind(), A.vals(),
                                     start, end, B.col_start(), B.row_ind(),
                                     B.vals());
  return B;
}
}  // namespace bjacobi
}  // namespace psmilu

#endif  // _PSMILU_BLOCKJACOBI_BLOCKBUILDER_HPP