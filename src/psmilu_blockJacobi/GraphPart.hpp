//@HEADER
//----------------------------------------------------------------------------
//                Copyright (C) 2019 The PSMILU AUTHORS
//----------------------------------------------------------------------------
//@HEADER

/// \file psmilu_blockJacobi/GraphPart.hpp
/// \brief METIS graph partition for building blocks
/// \authors Qiao,

#ifndef _PSMILU_BLOCKJACOBI_GRAPHPART_HPP
#define _PSMILU_BLOCKJACOBI_GRAPHPART_HPP

#include <algorithm>
#include <limits>
#include <new>
#include <type_traits>

#include <metis.h>

#include "psmilu_PermMatrix.hpp"
#include "psmilu_utils.hpp"

namespace psmilu {
namespace bjacobi {
namespace internal {
template <bool OneBased, class IndexArray>
inline std::size_t call_metis(const IndexArray &indptr,
                              const IndexArray &indices, const int nparts,
                              IndexArray &parts, IndexArray &P) {
  using index_type                     = typename IndexArray::value_type;
  using metis_index_array              = Array<idx_t>;
  using size_type                      = typename IndexArray::size_type;
  constexpr static size_type INDEX_MAX = std::numeric_limits<index_type>::max();
  constexpr static bool      SAFE_CAST = sizeof(idx_t) == sizeof(index_type);

  psmilu_error_if(indptr.size() - 1 > INDEX_MAX, "size overflow for METIS");

  if (!std::is_same<index_type, idx_t>::value) {
    psmilu_warning("METIS index type and input integer type mismatch");
    if (sizeof(index_type) >= sizeof(idx_t))
      psmilu_error_if(
          static_cast<size_type>(
              *std::max_element(indices.cbegin(), indices.cend()) > INDEX_MAX),
          "index overflow for using METIS!");
  }

  // graph data
  idx_t *xadj(nullptr), *adjncy(nullptr);
  if (SAFE_CAST) {
    xadj   = (idx_t *)indptr.data();
    adjncy = (idx_t *)indices.data();
  } else {
    xadj = new (std::nothrow) idx_t[indptr.size()];
    psmilu_error_if(!xadj, "memory allocation failed");
    adjncy = new (std::nothrow) idx_t[indices.size()];
    psmilu_error_if(!adjncy, "memory allocation failed");
    std::copy(indptr.cbegin(), indptr.cend(), xadj);
    std::copy(indices.cbegin(), indices.cend(), adjncy);
  }

  // METIS preparation

  idx_t nvtxs(indptr.size() - 1), ncon(1), opts[METIS_NOPTIONS], ncuts;
  METIS_SetDefaultOptions(opts);
  opts[METIS_OPTION_NUMBERING] = OneBased;  // set C or Fortran index
  opts[METIS_OPTION_OBJTYPE]   = METIS_OBJTYPE_CUT;
  // TODO enable debugging in METIS

  metis_index_array part(nvtxs);
  psmilu_error_if(part.status() == DATA_UNDEF, "memory allocation failed");
  P.resize(nvtxs);
  psmilu_error_if(P.status() == DATA_UNDEF, "memory allocation failed");
  parts.resize(nparts + 1);
  psmilu_error_if(parts.status() == DATA_UNDEF, "memory allocation failed");

  std::fill(parts.begin(), parts.end(), index_type(0));

  // call metis k-way partition with minimize edge cuts
  const int ret = METIS_PartGraphKway(&nvtxs, &ncon, xadj, adjncy, NULL, NULL,
                                      NULL, (idx_t *)&nparts, NULL, NULL, opts,
                                      &ncuts, part.data());
  if (!SAFE_CAST) {
    delete[] xadj;
    delete[] adjncy;
  }

  if (ret != METIS_OK) {
    psmilu_error_if(ret == METIS_ERROR_INPUT,
                    "METIS input error! Please report!");
    psmilu_error_if(ret == METIS_ERROR_MEMORY,
                    "METIS memory allocation failed");
    psmilu_error_if(ret == METIS_ERROR, "METIS error");
  }

  if (OneBased) std::for_each(part.begin(), part.end(), [](idx_t &i) { --i; });

  // build partitions
  const size_type n(nvtxs);
  for (size_type i(0); i < n; ++i) {
    psmilu_assert(part[i] < nparts, "partition index error");
    ++parts[part[i] + 1];
  }

  // accumulate the parts start array
  for (int i = 0; i < nparts; ++i) parts[i + 1] += parts[i];
  psmilu_assert((size_type)parts.back() == n, "fatal");

  // build permutation
  for (size_type ori_idx(0); ori_idx < n; ++ori_idx)
    P[parts[part[ori_idx]]++] = ori_idx;

  // revert the parts array
  index_type tmp(0);
  for (int i = 0; i < nparts; ++i) std::swap(parts[i], tmp);

  return ncuts;
}

template <bool OneBased, class IndexArray, class ValueArray>
inline void extract_perm_block(const IndexArray &i_indptr,
                               const IndexArray &i_indices,
                               const ValueArray &i_vals, const IndexArray &P,
                               const IndexArray &                   PT,
                               const typename IndexArray::size_type start,
                               const typename IndexArray::size_type end,
                               IndexArray &indptr, IndexArray &indices,
                               ValueArray &vals, ValueArray &buf) {
  using size_type  = typename IndexArray::size_type;
  using index_type = typename IndexArray::value_type;

  const size_type n = end - start;
  if (!n) {
    indptr.resize(1);
    indptr[0] = indptr[1] = OneBased;
    return;
  }
  indptr.resize(n + 1);
  psmilu_error_if(indptr.status() == DATA_UNDEF, "memory allocation failed");
  indptr.front() = OneBased;
  buf.resize(n);
  psmilu_error_if(buf.status() == DATA_UNDEF, "memory allocation failed");

  // create a handy lambda
  const auto c_idx = [](const size_type i) {
    return to_c_idx<size_type, OneBased>(i);
  };
  const auto in_range = [=](const size_type i) {
    return i < end && i >= start;
  };

  // determine total number of nonzeros
  for (size_type i(start); i < end; ++i) {
    const size_type last = i_indptr[P[i] + 1] - OneBased;
    size_type       nnz(0);
    for (size_type j = i_indptr[P[i]] - OneBased; j < last; ++j)
      if (in_range(PT[c_idx(i_indices[j])])) ++nnz;
    indptr[i - start + 1] = indptr[i - start] + nnz;
  }

  const size_type nnz = indptr[n] - OneBased;
  if (!nnz) return;

  indices.resize(nnz);
  psmilu_error_if(indices.status() == DATA_UNDEF, "memory allocation failed");
  vals.resize(nnz);
  psmilu_error_if(vals.status() == DATA_UNDEF, "memory allocation failed");

  auto i_itr = indices.begin();
  auto v_itr = vals.begin();

  // assemble the system

  for (size_type i(start); i < end; ++i) {
    const size_type last    = i_indptr[P[i] + 1] - OneBased;
    auto            itr_bak = i_itr;
    for (size_type j = i_indptr[P[i]] - OneBased; j < last; ++j) {
      const auto idx = PT[c_idx(i_indices[j])];
      if (in_range(idx)) {
        const auto tmp = idx - start;
        *i_itr++       = tmp;
        buf[tmp]       = i_vals[j];
      }
    }
    psmilu_assert(indptr[i - start + 1] - indptr[i - start] == i_itr - itr_bak,
                  "%zd row local size issue", size_type(i - start));
    std::sort(itr_bak, i_itr);
    for (auto itr = itr_bak; itr != i_itr; ++itr, ++v_itr) *v_itr = buf[*itr];
  }

  psmilu_assert(i_itr == indices.end(), "fatal");
  psmilu_assert(v_itr == vals.end(), "fatal");

  if (OneBased)
    std::for_each(indices.begin(), indices.end(), [](index_type &i) { ++i; });
}
}  // namespace internal

/// \class GraphPart
/// \tparam IndexType integer type, e.g. \a int
/// \brief Partition blocks and permutation based on graph algorithm
///
/// The goal is to build block-dominaint blocks for MILU, i.e.
/// $\boldsymbol{PAP}^T=\hat{\boldsymbol{A}}$, which is block-diagonal
/// dominaint.
template <class IndexType>
class GraphPart {
 public:
  typedef IndexType                       index_type;   ///< index type
  typedef Array<index_type>               index_array;  ///< index array
  typedef typename index_array::size_type size_type;    ///< size type

  GraphPart() = default;

  template <class CsType, typename T = size_type>
  inline typename std::enable_if<CsType::ROW_MAJOR, T>::type create_parts(
      const CsType &A, const int nparts) {
    psmilu_error_if(nparts < 1, "minimum partition requirement is 1");
    const auto &    indptr = A.row_start(), &indices = A.col_ind();
    const size_type ncuts = internal::call_metis<CsType::ONE_BASED>(
        indptr, indices, nparts, _part_start, _P());
    _P.inv().resize(A.nrows());
    psmilu_error_if(_P.inv().status() == DATA_UNDEF,
                    "memory allocation failed");
    _P.build_inv();
    return ncuts;
  }

  template <class CsType, typename T = size_type>
  inline typename std::enable_if<!CsType::ROW_MAJOR, T>::type create_parts(
      const CsType &A, const int nparts) {
    psmilu_error_if(nparts < 1, "minimum partition requirement is 1");
    const auto &    indptr = A.col_start(), &indices = A.row_ind();
    const size_type ncuts = internal::call_metis<CsType::ONE_BASED>(
        indptr, indices, nparts, _part_start, _P());
    _P.inv().resize(A.nrows());
    psmilu_error_if(_P.inv().status() == DATA_UNDEF,
                    "memory allocation failed");
    _P.build_inv();
    return ncuts;
  }

  /// \brief check number of partitions
  inline size_type nparts() const {
    return _part_start.empty() ? 0u : _part_start.size() - 1;
  }

  /// \brief check emptyness
  inline bool empty() const { return nparts() == 0u; }

  /// \brief check system size
  inline size_type size() const { return _P.size(); }

  template <class CsType, class T = CsType>
  inline typename std::enable_if<CsType::ROW_MAJOR, T>::type extract_block(
      const CsType &A, const size_type block,
      Array<typename CsType::value_type> &buf) const {
    psmilu_error_if(empty(), "empty graph partition");
    psmilu_error_if(block >= nparts(), "%zd exceeds partition size %zd", block,
                    nparts());
    psmilu_error_if(A.nrows() != size(), "global size mismatches");
    CsType          B;
    const size_type n = _part_start[block + 1] - _part_start[block];
    if (!n) return B;
    B.resize(n, n);
    const auto &indptr = A.row_start(), &indices = A.col_ind();
    internal::extract_perm_block<CsType::ONE_BASED>(
        indptr, indices, A.vals(), _P(), _P.inv(), _part_start[block],
        _part_start[block + 1], B.row_start(), B.col_ind(), B.vals(), buf);
    return B;
  }

  template <class CsType, class T = CsType>
  inline typename std::enable_if<!CsType::ROW_MAJOR, T>::type extract_block(
      const CsType &A, const size_type block,
      Array<typename CsType::value_type> &buf) const {
    psmilu_error_if(empty(), "empty graph partition");
    psmilu_error_if(block >= nparts(), "%zd exceeds partition size %zd", block,
                    nparts());
    psmilu_error_if(A.nrows() != size(), "global size mismatches");
    CsType          B;
    const size_type n = _part_start[block + 1] - _part_start[block];
    if (!n) return B;
    B.resize(n, n);
    const auto &indptr = A.col_start(), &indices = A.row_ind();
    internal::extract_perm_block<CsType::ONE_BASED>(
        indptr, indices, A.vals(), _P(), _P.inv(), _part_start[block],
        _part_start[block + 1], B.col_start(), B.row_ind(), B.vals(), buf);
    return B;
  }

  /// \brief get permutation vector
  inline const BiPermMatrix<index_type> &P() const { return _P; }

  /// \brief get the partition start array
  inline const index_array &part_start() const { return _part_start; }

 protected:
  index_array              _part_start;  ///< partition start array
  BiPermMatrix<index_type> _P;           ///< bi-directional permutation vector
};
}  // namespace bjacobi
}  // namespace psmilu

#endif  // _PSMILU_BLOCKJACOBI_GRAPHPART_HPP