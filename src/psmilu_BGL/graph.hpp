//@HEADER
//----------------------------------------------------------------------------
//                Copyright (C) 2019 The PSMILU AUTHORS
//----------------------------------------------------------------------------
//@HEADER

/// \file psmilu_BGL/graph.hpp
/// \brief Interface for creating undirected graph
/// \authors Qiao,

#ifndef _PSMILU_BGL_GRAPH_HPP
#define _PSMILU_BGL_GRAPH_HPP

#include <utility>

#ifndef BOOST_NO_HASH
#  define BOOST_NO_HASH
#endif  // BOOST_NO_HASH

#include "psmilu_Array.hpp"
#include "psmilu_log.hpp"

namespace psmilu {

/*!
 * \addtogroup pre
 * @{
 */

template <class IndexType>
using bgl_edge_type = std::pair<IndexType, IndexType>;
template <class IndexType>
using bgl_edges_type = Array<bgl_edge_type<IndexType>>;

/// \brief create graph
/// \tparam IsSymm if \a true, then assume symmetric input
/// \tparam CcsType input matrix storage \ref CCS
/// \param[in] B input matrix
/// \return graph representation of edges
template <bool IsSymm, class CcsType>
inline bgl_edges_type<typename CcsType::index_type> create_graph_edges(
    const CcsType &B) {
  using index_type                = typename CcsType::index_type;
  using size_type                 = typename CcsType::size_type;
  constexpr static bool ONE_BASED = CcsType::ONE_BASED;

  const size_type nv = B.ncols();  // number of vertices
  size_type       ne = B.nnz();    // upper bound for number of edges

  bgl_edges_type<index_type> edges(ne);
  psmilu_error_if(edges.status() == DATA_UNDEF, "memory allocation failed");
  size_type i(0);
  for (size_type col(0); col < nv; ++col) {
    if (IsSymm) {
      if (B.nnz_in_col(col)) {
        auto itr = B.row_ind_cbegin(col);
        if (*itr - ONE_BASED == (index_type)col) ++itr;
        for (auto j = itr; j != B.row_ind_cend(col); ++j)
          edges[i++] =
              std::make_pair(static_cast<index_type>(col), *j - ONE_BASED);
      }
    } else {
      for (auto itr = B.row_ind_cbegin(col); itr != B.row_ind_cend(col); ++itr)
        if (*itr - ONE_BASED != (index_type)col)
          edges[i++] =
              std::make_pair(static_cast<index_type>(col), *itr - ONE_BASED);
    }
  }
  ne = i;
  edges.resize(ne);
  return edges;
  // return GraphType(edges.cbegin(), edges.cend(), nv, ne);
}

/// \brief create crs graph
/// \tparam IsSymm if \a true, then assume symmetric input
/// \param[in] B input matrix
/// \return graph representation of edges
template <bool IsSymm, class CcsType>
inline bgl_edges_type<typename CcsType::index_type> create_csr_edges(
    const CcsType &B) {
  using index_type                = typename CcsType::index_type;
  using size_type                 = typename CcsType::size_type;
  constexpr static bool ONE_BASED = CcsType::ONE_BASED;

  const size_type nv = B.ncols();
  if (!IsSymm) return create_graph_edges<false>(B);

  // count number of edges
  bgl_edges_type<index_type> edges(B.nnz() * 2);
  size_type                  i(0);
  for (size_type col(0); col < nv; ++col)
    if (B.nnz_in_col(col)) {
      auto itr = B.row_ind_cbegin(col);
      if (*itr - ONE_BASED == (index_type)col) ++itr;
      for (auto j = itr; j != B.row_ind_cend(col); ++j) {
        edges[i++] =
            std::make_pair(static_cast<index_type>(col), *j - ONE_BASED);
        edges[i++] =
            std::make_pair(*j - ONE_BASED, static_cast<index_type>(col));
      }
    }
  edges.resize(i);
  return edges;
}

/*!
 * @}
 */

}  // namespace psmilu

#endif  // _PSMILU_BGL_GRAPH_HPP
