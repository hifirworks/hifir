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

#define BOOST_NO_HASH

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/properties.hpp>

#include "psmilu_Array.hpp"
#include "psmilu_log.hpp"

namespace psmilu {

namespace internal {

/// \brief create an undirected graph
/// \tparam IndexType index type, e.g. \a int
/// \ingroup pre
///
/// Notice that we use undirected graph, because for symmetric systems, we
/// only store the strict lower part, whereas the whole matrix (except for
/// diagonals) for general systems, which is equiv to A+A'.
template <class IndexType>
struct BGL_UndirectedGraphTrait {
  using index_type      = IndexType;  ///< index type
  using vertex_property = boost::property<
      boost::vertex_index_t, index_type,
      boost::property<
          boost::vertex_degree_t, index_type,
          boost::property<boost::vertex_color_t, boost::default_color_type,
                          boost::property<boost::vertex_priority_t, float>>>>;
  ///< vertex property
  using edge_property = boost::property<
      boost::edge_index_t, index_type,
      boost::property<boost::edge_color_t, boost::default_color_type>>;
  ///< edge property
  using graph_type =
      boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS,
                            vertex_property, edge_property, boost::no_property,
                            boost::vecS>;
  ///< graph type
};

}  // namespace internal

/// \brief create graph
/// \tparam IsSymm if \a true, then assume symmetric input
/// \tparam CcsType input matrix storage \ref CCS
/// \param[in] B input matrix
/// \return graph representation of matrix \a B
template <bool IsSymm, class CcsType>
inline typename internal::BGL_UndirectedGraphTrait<
    typename CcsType::index_type>::graph_type
create_graph(const CcsType &B) {
  using index_type = typename CcsType::index_type;
  using graph_type =
      typename internal::BGL_UndirectedGraphTrait<index_type>::graph_type;
  using size_type = typename CcsType::size_type;
  using edge_type = std::pair<index_type, index_type>;

  const size_type nv = B.ncols();  // number of vertices
  size_type       ne = B.nnz();    // upper bound for number of edges

  Array<edge_type> edges(ne);
  psmilu_error_if(edges.status() == DATA_UNDEF, "memory allocation failed");
  size_type i(0);
  for (size_type col(0); col < nv; ++col) {
    if (IsSymm) {
      if (B.nnz_in_col(col)) {
        auto itr = B.row_ind_cbegin(col);
        if (*itr - CcsType::ONE_BASED == (index_type)col) ++itr;
        for (auto j = itr; j != B.row_ind_cend(col); ++j)
          edges[i++] = std::make_pair(static_cast<index_type>(col),
                                      *j - CcsType::ONE_BASED);
      }
    } else {
      for (auto itr = B.row_ind_cbegin(col); itr != B.row_ind_cend(col); ++itr)
        if (*itr - CcsType::ONE_BASED != (index_type)col)
          edges[i++] = std::make_pair(static_cast<index_type>(col),
                                      *itr - CcsType::ONE_BASED);
    }
  }
  ne = i;
  edges.resize(ne);
  return graph_type(edges.cbegin(), edges.cend(), nv, ne);
}
}  // namespace psmilu

#endif  // _PSMILU_BGL_GRAPH_HPP
