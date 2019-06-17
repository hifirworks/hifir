//@HEADER
//----------------------------------------------------------------------------
//                Copyright (C) 2019 The PSMILU AUTHORS
//----------------------------------------------------------------------------
//@HEADER

/// \file psmilu_BGL/king.hpp
/// \brief Interface for calling BGL's King ordering
/// \authors Qiao,

#ifndef _PSMILU_BGL_KING_HPP
#define _PSMILU_BGL_KING_HPP

#include <vector>

#include "graph.hpp"

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/king_ordering.hpp>
#include <boost/graph/properties.hpp>

#include "psmilu_Options.h"

namespace psmilu {
/// \brief use King algorithm to reduce the bandwidth
/// \tparam IsSymm if \a true, use symmetric version (undirected graph)
/// \tparam CcsType_C c index \ref CCS matrix
/// \param[in] B input matrix with only *sparsity pattern*
/// \param[in] opt options
/// \return permutation vector
/// \ingroup pre
/// \note This requires Boost Graph Library (BGL)
template <bool IsSymm, class CcsType_C>
inline Array<typename CcsType_C::index_type> run_king(const CcsType_C &B,
                                                      const Options &  opt) {
  using size_type  = typename CcsType_C::size_type;
  using index_type = typename CcsType_C::index_type;
  using vertex_property =
      boost::property<boost::vertex_index_t, index_type,
                      boost::property<boost::vertex_degree_t, index_type>>;
  ///< vertex property
  using edge_property = boost::property<
      boost::edge_index_t, index_type,
      boost::property<boost::edge_color_t, boost::default_color_type>>;
  ///< edge property
  using graph_type =
      boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS,
                            vertex_property, edge_property, boost::no_property>;

  if (psmilu_verbose(PRE, opt)) psmilu_info("begin running King reordering...");

  const size_type nv = B.ncols();
  graph_type      graph;
  do {
    const auto edges = create_graph_edges<IsSymm>(B);
    graph = graph_type(edges.cbegin(), edges.cend(), nv, edges.size());
  } while (false);

  Array<index_type> P(nv);
  psmilu_error_if(P.status() == DATA_UNDEF, "memory allocation failed");
  // get property map
  std::vector<index_type> buf(P.size());
  // call King
  boost::king_ordering(graph, buf.rbegin());
  for (size_type i(0); i < nv; ++i) P[buf[i]] = i;
  if (psmilu_verbose(PRE, opt)) psmilu_info("finish King reordering...");
  return P;
}

}  // namespace psmilu

#endif  // _PSMILU_BGL_RCM_HPP
