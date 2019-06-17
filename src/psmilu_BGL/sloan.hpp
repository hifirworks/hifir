//@HEADER
//----------------------------------------------------------------------------
//                Copyright (C) 2019 The PSMILU AUTHORS
//----------------------------------------------------------------------------
//@HEADER

/// \file psmilu_BGL/sloan.hpp
/// \brief Interface for calling Boost Graph Sloan ordering
/// \authors Qiao,

#ifndef _PSMILU_BGL_SLOAN_HPP
#define _PSMILU_BGL_SLOAN_HPP

#include "graph.hpp"

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/properties.hpp>
#include <boost/graph/sloan_ordering.hpp>

#include "psmilu_Options.h"

namespace psmilu {
/// \brief use Sloan reordering method
/// \tparam IsSymm if \a true, use symmetric version (undirected graph)
/// \tparam CcsType ccs \ref CCS matrix
/// \param[in] B input matrix with only *sparsity pattern*
/// \param[in] opt options
/// \return permutation vector
/// \ingroup pre
/// \note This requires Boost Graph Library (BGL)
template <bool IsSymm, class CcsType>
inline Array<typename CcsType::index_type> run_sloan(const CcsType &B,
                                                     const Options &opt) {
  using size_type       = typename CcsType::size_type;
  using index_type      = typename CcsType::index_type;
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
      boost::adjacency_list<boost::setS, boost::vecS, boost::undirectedS,
                            vertex_property, edge_property, boost::no_property>;

  if (psmilu_verbose(PRE, opt))
    psmilu_info("begin running Sloan reordering...");

  const size_type nv = B.ncols();
  graph_type      graph;
  do {
    const auto edges = create_graph_edges<IsSymm>(B);
    graph = graph_type(edges.cbegin(), edges.cend(), nv, edges.size());
  } while (false);

  Array<index_type> P(nv);
  psmilu_error_if(P.status() == DATA_UNDEF, "memory allocation failed");
  do {
    // get property map
    // const typename boost::property_map<graph_type,
    // boost::vertex_index_t>::type
    //     index_map = boost::get(boost::vertex_index, graph);
    // call sloan
    Array<index_type> buf(P.size());
    boost::sloan_ordering(graph, buf.begin(),
                          boost::get(boost::vertex_color, graph),
                          boost::make_degree_map(graph),
                          boost::get(boost::vertex_priority, graph));
    for (size_type i(0); i < nv; ++i) P[buf[i]] = i;
  } while (false);
  if (psmilu_verbose(PRE, opt)) psmilu_info("finish Sloan reordering...");
  return P;
}
}  // namespace psmilu

#endif  // _PSMILU_BGL_SLOAN_HPP
