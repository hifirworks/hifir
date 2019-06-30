//@HEADER
//----------------------------------------------------------------------------
//                Copyright (C) 2019 The PSMILU AUTHORS
//----------------------------------------------------------------------------
//@HEADER

/// \file psmilu_BGL/rcm.hpp
/// \brief Interface for calling BGL's reverse Cuthill Mckee (RCM)
/// \authors Qiao,
/// \deprecated got better code

#ifndef _PSMILU_BGL_RCM_HPP
#define _PSMILU_BGL_RCM_HPP

#include <vector>

#include "graph.hpp"

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/cuthill_mckee_ordering.hpp>
#include <boost/graph/properties.hpp>

#include "psmilu_Options.h"

namespace psmilu {
/// \brief use RCM algorithm to reduce the bandwidth
/// \tparam IsSymm if \a true, use symmetric version (undirected graph)
/// \tparam CcsType ccs \ref CCS matrix
/// \param[in] B input matrix with only *sparsity pattern*
/// \param[in] opt options
/// \return permutation vector
/// \ingroup pre
/// \note This requires Boost Graph Library (BGL)
template <bool IsSymm, class CcsType>
inline Array<typename CcsType::index_type> run_rcm(const CcsType &B,
                                                   const Options &opt) {
  using size_type  = typename CcsType::size_type;
  using index_type = typename CcsType::index_type;
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
  using iarray_type = Array<index_type>;

  if (psmilu_verbose(PRE, opt)) psmilu_info("begin running RCM reordering...");
  const size_type nv = B.ncols();
  graph_type      graph;
  do {
    const auto edges = create_graph_edges<IsSymm>(B);
    graph = graph_type(edges.cbegin(), edges.cend(), nv, edges.size());
  } while (false);

  iarray_type P(nv);
  psmilu_error_if(P.status() == DATA_UNDEF, "memory allocation failed");

  // get property map
  // using prop_map =
  //     typename boost::property_map<graph_type, boost::vertex_index_t>::type;
  // const prop_map index_map = boost::get(boost::vertex_index, graph);

  // call RCM
  std::vector<index_type> buf(P.size());
  boost::cuthill_mckee_ordering(graph, buf.rbegin());
  for (size_type i(0); i < nv; ++i) P[buf[i]] = i;
  if (psmilu_verbose(PRE, opt)) psmilu_info("finish RCM reordering...");
  return P;
}

}  // namespace psmilu

#endif  // _PSMILU_BGL_RCM_HPP
