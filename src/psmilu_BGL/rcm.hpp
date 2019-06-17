//@HEADER
//----------------------------------------------------------------------------
//                Copyright (C) 2019 The PSMILU AUTHORS
//----------------------------------------------------------------------------
//@HEADER

/// \file psmilu_BGL/rcm.hpp
/// \brief Interface for calling BGL's reverse Cuthill Mckee (RCM)
/// \authors Qiao,

#ifndef _PSMILU_BGL_RCM_HPP
#define _PSMILU_BGL_RCM_HPP

#include <vector>

#include "graph.hpp"

#include <boost/graph/compressed_sparse_row_graph.hpp>
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
  using graph_type =
      boost::compressed_sparse_row_graph<boost::directedS, boost::no_property,
                                         boost::no_property, boost::no_property,
                                         index_type, index_type>;
  using iarray_type = Array<index_type>;

  if (psmilu_verbose(PRE, opt)) psmilu_info("begin running RCM reordering...");
  const size_type nv = B.ncols();
  graph_type      graph;
  do {
    const auto edges = create_csr_edges<IsSymm>(B);
    graph = IsSymm ? graph_type(boost::edges_are_unsorted, edges.cbegin(),
                                edges.cend(), (index_type)nv)
                   : graph_type(boost::edges_are_sorted, edges.cbegin(),
                                edges.cend(), (index_type)nv);
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
