//@HEADER
//----------------------------------------------------------------------------
//                Copyright (C) 2019 The PSMILU AUTHORS
//----------------------------------------------------------------------------
//@HEADER

/// \file psmilu_RCM/driver.hpp
/// \brief Interface for calling Boost Graph RCM
/// \authors Qiao,

#ifndef _PSMILU_RCM_DRIVER_HPP
#define _PSMILU_RCM_DRIVER_HPP

#define BOOST_NO_HASH

#include <memory>
#include <new>
#include <type_traits>
#include <utility>
#include <vector>

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/compressed_sparse_row_graph.hpp>
#include <boost/graph/cuthill_mckee_ordering.hpp>
#include <boost/graph/properties.hpp>

#include "psmilu_Array.hpp"
#include "psmilu_Options.h"
#include "psmilu_log.hpp"

namespace psmilu {
/// \brief use RCM algorithm to reduce the bandwidth
/// \tparam IsSymm if \a true, use symmetric version (undirected graph)
/// \tparam CcsType_C c index \ref CCS matrix
/// \param[in] B input matrix with only *sparsity pattern*
/// \param[in] opt options
/// \return permutation vector
/// \ingroup pre
/// \note This requires Boost Graph Library (BGL)
template <bool IsSymm, class CcsType_C>
inline Array<typename CcsType_C::index_type> run_rcm(const CcsType_C &B,
                                                     const Options &  opt) {
  using size_type = typename CcsType_C::size_type;
  static_assert(!CcsType_C::ONE_BASED, "must be C index");
  using index_type = typename CcsType_C::index_type;

  using vertex_property = boost::property<boost::vertex_index_t, index_type>;
  using edge_property   = boost::property<boost::edge_index_t, index_type>;

  using symm_graph_type =
      boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS,
                            vertex_property, edge_property>;
  using asymm_graph_type =
      boost::adjacency_list<boost::vecS, boost::vecS, boost::directedS,
                            vertex_property, edge_property>;

  // NOTE that since this is called for symmetric, ccs is crs
  using graph_type = typename std::conditional<IsSymm, symm_graph_type,
                                               asymm_graph_type>::type;

  if (psmilu_verbose(PRE, opt)) psmilu_info("begin running RCM reordering...");

  const size_type nv = B.ncols();  // # of vertices
  size_type       ne = B.nnz();    // # of edges upper bound

  std::shared_ptr<graph_type> G;

  // do intermidiate inside scope to ensure memory free before constructing
  // the graph
  do {
    using edge_type = std::pair<index_type, index_type>;
    Array<edge_type> edges(ne);
    psmilu_error_if(edges.status() == DATA_UNDEF, "memory allocation failed");
    size_type i(0);
    for (size_type col(0); col < nv; ++col)
      for (auto itr = B.row_ind_cbegin(col); itr != B.row_ind_cend(col);
           ++itr) {
        const bool add_edge = IsSymm ? *itr > col : *itr != col;
        if (add_edge)
          edges[i++] = std::make_pair(static_cast<index_type>(col), *itr);
      }
    ne = i;
    edges.resize(ne);

    // build graph
    G.reset(new (std::nothrow)
                graph_type(edges.cbegin(), edges.cend(), nv, ne));
    psmilu_error_if(!G, "memory allocation failed for boost graph");
  } while (false);

  // then, call RCM
  Array<index_type> P(nv);
  psmilu_error_if(P.status() == DATA_UNDEF, "memory allocation failed");
  const graph_type &graph = *G;
  do {
    // get property map
    const typename boost::property_map<graph_type, boost::vertex_index_t>::type
                            index_map = boost::get(boost::vertex_index, graph);
    std::vector<index_type> inv_perm(nv);
    // call RCM
    boost::cuthill_mckee_ordering(graph, inv_perm.rbegin());
    for (size_type i(0); i < nv; ++i) P[i] = index_map[inv_perm[i]];
  } while (false);
  if (psmilu_verbose(PRE, opt)) psmilu_info("finish RCM reordering...");
  return P;
}
}  // namespace psmilu

#endif  // _PSMILU_RCM_DRIVER_HPP