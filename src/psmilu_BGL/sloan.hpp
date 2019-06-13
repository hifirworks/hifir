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

#include <vector>

#include "graph.hpp"

#include <boost/graph/sloan_ordering.hpp>

#include "psmilu_Options.h"

namespace psmilu {
/// \brief use Sloan reordering method
/// \tparam IsSymm if \a true, use symmetric version (undirected graph)
/// \tparam CcsType_C c index \ref CCS matrix
/// \param[in] B input matrix with only *sparsity pattern*
/// \param[in] opt options
/// \return permutation vector
/// \ingroup pre
/// \note This requires Boost Graph Library (BGL)
template <bool IsSymm, class CcsType_C>
inline Array<typename CcsType_C::index_type> run_sloan(const CcsType_C &B,
                                                       const Options &  opt) {
  using size_type = typename CcsType_C::size_type;
  static_assert(!CcsType_C::ONE_BASED, "must be C index");
  using index_type = typename CcsType_C::index_type;
  using graph_type =
      typename internal::BGL_UndirectedGraphTrait<index_type>::graph_type;

  if (psmilu_verbose(PRE, opt))
    psmilu_info("begin running Sloan reordering...");

  graph_type graph = create_graph<IsSymm>(B);
  const size_type  nv    = B.ncols();

  Array<index_type> P(nv);
  psmilu_error_if(P.status() == DATA_UNDEF, "memory allocation failed");
  do {
    // get property map
    const typename boost::property_map<graph_type, boost::vertex_index_t>::type
        index_map = boost::get(boost::vertex_index, graph);
    // call sloan
    boost::sloan_ordering(graph, P.begin(),
                          boost::get(boost::vertex_color, graph),
                          boost::make_degree_map(graph),
                          boost::get(boost::vertex_priority, graph));
    for (size_type i(0); i < nv; ++i) P[i] = index_map[P[i]];
  } while (false);
  if (psmilu_verbose(PRE, opt)) psmilu_info("finish RCM reordering...");
  return P;
}
}  // namespace psmilu

#endif  // _PSMILU_BGL_SLOAN_HPP
