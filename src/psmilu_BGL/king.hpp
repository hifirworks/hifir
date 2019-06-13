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

#include <iterator>
#include <vector>

#include "graph.hpp"

#include <boost/graph/king_ordering.hpp>

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
  using size_type = typename CcsType_C::size_type;
  static_assert(!CcsType_C::ONE_BASED, "must be C index");
  using index_type = typename CcsType_C::index_type;
  using graph_type =
      typename internal::BGL_UndirectedGraphTrait<index_type>::graph_type;

  if (psmilu_verbose(PRE, opt)) psmilu_info("begin running King reordering...");

  const graph_type graph = create_graph<IsSymm>(B);
  const size_type  nv    = B.ncols();

  Array<index_type> P(nv);
  psmilu_error_if(P.status() == DATA_UNDEF, "memory allocation failed");
  // get property map
  const typename boost::property_map<graph_type, boost::vertex_index_t>::type
                                                              index_map = boost::get(boost::vertex_index, graph);
  std::reverse_iterator<typename Array<index_type>::iterator> r_itr(P.end());
  // call King
  boost::king_ordering(graph, r_itr);
  for (size_type i(0); i < nv; ++i) P[i] = index_map[P[i]];
  if (psmilu_verbose(PRE, opt)) psmilu_info("finish King reordering...");
  return P;
}

}  // namespace psmilu

#endif  // _PSMILU_BGL_RCM_HPP
