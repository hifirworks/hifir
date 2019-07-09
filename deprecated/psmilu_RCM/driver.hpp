//@HEADER
//----------------------------------------------------------------------------
//                Copyright (C) 2019 The PSMILU AUTHORS
//----------------------------------------------------------------------------
//@HEADER

/// \file psmilu_RCM/driver.hpp
/// \brief RCM wrapper around RCM implementation with Alan George
/// \authors Qiao,

// Reference:
// Computer Solution of Sparse Linear Systems (chapter 4.4)

#ifndef _PSMILU_RCM_DRIVER_HPP
#define _PSMILU_RCM_DRIVER_HPP

#include "psmilu_Options.h"
#include "psmilu_log.hpp"

#include "psmilu_RCM/rcm.hpp"

namespace psmilu {
/// \brief use RCM algorithm to reduce the bandwidth
/// \tparam CcsType ccs \ref CCS matrix
/// \param[in] B input matrix with only *sparsity pattern*
/// \param[in] opt options
/// \return permutation vector
/// \ingroup pre
template <class CcsType>
inline typename CcsType::iarray_type run_rcm(const CcsType &B,
                                             const Options &opt,
                                             const bool ensure_index = false) {
  using index_type  = typename CcsType::index_type;
  using iarray_type = typename CcsType::iarray_type;
  using rcm_type    = rcm::RCM<index_type>;

  if (psmilu_verbose(PRE, opt)) psmilu_info("begin running RCM reordering...");
  const auto   n      = B.nrows();
  iarray_type &xadj   = const_cast<iarray_type &>(B.col_start());
  iarray_type &adjncy = const_cast<iarray_type &>(B.row_ind());
  if (!CcsType::ONE_BASED) {
    for (auto &v : xadj) ++v;
    for (auto &v : adjncy) ++v;
  }
  iarray_type P(n);
  rcm_type().apply(n, xadj.data(), adjncy.data(), P.data());
  if (!CcsType::ONE_BASED)
    for (auto &v : P) --v;
  if (psmilu_verbose(PRE, opt)) psmilu_info("finish RCM reordering...");
  if (!CcsType::ONE_BASED && ensure_index) {
    for (auto &v : xadj) --v;
    for (auto &v : adjncy) --v;
  }
  return P;
}

}  // namespace psmilu

#endif  // _PSMILU_RCM_DRIVER_HPP