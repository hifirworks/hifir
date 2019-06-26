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

#include <cstdint>

#include "psmilu_fc_mangling.h"

#ifndef DOXYGEN_SHOULD_SKIP_THIS

extern "C" {
void PSMILU_FC(genrcm, GENRCM)(int *, int *, int *, int *, int *, int *);
void PSMILU_FC(genrcm_l, GENRCM_L)(std::int64_t *, std::int64_t *,
                                   std::int64_t *, std::int64_t *,
                                   std::int64_t *, std::int64_t *);
}

namespace psmilu {
inline void rcm(int n, int *xadj, int *adjncy, int *perm, int *mask, int *xls) {
  PSMILU_FC(genrcm, GENRCM)(&n, xadj, adjncy, perm, mask, xls);
}

inline void rcm(std::int64_t n, std::int64_t *xadj, std::int64_t *adjncy,
                std::int64_t *perm, std::int64_t *mask, std::int64_t *xls) {
  PSMILU_FC(genrcm_l, GENRCM_L)(&n, xadj, adjncy, perm, mask, xls);
}
}  // namespace psmilu

#endif  // DOXYGEN_SHOULD_SKIP_THIS

#include "psmilu_Options.h"
#include "psmilu_log.hpp"

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

  if (psmilu_verbose(PRE, opt)) psmilu_info("begin running RCM reordering...");
  const auto   n      = B.nrows();
  iarray_type &xadj   = const_cast<iarray_type &>(B.col_start());
  iarray_type &adjncy = const_cast<iarray_type &>(B.row_ind());
  if (!CcsType::ONE_BASED) {
    for (auto &v : xadj) ++v;
    for (auto &v : adjncy) ++v;
  }
  iarray_type P(n), mask(n), xls(n + 1);
  if (sizeof(index_type) == sizeof(int))
    rcm(n, (int *)xadj.data(), (int *)adjncy.data(), (int *)P.data(),
        (int *)mask.data(), (int *)xls.data());
  else
    rcm(n, (std::int64_t *)xadj.data(), (std::int64_t *)adjncy.data(),
        (std::int64_t *)P.data(), (std::int64_t *)mask.data(),
        (std::int64_t *)xls.data());
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