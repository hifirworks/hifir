///////////////////////////////////////////////////////////////////////////////
//                  This file is part of the HIFIR library                   //
///////////////////////////////////////////////////////////////////////////////

/*!
 * \file hif/pre/reordering.hpp
 * \brief calling interface for reordering
 * \author Qiao Chen

\verbatim
Copyright (C) 2021 NumGeom Group at Stony Brook University

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
\endverbatim

 */

#ifndef _HIF_PRE_REORDERING_HPP
#define _HIF_PRE_REORDERING_HPP

#include "hif/pre/amd.hpp"
#include "hif/pre/rcm.hpp"

#include "hif/Options.h"
#include "hif/ds/Array.hpp"
#include "hif/utils/log.hpp"

namespace hif {

/*!
 * \addtogroup pre
 * @{
 */

/// \brief apply approximated mimimum degree ordering
/// \tparam IsSymm if \a true, then assume symmetric pattern input
/// \tparam CcsType ccs sparse storage, see \ref CCS
/// \param[in] B matrix with pattern given
/// \param[in] opt options
/// \return symmetric permutation so that PAP' satisties AMD property
template <bool IsSymm, class CcsType>
inline Array<typename CcsType::index_type> run_amd(const CcsType &B,
                                                   const Options &opt) {
  using size_type  = typename CcsType::size_type;
  using index_type = typename CcsType::index_type;
  using amd        = amd::AMD<index_type>;

  const size_type m = B.nrows();
  hif_assert(B.nrows() == m, "the leading block size should be size(B)");

  // reordering
  double Control[HIF_AMD_CONTROL], Info[AMD_INFO];
  amd::defaults(Control);
  Control[HIF_AMD_SYMM_FLAG] = !IsSymm;

  if (hif_verbose(PRE, opt)) {
    hif_info("performing AMD reordering");
    std::stringstream s;
    amd::control(s, Control);
    hif_info(s.str().c_str());
  }
  Array<index_type> P(m);
  hif_error_if(P.status() == DATA_UNDEF, "memory allocation failed");
  const int result = amd::order(m, B.col_start().data(), B.row_ind().data(),
                                P.data(), Control, Info);
  if (result != AMD_OK && result != AMD_OK_BUT_JUMBLED) {
    // NOTE that we modified AMD to utilize jumbled return to automatically
    // compute the transpose
    std::stringstream s;
    amd::info(s, Info);
    const std::string msg =
        "AMD returned invalid flag " + std::to_string(result) +
        ", the following message was loaded from AMD info routine:\n" + s.str();
    hif_error(msg.c_str());
  }

  if (hif_verbose(PRE, opt)) {
    hif_info("AMD reordering done with information:\n");
    std::stringstream s;
    amd::info(s, Info);
    hif_info(s.str().c_str());
  }

  return P;
}

/// \brief apply reversed Cuthill-Mckee ordering
/// \tparam CcsType ccs sparse storage, see \ref CCS
/// \param[in] B matrix with pattern given
/// \param[in] opt options
/// \return symmetric permutation so that PAP' satisties RCM property
template <class CcsType>
inline typename CcsType::iarray_type run_rcm(const CcsType &B,
                                             const Options &opt) {
  using index_type  = typename CcsType::index_type;
  using iarray_type = typename CcsType::iarray_type;
  using rcm_type    = rcm::RCM<index_type>;

  if (hif_verbose(PRE, opt)) hif_info("begin running RCM reordering...");
  const auto   n      = B.nrows();
  iarray_type &xadj   = const_cast<iarray_type &>(B.col_start());
  iarray_type &adjncy = const_cast<iarray_type &>(B.row_ind());
  for (auto &v : xadj) ++v;
  for (auto &v : adjncy) ++v;
  iarray_type P(n);
  rcm_type().apply(n, xadj.data(), adjncy.data(), P.data());
  for (auto &v : P) --v;
  if (hif_verbose(PRE, opt)) hif_info("finish RCM reordering...");
  return P;
}

/*!
 * @}
 */

}  // namespace hif

#endif  // _HIF_PRE_REORDERING_HPP