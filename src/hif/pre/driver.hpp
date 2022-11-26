///////////////////////////////////////////////////////////////////////////////
//                  This file is part of the HIFIR library                   //
///////////////////////////////////////////////////////////////////////////////

/*!
 * \file hif/pre/driver.hpp
 * \brief Top-level driver routine to handle preprocessing
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

#ifndef _HIF_PRE_DRIVER_HPP
#define _HIF_PRE_DRIVER_HPP

#include <cmath>
#include <sstream>
#include <string>

#include "hif/Options.h"
#include "hif/ds/Array.hpp"
#include "hif/ds/CompressedStorage.hpp"
#include "hif/utils/Timer.hpp"

#include "hif/pre/matching_scaling.hpp"
#include "hif/pre/reordering.hpp"

namespace hif {

/// \brief routine to perform preprocessing for improving the quality of input
/// \tparam IsSymm if \a true, the assume the leading block is symmetric
/// \tparam CcsType input matrix type, see \ref CCS
/// \tparam ScalingArray array type for scaling vectors, see \ref Array
/// \tparam PermType permutation type, see \ref BiPermMatrix
/// \param[in] A input matrix
/// \param[in] A_crs CRS format of input \a A
/// \param[in] m0 leading block size
/// \param[in] level current factorization level
/// \param[in] opt control parameters
/// \param[out] s row scaling vector
/// \param[out] t column scaling vector
/// \param[out] p row permutation
/// \param[out] q column permutation
/// \return The actual leading block size, no larger than \a m0
/// \ingroup pre
///
/// Notice that, in general, the preprocessing involves two steps: 1) perform
/// matching/scaling to improve the diagonal dominance and conditioning, and 2)
/// perform reordering to improve the sparsity of LU decomposition.
template <bool IsSymm, class CcsType, class CrsType, class ScalingArray,
          class PermType>
inline typename CcsType::size_type do_preprocessing(
    const CcsType &A, const CrsType &A_crs,
    const typename CcsType::size_type m0,
    const typename CcsType::size_type level, const Options &opt,
    ScalingArray &s, ScalingArray &t, PermType &p, PermType &q) {
  static_assert(!CcsType::ROW_MAJOR, "must be CCS");
  using indptr_type = typename CcsType::indptr_type;
  using size_type   = typename CcsType::size_type;

  if (opt.reorder < 0 || opt.reorder >= REORDER_NULL)
    hif_error("Invalid reorder flag %d", opt.reorder);

  DefaultTimer timer;

  if (hif_verbose(PRE, opt)) hif_info("Performing matching step");

  timer.start();

  auto match_res = do_maching<IsSymm>(A, A_crs, m0, level, opt, s, t, p, q);

  timer.finish();
  if (hif_verbose(PRE_TIME, opt))
    hif_info("Matching took %gs.", (double)timer.time());

  const size_type m = match_res.second;
  if (opt.reorder != REORDER_OFF && m) {
    if (hif_verbose(PRE, opt)) hif_info("Performing reordering step");
    timer.start();

    std::string reorder_name = "AMD";

    auto &             B = match_res.first;
    Array<indptr_type> P;
    if (opt.reorder == REORDER_AUTO) {
      // for auto reordering, we use rcm only if first level symmetry and
      // have static deferrals
      const bool try_use_rcm = IsSymm && level == 1u && B.nrows() != m0;
      P = try_use_rcm ? run_rcm(B, opt) : run_amd<false>(B, opt);
      if (try_use_rcm) reorder_name = "RCM";
    } else if (opt.reorder == REORDER_AMD) {
      P = run_amd<false>(B, opt);
    } else {
      P            = run_rcm(B, opt);
      reorder_name = "RCM";
    }

    timer.finish();

    if (hif_verbose(PRE_TIME, opt))
      hif_info("Reordering %s took %gs.", reorder_name.c_str(),
               (double)timer.time());

    // now let's reorder the permutation arrays
    // we use the inverse mapping as buffer
    const auto reorder_finalize_perm = [&P, m](PermType &Q) {
      auto &          forward = Q(), &buf = Q.inv();
      const size_type N = forward.size();
      size_type       i(0);
      for (; i < m; ++i) buf[i] = forward[P[i]];
      for (; i < N; ++i) buf[i] = forward[i];
      forward.swap(buf);
      Q.build_inv();
    };

    reorder_finalize_perm(p);
    reorder_finalize_perm(q);
  } else {
    if (hif_verbose(PRE, opt))
      m ? hif_info("Reordering skipped")
        : hif_info("Reordering skipped due to empty leading block");
    p.build_inv();
    q.build_inv();
  }
  return m;
}

}  // namespace hif

#endif  // _HIF_PRE_DRIVER_HPP
