///////////////////////////////////////////////////////////////////////////////
//                This file is part of HILUCSI project                       //
///////////////////////////////////////////////////////////////////////////////

/*!
 * \file hilucsi/pre/driver.hpp
 * \brief Top-level driver routine to handle preprocessing
 * \author Qiao Chen

\verbatim
Copyright (C) 2019 NumGeom Group at Stony Brook University

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
\endverbatim

 */

#ifndef _HILUCSI_PRE_DRIVER_HPP
#define _HILUCSI_PRE_DRIVER_HPP

#include <cmath>
#include <sstream>
#include <string>

#include "hilucsi/Options.h"
#include "hilucsi/ds/Array.hpp"
#include "hilucsi/ds/CompressedStorage.hpp"
#include "hilucsi/utils/Timer.hpp"

#include "hilucsi/pre/matching_scaling.hpp"
#include "hilucsi/pre/reordering.hpp"

namespace hilucsi {

/// \brief routine to perform preprocessing for improving the quality of input
/// \tparam IsSymm if \a true, the assume the leading block is symmetric
/// \tparam CcsType input matrix type, see \ref CCS
/// \tparam ScalingArray array type for scaling vectors, see \ref Array
/// \tparam PermType permutation type, see \ref BiPermMatrix
/// \param[in] A input matrix
/// \param[in] A_crs \ref CRS format of input \a A
/// \param[in] m0 leading block size
/// \param[in] opt control parameters
/// \param[in] level current factorization level
/// \param[out] s row scaling vector
/// \param[out] t column scaling vector
/// \param[out] p row permutation
/// \param[out] q column permutation
/// \param[in] hdl_zero_diag if \a false (default), assume not saddle point
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
    const typename CcsType::size_type m0, const Options &opt,
    const typename CcsType::size_type level, ScalingArray &s, ScalingArray &t,
    PermType &p, PermType &q, const bool hdl_zero_diag = false) {
  static_assert(!CcsType::ROW_MAJOR, "must be CCS");
  using index_type = typename CcsType::index_type;
  using size_type  = typename CcsType::size_type;

  if (opt.reorder < 0 || opt.reorder >= REORDER_NULL)
    hilucsi_error("invalid reorder flag %d", opt.reorder);

  DefaultTimer timer;

  if (hilucsi_verbose(PRE, opt)) hilucsi_info("performing matching step");

  timer.start();

  auto match_res = do_maching<IsSymm>(A, A_crs, m0, opt.verbose, s, t, p, q,
                                      opt, hdl_zero_diag);

  timer.finish();
  if (hilucsi_verbose(PRE_TIME, opt))
    hilucsi_info("matching took %gs.", (double)timer.time());

  const size_type m = match_res.second;
  if (opt.reorder != REORDER_OFF) {
    if (hilucsi_verbose(PRE, opt)) hilucsi_info("performing reordering step");
    timer.start();

    std::string reorder_name = "AMD";

    auto &            B = match_res.first;
    Array<index_type> P;
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

    if (hilucsi_verbose(PRE_TIME, opt))
      hilucsi_info("reordering %s took %gs.", reorder_name.c_str(),
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
    if (hilucsi_verbose(PRE, opt)) hilucsi_info("reordering skipped");
    p.build_inv();
    q.build_inv();
  }
  return m;
}

}  // namespace hilucsi

#endif  // _HILUCSI_PRE_DRIVER_HPP
