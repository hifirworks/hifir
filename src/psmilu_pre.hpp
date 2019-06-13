//@HEADER
//----------------------------------------------------------------------------
//                Copyright (C) 2019 The PSMILU AUTHORS
//----------------------------------------------------------------------------
//@HEADER

/// \file psmilu_pre.hpp
/// \brief Routine to handle preprocessing
/// \authors Qiao,

#ifndef _PSMILU_PRE_HPP
#define _PSMILU_PRE_HPP

#include <cmath>
#include <sstream>
#include <string>

#include "psmilu_AMD/driver.hpp"
#include "psmilu_Array.hpp"
#include "psmilu_Options.h"
#include "psmilu_log.hpp"
#include "psmilu_matching/driver.hpp"

#ifndef PSMILU_DISABLE_BGL
#  include "psmilu_BGL/king.hpp"
#  include "psmilu_BGL/rcm.hpp"
#  include "psmilu_BGL/sloan.hpp"
#endif  // PSMILU_DISABLE_BGL

namespace psmilu {

/// \brief routine to perform preprocessing for improving the quality of input
/// \tparam IsSymm if \a true, the assume the leading block is symmetric
/// \tparam CcsType input matrix type, see \ref CCS
/// \tparam ScalingArray array type for scaling vectors, see \ref Array
/// \tparam PermType permutation type, see \ref BiPermMatrix
/// \param[in] A input matrix
/// \param[in] m0 leading block size
/// \param[in] opt control parameters
/// \param[out] s row scaling vector
/// \param[out] t column scaling vector
/// \param[out] p row permutation
/// \param[out] q column permutation
/// \param[in] hdl_zero_diag if \a false (default), assume not saddle point
/// \return The actual leading block size, no larger than \a m0
/// \ingroup pre
///
/// Notice that, in general, the preprocessing involves two steps: 1) perform
/// matching to improve the diagonal domination, and 2) perform reordering
/// to improve the sparsity of LU decomposition. Currently, the reorder step
/// is done by calling AMD package, which is embedded in PSMILU. The matching
/// step uses HSL_MC64.
///
/// \todo Implement matching algorithm to drop the dependency on MC64.
template <bool IsSymm, class CcsType, class CrsType, class ScalingArray,
          class PermType>
inline typename CcsType::size_type do_preprocessing(
    const CcsType &A, const CrsType &A_crs,
    const typename CcsType::size_type m0, const Options &opt, ScalingArray &s,
    ScalingArray &t, PermType &p, PermType &q,
    const bool hdl_zero_diag = false) {
  static_assert(!CcsType::ROW_MAJOR, "must be CCS");
  using index_type = typename CcsType::index_type;
  using amd        = AMD<index_type>;
  using size_type  = typename CcsType::size_type;

  if (psmilu_verbose(PRE, opt)) psmilu_info("performing matching step");

  const auto match_res =
      do_maching<IsSymm>(A, A_crs, m0, opt.verbose, s, t, p, q, hdl_zero_diag);

  const size_type m = match_res.second;
#ifndef PSMILU_DISABLE_REORDERING
  const auto &B = match_res.first;
  Array<index_type> P;

#  ifdef PSMILU_DISABLE_BGL
  P = run_amd<IsSymm>(B, opt);
#  else
  P = run_rcm<IsSymm>(B, opt);
#  endif  // PSMILU_DISABLE_RCM

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
#else
  p.build_inv();
  q.build_inv();
#endif  // PSMILU_DISABLE_REORDERING

  return m;
}

/// \brief defer dense row and column
/// \tparam CrsType crs matrix, see \ref CRS
/// \tparam CcsType ccs matrix, see \ref CCS
/// \tparam PermType permutation array, see \ref BiPermMatrix
/// \param[in] A_crs crs input
/// \param[in] A_ccs ccs input
/// \param[in] p row permutation
/// \param[in] q column permutation
/// \param[in] m0 leading block size
/// \param[in] N reference system size
/// \return m remaining sparse leading block
template <class CrsType, class CcsType, class PermType>
inline typename CrsType::size_type defer_dense_tail(
    const CrsType &A_crs, const CcsType &A_ccs, const PermType &p,
    const PermType &q, const typename CrsType::size_type m0,
    const typename CrsType::size_type N = 0u) {
  using size_type = typename CrsType::size_type;

  const size_type n = A_crs.nrows();
  psmilu_error_if(m0 > n, "invalid leading block size %zd", m0);
  psmilu_error_if(m0 > A_crs.ncols(), "invalid leading block size %zd", m0);

  // TODO should we use cbrt(n) or cbrt(m0)?
  const size_type dense_thres =
      N ? static_cast<size_type>(std::ceil(std::sqrt((double)N)))
        : static_cast<size_type>(std::ceil(std::sqrt((double)A_crs.nnz())));

  const auto is_dense = [&, dense_thres](const size_type j) -> bool {
    return A_crs.nnz_in_row(p[j]) > dense_thres ||
           A_ccs.nnz_in_col(q[j]) > dense_thres;
  };

  size_type m(m0);

  for (size_type i(m0); i > 0u; --i)
    if (is_dense(i - 1)) --m;

  return m;
}

}  // namespace psmilu

#endif  // _PSMILU_PRE_HPP
