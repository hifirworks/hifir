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
#include "psmilu_CompressedStorage.hpp"
#include "psmilu_Options.h"
#include "psmilu_Timer.hpp"
#include "psmilu_log.hpp"
#include "psmilu_matching/driver.hpp"

#ifndef PSMILU_DISABLE_BGL
#  include "psmilu_BGL/king.hpp"
#  include "psmilu_BGL/rcm.hpp"
#  include "psmilu_BGL/sloan.hpp"
#endif  // PSMILU_DISABLE_BGL

namespace psmilu {

/*!
 * \addtogroup pre
 * @{
 */

/// \brief routine to perform preprocessing for improving the quality of input
/// \tparam IsSymm if \a true, the assume the leading block is symmetric
/// \tparam CcsType input matrix type, see \ref CCS
/// \tparam ScalingArray array type for scaling vectors, see \ref Array
/// \tparam PermType permutation type, see \ref BiPermMatrix
/// \param[in] A input matrix
/// \param[in] m0 leading block size
/// \param[in] opt control parameters
/// \param[in] level current factorization level
/// \param[out] s row scaling vector
/// \param[out] t column scaling vector
/// \param[out] p row permutation
/// \param[out] q column permutation
/// \param[in] hdl_zero_diag if \a false (default), assume not saddle point
/// \return The actual leading block size, no larger than \a m0
///
/// Notice that, in general, the preprocessing involves two steps: 1) perform
/// matching to improve the diagonal domination, and 2) perform reordering
/// to improve the sparsity of LU decomposition. Currently, the reorder step
/// is done by calling AMD package, which is embedded in PSMILU. The matching
/// step uses HSL_MC64.
template <bool IsSymm, class CcsType, class CrsType, class ScalingArray,
          class PermType>
inline typename CcsType::size_type do_preprocessing(
    const CcsType &A, const CrsType &A_crs,
    const typename CcsType::size_type m0, const Options &opt,
    const typename CcsType::size_type level, ScalingArray &s, ScalingArray &t,
    PermType &p, PermType &q, const bool hdl_zero_diag = false) {
  static_assert(!CcsType::ROW_MAJOR, "must be CCS");
  using index_type = typename CcsType::index_type;
  using amd        = AMD<index_type>;
  using size_type  = typename CcsType::size_type;

  if (opt.reorder < 0 || opt.reorder >= REORDER_NULL)
    psmilu_error("invalid reorder flag %d", opt.reorder);

  DefaultTimer timer;

  if (psmilu_verbose(PRE, opt)) psmilu_info("performing matching step");

  timer.start();

  const auto match_res = do_maching<IsSymm>(A, A_crs, m0, opt.verbose, s, t, p,
                                            q, opt, hdl_zero_diag);

  timer.finish();
  if (psmilu_verbose(PRE_TIME, opt))
    psmilu_info("matching took %gs.", (double)timer.time());

  const size_type m = match_res.second;
  if (opt.reorder != REORDER_OFF) {
    if (psmilu_verbose(PRE, opt)) psmilu_info("performing reordering step");
    timer.start();

    std::string reorder_name = "AMD";

    const auto &      B = match_res.first;
    Array<index_type> P;
// The reordering should treat output as general systems
#ifdef PSMILU_DISABLE_BGL
    if (opt.reorder != REORDER_AUTO && opt.reorder != REORDER_AMD)
      psmilu_warning(
          "%s ordering is only available in BGL, rebuild with Boost\n"
          "Ordering method fallback to AMD",
          get_reorder_name(opt).c_str());
    P = run_amd<false>(B, opt);
#else
    switch (opt.reorder) {
      // case REORDER_AUTO: {
      //   if (IsSymm && level == 1u) {
      //     P            = run_rcm<false>(B, opt);
      //     reorder_name = "RCM";
      //   } else {
      //     P            = run_amd<false>(B, opt);
      //     reorder_name = "AMD";
      //   }
      // } break;
      case REORDER_AUTO:
      case REORDER_AMD:
        P = run_amd<false>(B, opt);
        break;
      case REORDER_RCM:
        P = run_rcm<false>(B, opt);
        break;
      case REORDER_KING:
        P = run_king<false>(B, opt);
        break;
      default:
        P = run_sloan<false>(B, opt);
        break;
    }
    if (opt.reorder != REORDER_AUTO) reorder_name = get_reorder_name(opt);
#endif  // PSMILU_DISABLE_RCM

    timer.finish();

    if (psmilu_verbose(PRE_TIME, opt))
      psmilu_info("reordering %s took %gs.", reorder_name.c_str(),
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
    if (psmilu_verbose(PRE, opt)) psmilu_info("reordering skipped");
    p.build_inv();
    q.build_inv();
  }
  return m;
}

// only for complete general matrices
template <class CcsType, class CrsType, class ScalingArray, class PermType>
inline typename CcsType::size_type do_preprocessing2(
    const CcsType &A, const CrsType &A_crs, const Options &opt,
    const typename CcsType::size_type level, ScalingArray &s, ScalingArray &t,
    PermType &p, PermType &q, const bool hdl_zero_diag = false) {
  static_assert(!CcsType::ROW_MAJOR, "must be CCS");
  using index_type = typename CcsType::index_type;
  using amd        = AMD<index_type>;
  using size_type  = typename CcsType::size_type;
  using value_type = typename CcsType::value_type;

  if (opt.pre_reorder < 0 || opt.pre_reorder >= REORDER_NULL)
    psmilu_error("invalid pre_reorder flag %d", opt.pre_reorder);

  if (psmilu_verbose(PRE, opt)) psmilu_info("performing pre-reordering");

  DefaultTimer timer;

  std::string reorder_name = "AMD";
  PermType    P;
#ifdef PSMILU_DISABLE_BGL
  if (opt.pre_reorder != REORDER_AMD && opt.pre_reorder != REORDER_AUTO)
    psmilu_warning(
        "%s ordering is only available in BGL, rebuild with Boost\n"
        "Ordering method fallback to AMD",
        get_reorder_name(opt).c_str());
#else
  timer.start();
  if (opt.pre_reorder == REORDER_AMD || opt.pre_reorder == REORDER_AUTO) {  // 1
#endif  // PSMILU_DISABLE_RCM
  CCS<value_type, index_type> A2;
  if (CcsType::ONE_BASED) {
    // for Fortran index, we need to convert to C
    A2.resize(A.nrows(), A.ncols());
    A2.col_start().resize(A.ncols() + 1);
    psmilu_error_if(A2.col_start().status() == DATA_UNDEF,
                    "memory allocation failed");
    A2.row_ind().resize(A.nnz());
    psmilu_error_if(A2.row_ind().status() == DATA_UNDEF,
                    "memory allocation failed");
    for (size_type i(0); i < A.ncols(); ++i)
      A2.col_start()[i + 1] = A.col_start()[i + 1] - 1;
    A2.col_start().front() = 0;
    for (size_type i(0); i < A.nnz(); ++i) A2.row_ind()[i] = A.row_ind()[i] - 1;
  } else
    A2 = CCS<value_type, index_type>(
        A.nrows(), A.ncols(), (index_type *)A.col_start().data(),
        (index_type *)A.row_ind().data(), (value_type *)A.vals().data(), true);
  P() = run_amd<false>(A2, opt);
#ifndef PSMILU_DISABLE_BGL
}  // 1
else {
  switch (opt.pre_reorder) {
    case REORDER_RCM:
      P()          = run_rcm<false>(A, opt);
      reorder_name = "RCM";
      break;
    case REORDER_KING:
      P()          = run_king<false>(A, opt);
      reorder_name = "King";
      break;
    default:
      P()          = run_sloan<false>(A, opt);
      reorder_name = "Sloan";
      break;
  }
}
#endif  // PSMILU_DISABLE_RCM
timer.finish();
if (psmilu_verbose(PRE_TIME, opt))
  psmilu_info("pre-reordering %s took %gs.", reorder_name.c_str(),
              timer.time());
const size_type n(A.nrows());  // assume squared
if (P.is_eye())
  return do_preprocessing<false>(A, A_crs, n, opt, level, s, t, p, q,
                                 hdl_zero_diag);
P.inv().resize(n);
psmilu_error_if(P.inv().status() == DATA_UNDEF, "memory allocation failed");
P.build_inv();
const CcsType AA     = A.compute_perm(P.inv(), P());
const CrsType AA_crs = CrsType(AA);

// do regular preprocessing
const size_type m = do_preprocessing<false>(AA, AA_crs, n, opt, level, s, t, p,
                                            q, hdl_zero_diag);

Array<value_type> buf(n);
for (size_type i(0); i < n; ++i) buf[P[i]] = s[i];
s.swap(buf);
for (size_type i(0); i < n; ++i) buf[P[i]] = t[i];
t.swap(buf);
do {
  auto &i_buf = p.inv();
  for (size_type i(0); i < n; ++i) i_buf[i] = P[p[i]];
  p().swap(i_buf);
} while (false);
do {
  auto &i_buf = q.inv();
  for (size_type i(0); i < n; ++i) i_buf[i] = P[q[i]];
  q().swap(i_buf);
} while (false);
p.build_inv();
q.build_inv();

return m;
}  // namespace psmilu

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

/*!
 * @}
 */

}  // namespace psmilu

#endif  // _PSMILU_PRE_HPP
