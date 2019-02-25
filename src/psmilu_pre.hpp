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

#include <sstream>
#include <string>

#include "psmilu_AMD/amd.hpp"
#include "psmilu_Array.hpp"
#include "psmilu_Options.h"
#include "psmilu_log.hpp"
#include "psmilu_matching/driver.hpp"

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
/// \ingroup pre
///
/// Notice that, in general, the preprocessing involves two steps: 1) perform
/// matching to improve the diagonal domination, and 2) perform reordering
/// to improve the sparsity of LU decomposition. Currently, the reorder step
/// is done by calling AMD package, which is embedded in PSMILU. The matching
/// step uses HSL_MC64.
///
/// \todo Implement matching algorithm to drop the dependency on MC64.
template <bool IsSymm, class CcsType, class ScalingArray, class PermType>
inline void do_preprocessing(const CcsType &                   A,
                             const typename CcsType::size_type m0,
                             const Options &opt, ScalingArray &s,
                             ScalingArray &t, PermType &p, PermType &q) {
  static_assert(!CcsType::ROW_MAJOR, "must be CCS");
  using index_type = typename CcsType::index_type;
  using amd        = AMD<index_type>;
  using size_type  = typename CcsType::size_type;

  // TODO we shall put this in Options?
  constexpr static bool let_us_ignore_zero_entries_for_now = false;

  if (psmilu_verbose(PRE, opt)) psmilu_info("performing matching step");

  const auto match_res = do_maching<IsSymm>(A, m0, opt.verbose, s, t, p, q,
                                            let_us_ignore_zero_entries_for_now);

  const auto &    B = match_res.first;
  const size_type m = match_res.second;

  psmilu_assert(B.nrows() == m, "the leading block size should be size(B)");

  if (psmilu_verbose(PRE, opt))
    psmilu_info("matching step done with input/output block sizes %zd/%zd.", m0,
                m);

  // reordering
  double Control[PSMILU_AMD_CONTROL], Info[AMD_INFO];
  amd::defaults(Control);

#ifdef NDEBUG
  Control[PSMILU_AMD_CHECKING] = 0;
#endif

  Control[PSMILU_AMD_SYMM_FLAG] = !IsSymm;

  if (psmilu_verbose(PRE, opt)) {
    psmilu_info("performing AMD reordering");
    std::stringstream s;
    amd::control(s, Control);
    psmilu_info(s.str().c_str());
  }
  Array<index_type> P(m);
  psmilu_error_if(P.status() == DATA_UNDEF, "memory allocation failed");
  const int result = amd::order(m, B.col_start().data(), B.row_ind().data(),
                                P.data(), Control, Info);
  if (result != AMD_OK) {
    if (result == AMD_OK_BUT_JUMBLED)
      psmilu_warning(
          "the input matrix has duplicated entries or is not "
          "sorted, this should not happen, cont anyway...");
    else {
      std::stringstream s;
      amd::info(s, Info);
      const std::string msg =
          "AMD returned invalid flag " + std::to_string(result) +
          ", the following message was loaded from AMD info routine:\n" +
          s.str();
      psmilu_error(msg.c_str());
    }
  }

  if (psmilu_verbose(PRE, opt)) {
    psmilu_info("AMD reordering done with information:\n");
    std::stringstream s;
    amd::info(s, Info);
    psmilu_info(s.str().c_str());
  }

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
}
}  // namespace psmilu

#endif  // _PSMILU_PRE_HPP
