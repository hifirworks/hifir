//@HEADER
//----------------------------------------------------------------------------
//                Copyright (C) 2019 The PSMILU AUTHORS
//----------------------------------------------------------------------------
//@HEADER

/// \file psmilu_matching/driver.hpp
/// \brief Matching driver interface
/// \authors Qiao,

#ifndef _PSMILU_MATCHING_DRIVER_HPP
#define _PSMILU_MATCHING_DRIVER_HPP

#if 1
#  include "hsl_mc64d.h"
#endif

#include <utility>

#include "psmilu_Array.hpp"
#include "psmilu_CompressedStorage.hpp"
#include "psmilu_log.hpp"

#ifdef mc64_matching

#  include "HSL_MC64.hpp"

#else
#  error "PSMILU requires HSL_MC64, for now..."
#endif

namespace psmilu {
namespace internal {
template <bool IsSymm, class CcsType, class PermType>
inline typename CcsType::size_type check_zero_diags(
    const CcsType &A, const typename CcsType::size_type m0, const PermType &p,
    const PermType &q);

template <bool IsSymm, class ReturnCcsType, class CcsType, class PermType>
inline ReturnCcsType compute_perm_leading_block(
    const CcsType &A, const typename CcsType::size_type m, const PermType &p,
    const PermType &q);
}  // namespace internal

template <bool IsSymm, class CcsType, class ScalingArray, class PermType>
inline std::pair<
    CCS<typename CcsType::value_type, typename CcsType::index_type>,
    typename CcsType::size_type>
do_maching(const CcsType &A, const typename CcsType::size_type m0,
           const int verbose, ScalingArray &s, ScalingArray &t, PermType &p,
           PermType &q) {
  static_assert(!CcsType::ROW_MAJOR, "input must be CCS type");
  using value_type                = typename CcsType::value_type;
  using index_type                = typename CcsType::index_type;
  constexpr static bool ONE_BASED = CcsType::ONE_BASED;
  using match_driver = MatchingDriver<value_type, index_type, ONE_BASED>;
  using input_type   = typename match_driver::input_type;
  constexpr static bool INPUT_ONE_BASED = input_type::ONE_BASED;
  using return_type                     = CCS<value_type, index_type>;
  using size_type                       = typename CcsType::size_type;
  constexpr static value_type ZERO      = value_type();

  const size_type M = A.nrows(), N = A.ncols();
  p.resize(M);
  q.resize(N);
  s.resize(M);
  t.resize(N);
  matching_control_type control;
  matching_info_type    info;
  return_type           B;
  set_default_control(verbose, control, ONE_BASED);
  // first extract matching
  input_type B1 = internal::extract_leading_block4matching<IsSymm>(A, m0);
  // then compute matching
  match_driver::do_maching<IsSymm>(B1, control, p(), q(), s, t, info);
  // then determine zero diags
  const size_type m = internal::check_zero_diags<IsSymm>(B1, m0, p, q);
  if (!INPUT_ONE_BASED && m == m0 && m0 == M) {
    B = return_type(M, N, B1.col_start().data(), B1.row_ind().data(),
                    B1.vals().dat(), true);
  }

  if (INPUT_ONE_BASED) {
  }
}
}  // namespace psmilu

#endif  // _PSMILU_MATCHING_DRIVER_HPP
