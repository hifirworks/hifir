//@HEADER
//----------------------------------------------------------------------------
//                Copyright (C) 2019 The PSMILU AUTHORS
//----------------------------------------------------------------------------
//@HEADER

/// \file psmilu_matching/mc64_driver.hpp
/// \brief Matching and scaling driver interface for MC64
/// \authors Qiao,

#ifndef _PSMILU_MATCHING_MC64DRIVER_HPP
#define _PSMILU_MATCHING_MC64DRIVER_HPP

#include "HSL_MC64.hpp"
#ifndef PSMILU_DISABLE_F77MC64
#  include "MC64.hpp"
#endif  // PSMILU_DISABLE_F77MC64

namespace psmilu {
template <bool IsSymm, class CcsType, class ScaleType, class PermType>
inline void do_mc64(const CcsType &A, const int verbose, ScaleType &s,
                    ScaleType &t, PermType &p, PermType &q) {
  static_assert(!CcsType::ROW_MAJOR, "input must be CCS type");
  using value_type                = typename CcsType::value_type;
  using index_type                = typename CcsType::index_type;
  constexpr static bool ONE_BASED = CcsType::ONE_BASED;
  using hsl_driver = MatchingDriver<value_type, index_type, ONE_BASED>;
#ifdef PSMILU_DISABLE_F77MC64
  matching_control_type control;
  set_default_control(verbose, control, ONE_BASED);
  hsl_driver::template do_matching<IsSymm>(A, control, p, q, s, t);
#else
  using f77_driver = MC64<value_type, index_type, ONE_BASED>;
  if (IsSymm) {
    matching_control_type control;
    set_default_control(verbose, control, ONE_BASED);
    hsl_driver::template do_matching<IsSymm>(A, control, p, q, s, t);
  } else {
    std::cout << "here\n";
    f77_driver::do_matching(A, verbose, q, s, t);
    // set I to p
    for (typename CcsType::size_type i(0); i < A.nrows(); ++i) p[i] = i;
  }
#endif  // PSMILU_DISABLE_F77MC64
}
}  // namespace psmilu

#endif  // _PSMILU_MATCHING_MC64DRIVER_HPP