//@HEADER
//----------------------------------------------------------------------------
//                Copyright (C) 2019 The PSMILU AUTHORS
//----------------------------------------------------------------------------
//@HEADER

#ifndef _PSMILU_AMD_DRIVER_HPP
#define _PSMILU_AMD_DRIVER_HPP

#include "amd.hpp"

#include "psmilu_Array.hpp"
#include "psmilu_Options.h"
#include "psmilu_log.hpp"

namespace psmilu {

template <bool IsSymm, class CcsType_C>
inline Array<typename CcsType_C::index_type> run_amd(const CcsType_C &B,
                                                     const Options &  opt) {
  using size_type = typename CcsType_C::size_type;
  static_assert(!CcsType_C::ONE_BASED, "must be C order");
  using index_type = typename CcsType_C::index_type;
  using amd        = AMD<index_type>;

  const size_type m = B.nrows();

  psmilu_assert(B.nrows() == m, "the leading block size should be size(B)");

  // reordering
  double Control[PSMILU_AMD_CONTROL], Info[AMD_INFO];
  amd::defaults(Control);

  // #  ifdef NDEBUG
  //   Control[PSMILU_AMD_CHECKING] = 0;
  // #  endif

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
  if (result != AMD_OK && result != AMD_OK_BUT_JUMBLED) {
    // NOTE that we modified AMD to utilize jumbled return to automatically
    // compute the transpose
    std::stringstream s;
    amd::info(s, Info);
    const std::string msg =
        "AMD returned invalid flag " + std::to_string(result) +
        ", the following message was loaded from AMD info routine:\n" + s.str();
    psmilu_error(msg.c_str());
  }

  if (psmilu_verbose(PRE, opt)) {
    psmilu_info("AMD reordering done with information:\n");
    std::stringstream s;
    amd::info(s, Info);
    psmilu_info(s.str().c_str());
  }

  return P;
}
}  // namespace psmilu

#endif  // _PSMILU_AMD_DRIVER_HPP