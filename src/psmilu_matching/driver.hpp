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

#ifdef mc64_matching

#  include "HSL_MC64.hpp"

#else
#  error "PSMILU requires HSL_MC64, for now..."
#endif

namespace psmilu {

}

#endif  // _PSMILU_MATCHING_DRIVER_HPP
