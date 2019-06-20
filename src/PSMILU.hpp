//@HEADER
//----------------------------------------------------------------------------
//                Copyright (C) 2019 The PSMILU AUTHORS
//----------------------------------------------------------------------------
//@HEADER

/// \file PSMILU.hpp
/// \brief Main interface file
/// \authors Qiao,
///
/// Only files that are explicitly included in this file are considerred to be
/// user-usable. Other files that are recursively included should not be used
/// unless the user knows what he/she is doing!

#ifndef _PSMILU_HPP
#define _PSMILU_HPP

#if __cplusplus < 201103L
#  error "PSMILU requires at least C++11!"
#endif  // check C++11

#include <string>

#include "psmilu_macros.hpp"
#include "psmilu_version.h"

namespace psmilu {

/// \brief get the version string representation during runtime
/// \return string representation of version
/// \ingroup cpp
inline std::string version() {
  using std::to_string;
  return to_string(PSMILU_GLOBAL_VERSION) + "." +
         to_string(PSMILU_MAJOR_VERSION) + "." +
         to_string(PSMILU_MINOR_VERSION);
}
}  // namespace psmilu

// data structure
#include "psmilu_Array.hpp"
#include "psmilu_CompressedStorage.hpp"
#include "psmilu_DenseMatrix.hpp"

// interfaces
#include "psmilu_Builder.hpp"
#include "psmilu_Options.h"
#include "psmilu_Prec.hpp"

#endif  // _PSMILU_HPP
