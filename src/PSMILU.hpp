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

/// \def PSMILU_GLOBAL_VERSION
/// \brief PSMILU global version
/// \ingroup itr

/// \def PSMILU_MAJOR_VERSION
/// \brief PSMILU major version
/// \ingroup itr

/// \def PSMILU_MINOR_VERSION
/// \brief PSMILU minor version
/// \ingroup itr

/// \def PSMILU_VERSION
/// \brief PSMILU uses three-digit version system, i.e. \a global.major.minor
/// \ingroup itr

#define PSMILU_GLOBAL_VERSION 0
#define PSMILU_MAJOR_VERSION 0
#define PSMILU_MINOR_VERSION 0
#define PSMILU_VERSION                                       \
  (100 * PSMILU_GLOBAL_VERSION + 10 * PSMILU_MAJOR_VERSION + \
   PSMILU_MINOR_VERSION)

namespace psmilu {

/// \brief get the version string representation during runtime
/// \return string representation of version
/// \ingroup itr
inline std::string version() { return std::to_string(PSMILU_VERSION); }
}  // namespace psmilu

// data structure
#include "psmilu_Array.hpp"
#include "psmilu_CompressedStorage.hpp"

// interfaces
#include "psmilu_Options.h"
#include "psmilu_Prec.hpp"

#endif  // _PSMILU_HPP
