//@HEADER
//----------------------------------------------------------------------------
//                Copyright (C) 2019 The HILUCSI AUTHORS
//----------------------------------------------------------------------------
//@HEADER

/// \file HILUCSI.hpp
/// \brief top-level user interface
/// \authors Qiao,

#ifndef _HILUCSI_HPP
#define _HILUCSI_HPP

#include "hilucsi/macros.hpp"

#include "hilucsi/builder.hpp"

#include "hilucsi/ksp/interface.hpp"

namespace hilucsi {

/// \brief get the version string representation during runtime
/// \return string representation of version
/// \ingroup itr
inline std::string version() {
  using std::to_string;
  return to_string(HILUCSI_GLOBAL_VERSION) + "." +
         to_string(HILUCSI_MAJOR_VERSION) + "." +
         to_string(HILUCSI_MINOR_VERSION);
}

}  // namespace hilucsi

#endif  // _HILUCSI_HPP