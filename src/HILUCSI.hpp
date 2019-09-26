///////////////////////////////////////////////////////////////////////////////
//                This file is part of HILUCSI project                       //
///////////////////////////////////////////////////////////////////////////////

/*!
 * \file HILUCSI.hpp
 * \brief top-level user interface
 * \author Qiao Chen

\verbatim
Copyright (C) 2019 NumGeom Group at Stony Brook University

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
\endverbatim

 */

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