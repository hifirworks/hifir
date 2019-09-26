/*
///////////////////////////////////////////////////////////////////////////////
//                This file is part of HILUCSI project                       //
///////////////////////////////////////////////////////////////////////////////
*/

/*!
 * \file hilucsi/version.h
 * \brief HILUCSI version header
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

#ifndef _HILUCSI_VERSION_H
#define _HILUCSI_VERSION_H

/// \def HILUCSI_GLOBAL_VERSION
/// \brief HILUCSI global version
/// \ingroup itr

/// \def HILUCSI_MAJOR_VERSION
/// \brief HILUCSI major version
/// \ingroup itr

/// \def HILUCSI_MINOR_VERSION
/// \brief HILUCSI minor version
/// \ingroup itr

/// \def HILUCSI_VERSION
/// \brief HILUCSI uses three-digit version system, i.e. \a global.major.minor
/// \ingroup itr

#define HILUCSI_GLOBAL_VERSION 1
#define HILUCSI_MAJOR_VERSION 0
#define HILUCSI_MINOR_VERSION 0
#define HILUCSI_VERSION                                        \
  (100 * HILUCSI_GLOBAL_VERSION + 10 * HILUCSI_MAJOR_VERSION + \
   HILUCSI_MINOR_VERSION)

#endif /* _HILUCSI_VERSION_H */
