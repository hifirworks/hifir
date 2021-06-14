/*
///////////////////////////////////////////////////////////////////////////////
//                  This file is part of the HIFIR library                   //
///////////////////////////////////////////////////////////////////////////////
*/

/*!
 * \file hif/version.h
 * \brief HIF version header
 * \author Qiao Chen

\verbatim
Copyright (C) 2021 NumGeom Group at Stony Brook University

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

#ifndef _HIF_VERSION_H
#define _HIF_VERSION_H

/// \def HIF_GLOBAL_VERSION
/// \brief HIF global version
/// \ingroup itr

/// \def HIF_MAJOR_VERSION
/// \brief HIF major version
/// \ingroup itr

/// \def HIF_MINOR_VERSION
/// \brief HIF minor version
/// \ingroup itr

/// \def HIF_VERSION
/// \brief HIF uses three-digit version system, i.e. \a global.major.minor
/// \ingroup itr

#define HIF_GLOBAL_VERSION 1
#define HIF_MAJOR_VERSION 0
#define HIF_MINOR_VERSION 0
#define HIF_VERSION \
  (100 * HIF_GLOBAL_VERSION + 10 * HIF_MAJOR_VERSION + HIF_MINOR_VERSION)

#endif /* _HIF_VERSION_H */
