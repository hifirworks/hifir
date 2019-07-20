/*
//@HEADER
//----------------------------------------------------------------------------
//                Copyright (C) 2019 The HILUCSI AUTHORS
//----------------------------------------------------------------------------
//@HEADER
*/

/*!
 * \file hilucsi/version.h
 * \brief HILUCSI version header
 * \authors Qiao,
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
