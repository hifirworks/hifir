/*
//@HEADER
//----------------------------------------------------------------------------
//                Copyright (C) 2019 The PSMILU AUTHORS
//----------------------------------------------------------------------------
//@HEADER
*/

/*!
 * \file psmilu_version.h
 * \brief PS-MILU version header
 * \authors Qiao,
 */

#ifndef _PSMILU_VERSION_H
#define _PSMILU_VERSION_H

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

#endif /* _PSMILU_VERSION_H */
