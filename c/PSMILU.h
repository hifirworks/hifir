/*
//@HEADER
//----------------------------------------------------------------------------
//                Copyright (C) 2019 The PSMILU AUTHORS
//----------------------------------------------------------------------------
//@HEADER
*/

/*!
 * \file PSMILU.h
 * \brief PS-MILU C interface
 * \authors Qiao,
 * \note Compatible with C99, and must be \b C99 or higher!
 */

#ifndef _PSMILU_H
#define _PSMILU_H

/* version */
#include "psmilu_version.h"
/* options */
#include "psmilu_Options.h"

#ifdef __cplusplus
extern "C" {
#endif

/*!
 * \addtogroup c
 * @{
 */

/*!
 * \brief check version
 */
const char *psmilu_get_version(void);

/*!
 * \brief check error code
 * \return if error occurs, then 1 will be returned
 */
int psmilu_error_check(void);

/*!
 * \brief check error message
 * \return message if \ref psmilu_error_check returns 1
 * \sa psmilu_error_check
 */
const char *psmilu_error_msg(void);

/*!
 * \def PSMILU_SINGLE
 * \brief single precision data type
 * \sa PSMILU_DOUBLE
 */
#define PSMILU_SINGLE 100

/*!
 * \def PSMILU_DOUBLE
 * \brief double precision data type
 * \sa PSMILU_SINGLE
 */
#define PSMILU_DOUBLE 101

/*!
 * \def PSMILU_CCS
 * \brief compressed column storage
 * \sa PSMILU_CRS
 */
#define PSMILU_CCS 999

/*!
 * \def PSMILU_CRS
 * \brief compressed row storage
 * \sa PSMILU_CCS
 */
#define PSMILU_CRS 1000

/*!
 * @}
 */ /* c interface group */

#ifdef __cplusplus
}
#endif

#endif /* _PSMILU_H */
