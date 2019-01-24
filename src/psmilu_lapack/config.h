/*
//@HEADER
//----------------------------------------------------------------------------
//                Copyright (C) 2019 The PSMILU AUTHORS
//----------------------------------------------------------------------------
//@HEADER
*/

/*!
 * \file psmilu_lapack/config.h
 * \brief Simple configuration for using LAPACK
 * \authors Qiao,
 */

#ifndef _PSMILU_LAPACK_CONFIG_HPP
#define _PSMILU_LAPACK_CONFIG_HPP

#include "psmilu_fc_mangling.h"

#ifndef PSMILU_LAPACK_DEFAULT_INT
#  ifdef MKL_INT
#    define PSMILU_LAPACK_DEFAULT_INT MKL_INT
#  elif defined(OPENBLAS_CONFIG_H)
#    ifdef OPENBLAS_USE64BITINT
#      define PSMILU_LAPACK_DEFAULT_INT BLASLONG
#    else
#      define PSMILU_LAPACK_DEFAULT_INT int
#    endif
#  elif defined(lapack_int)
#    define PSMILU_LAPACK_DEFAULT_INT lapack_int
#  else
#    define PSMILU_LAPACK_DEFAULT_INT int
#  endif
#endif /* PSMILU_LAPACK_DEFAULT_INT */

#ifndef __cplusplus
extern "C" {
#endif

/*!
 * \typedef psmilu_lapack_int
 * \brief lapack integer type
 * \ingroup util
 */
typedef PSMILU_LAPACK_DEFAULT_INT psmilu_lapack_int;

#ifndef __cplusplus
}
#endif

#endif /* _PSMILU_LAPACK_CONFIG_HPP */
