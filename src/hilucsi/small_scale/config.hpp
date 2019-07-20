/*
//@HEADER
//----------------------------------------------------------------------------
//                Copyright (C) 2019 The HILUCSI AUTHORS
//----------------------------------------------------------------------------
//@HEADER
*/

/*!
 * \file hilucsi/small_scale/config.hpp
 * \brief Simple configuration for using LAPACK
 * \authors Qiao,
 */

#ifndef _HILUCSI_SMALLSCALE_LAPACK_CONFIG_HPP
#define _HILUCSI_SMALLSCALE_LAPACK_CONFIG_HPP

#include "hilucsi/utils/fc_mangling.hpp"

#ifndef HILUCSI_LAPACK_INT
#  ifdef MKL_INT
#    define HILUCSI_LAPACK_INT MKL_INT
#  elif defined(OPENBLAS_CONFIG_H)
#    ifdef OPENBLAS_USE64BITINT
#      define HILUCSI_LAPACK_INT BLASLONG
#    else
#      define HILUCSI_LAPACK_INT int
#    endif
#  elif defined(lapack_int)
#    define HILUCSI_LAPACK_INT lapack_int
#  else
#    define HILUCSI_LAPACK_INT int
#  endif
#endif /* HILUCSI_LAPACK_INT */

/*!
 * \typedef hilucsi_lapack_int
 * \brief lapack integer type
 * \ingroup sss
 */
typedef HILUCSI_LAPACK_INT hilucsi_lapack_int;

#endif /* _HILUCSI_SMALLSCALE_LAPACK_CONFIG_HPP */
