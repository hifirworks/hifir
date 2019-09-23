/*
///////////////////////////////////////////////////////////////////////////////
//                This file is part of HILUCSI project                       //
///////////////////////////////////////////////////////////////////////////////
*/

/*!
 * \file hilucsi/small_scale/config.hpp
 * \brief Simple configuration for using LAPACK
 * \authors Qiao,

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
