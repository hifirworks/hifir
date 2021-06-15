/*
///////////////////////////////////////////////////////////////////////////////
//                  This file is part of the HIFIR library                   //
///////////////////////////////////////////////////////////////////////////////
*/

/*!
 * \file hif/small_scale/config.hpp
 * \brief Simple configuration for using LAPACK
 * \author Qiao Chen

\verbatim
Copyright (C) 2019--2021 NumGeom Group at Stony Brook University

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

#ifndef _HIF_SMALLSCALE_LAPACK_CONFIG_HPP
#define _HIF_SMALLSCALE_LAPACK_CONFIG_HPP

#include "hif/utils/fc_mangling.hpp"

#ifndef HIF_LAPACK_INT
#  ifdef MKL_INT
#    define HIF_LAPACK_INT MKL_INT
#  elif defined(OPENBLAS_CONFIG_H)
#    ifdef OPENBLAS_USE64BITINT
#      define HIF_LAPACK_INT BLASLONG
#    else
#      define HIF_LAPACK_INT int
#    endif
#  elif defined(lapack_int)
#    define HIF_LAPACK_INT lapack_int
#  else
#    define HIF_LAPACK_INT int
#  endif
#endif /* HIF_LAPACK_INT */

/*!
 * \typedef hif_lapack_int
 * \brief lapack integer type
 * \ingroup sss
 */
typedef HIF_LAPACK_INT hif_lapack_int;

#endif /* _HIF_SMALLSCALE_LAPACK_CONFIG_HPP */
