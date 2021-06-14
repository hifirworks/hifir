/*
///////////////////////////////////////////////////////////////////////////////
//                  This file is part of HIF project                         //
///////////////////////////////////////////////////////////////////////////////
*/

/*!
 * \file hif/utils/fc_mangling.hpp
 * \brief HIFIR Fortran name mangling interface
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

#ifndef _HIF_UTILS_FCMANGLING_HPP
#define _HIF_UTILS_FCMANGLING_HPP

// determine Fortran name mangling
//  1: lower
//  2: lower_
//  3: lower__
//  4: UPPER
//  5: UPPER_
//  6: UPPER__

// For MKL, use convention 1
#ifdef HIF_HAS_MKL
#  ifdef HIF_FC
#    undef HIF_FC
#  endif
#  define HIF_FC 1
#endif

#ifndef HIF_FC
/*
 * if neither upper nor lower is defined, we first check some defines from
 * common blas and lapack implementations
 */
#    ifdef F77_GLOBAL /* cblas */
#      define HIF_FC F77_GLOBAL
#    elif defined(LAPACK_GLOBAL) /* lapacke */
#      define HIF_FC LAPACK_GLOBAL
#    elif defined(OPENBLAS_NEEDBUNDERSCORE) /* openblas configuration */
#      define HIF_FC(l, U) l##_
#    else
/* fallback to lower case with single _ */
#      define HIF_FC(l, U) l##_
#    endif

#elif HIF_FC == 1
#  undef HIF_FC
#  define HIF_FC(__l, __U) __l
#elif HIF_FC == 2
#  undef HIF_FC
#  define HIF_FC(__l, __U) __l##_
#elif HIF_FC == 3
#  undef HIF_FC
#  define HIF_FC(__l, __U) __l##__
#elif HIF_FC == 4
#  undef HIF_FC
#  define HIF_FC(__l, __U) __U
#elif HIF_FC == 5
#  undef HIF_FC
#  define HIF_FC(__l, __U) __U##_
#elif HIF_FC == 6
#  undef HIF_FC
#  define HIF_FC(__l, __U) __U##__
#else
#  error "Unknown HIF_FC option, must be in (1,2,3,4,5,6)"
#endif

#endif /* _HIF_UTILS_FCMANGLING_HPP */
