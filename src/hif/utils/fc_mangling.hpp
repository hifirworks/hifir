/*
///////////////////////////////////////////////////////////////////////////////
//                  This file is part of HIF project                         //
///////////////////////////////////////////////////////////////////////////////
*/

/*!
 * \file hif/utils/fc_mangling.hpp
 * \brief PS-MILU Fortran name mangling interface
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

// handle MKL
#if HIF_HAS_MKL
#  ifdef HIF_FC
#    undef HIF_FC
#  endif
#  define HIF_FC(l, U) l
#endif  // HIF_HAS_MKL

#ifndef HIF_FC
#  ifdef HIF_FC_UPPER
#    ifdef HIF_FC_APPEND_
#      define HIF_FC(l, U) U##_
#    elif defined(HIF_FC_NC)
#      define HIF_FC(l, U) U
#    elif defined(HIF_FC_APPEND__)
#      define HIF_FC(l, U) U##__
#    else
/* fallback to single _ */
#      define HIF_FC(l, U) U##_
#    endif
#  elif defined(HIF_FC_LOWER)
#    ifdef HIF_FC_APPEND_
#      define HIF_FC(l, U) l##_
#    elif defined(HIF_FC_NC)
#      define HIF_FC(l, U) l
#    elif defined(HIF_FC_APPEND__)
#      define HIF_FC(l, U) l##__
#    else
/* fallback to single _ */
#      define HIF_FC(l, U) l##_
#    endif
#  else
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
#  endif
#endif /* HIF_FC */

#endif /* _HIF_UTILS_FCMANGLING_HPP */
