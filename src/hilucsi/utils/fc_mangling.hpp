/*
//@HEADER
//----------------------------------------------------------------------------
//                Copyright (C) 2019 The HILUCSI AUTHORS
//----------------------------------------------------------------------------
//@HEADER
*/

/*!
 * \file hilucsi/utils/fc_mangling.hpp
 * \brief PS-MILU Fortran name mangling interface
 * \authors Qiao,
 */

#ifndef _HILUCSI_UTILS_FCMANGLING_HPP
#define _HILUCSI_UTILS_FCMANGLING_HPP

#ifndef HILUCSI_FC
#  ifdef HILUCSI_FC_UPPER
#    ifdef HILUCSI_FC_APPEND_
#      define HILUCSI_FC(l, U) U##_
#    elif defined(HILUCSI_FC_NC)
#      define HILUCSI_FC(l, U) U
#    elif defined(HILUCSI_FC_APPEND__)
#      define HILUCSI_FC(l, U) U##__
#    else
/* fallback to single _ */
#      define HILUCSI_FC(l, U) U##_
#    endif
#  elif defined(HILUCSI_FC_LOWER)
#    ifdef HILUCSI_FC_APPEND_
#      define HILUCSI_FC(l, U) l##_
#    elif defined(HILUCSI_FC_NC)
#      define HILUCSI_FC(l, U) l
#    elif defined(HILUCSI_FC_APPEND__)
#      define HILUCSI_FC(l, U) l##__
#    else
/* fallback to single _ */
#      define HILUCSI_FC(l, U) l##_
#    endif
#  else
/*
 * if neither upper nor lower is defined, we first check some defines from
 * common blas and lapack implementations
 */
#    ifdef F77_GLOBAL /* cblas */
#      define HILUCSI_FC F77_GLOBAL
#    elif defined(LAPACK_GLOBAL) /* lapacke */
#      define HILUCSI_FC LAPACK_GLOBAL
#    elif defined(OPENBLAS_NEEDBUNDERSCORE) /* openblas configuration */
#      define HILUCSI_FC(l, U) l##_
#    else
/* fallback to lower case with single _ */
#      define HILUCSI_FC(l, U) l##_
#    endif
#  endif
#endif /* HILUCSI_FC */

#endif /* _HILUCSI_UTILS_FCMANGLING_HPP */