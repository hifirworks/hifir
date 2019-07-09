/*
//@HEADER
//----------------------------------------------------------------------------
//                Copyright (C) 2019 The PSMILU AUTHORS
//----------------------------------------------------------------------------
//@HEADER
*/

/*!
 * \file psmilu_fc_mangling.h
 * \brief PS-MILU Fortran name mangling interface
 * \authors Qiao,
 */

#ifndef _PSMILU_FCMANGLING_H
#define _PSMILU_FCMANGLING_H

#ifndef PSMILU_FC
#  ifdef PSMILU_FC_UPPER
#    ifdef PSMILU_FC_APPEND_
#      define PSMILU_FC(l, U) U##_
#    elif defined(PSMILU_FC_NC)
#      define PSMILU_FC(l, U) U
#    elif defined(PSMILU_FC_APPEND__)
#      define PSMILU_FC(l, U) U##__
#    else
/* fallback to single _ */
#      define PSMILU_FC(l, U) U##_
#    endif
#  elif defined(PSMILU_FC_LOWER)
#    ifdef PSMILU_FC_APPEND_
#      define PSMILU_FC(l, U) l##_
#    elif defined(PSMILU_FC_NC)
#      define PSMILU_FC(l, U) l
#    elif defined(PSMILU_FC_APPEND__)
#      define PSMILU_FC(l, U) l##__
#    else
/* fallback to single _ */
#      define PSMILU_FC(l, U) l##_
#    endif
#  else
/*
 * if neither upper nor lower is defined, we first check some defines from
 * common blas and lapack implementations
 */
#    ifdef F77_GLOBAL /* cblas */
#      define PSMILU_FC F77_GLOBAL
#    elif defined(LAPACK_GLOBAL) /* lapacke */
#      define PSMILU_FC LAPACK_GLOBAL
#    elif defined(OPENBLAS_NEEDBUNDERSCORE) /* openblas configuration */
#      define PSMILU_FC(l, U) l##_
#    else
/* fallback to lower case with single _ */
#      define PSMILU_FC(l, U) l##_
#    endif
#  endif
#endif /* PSMILU_FC */

#endif /* _PSMILU_FCMANGLING_H */