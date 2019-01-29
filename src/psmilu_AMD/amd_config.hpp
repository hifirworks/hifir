//@HEADER
//----------------------------------------------------------------------------
//                Copyright (C) 1996-2015, Timothy A. Davis,
//                                         Patrick R. Amestoy, and
//                                         Iain S. Duff.
//
//                Copyright (C) 2019 The PSMILU AUTHORS
//----------------------------------------------------------------------------
//@HEADER

#ifndef _PSMILU_AMD_CONFIG_HPP
#define _PSMILU_AMD_CONFIG_HPP

#include <cstdint>  // c++11

#ifndef AMD_H
#  define AMD_CONTROL 5
#  define AMD_INFO 20

/* contents of Control */
#  define AMD_DENSE 0
#  define AMD_AGGRESSIVE 1

/* default Control settings */
#  define AMD_DEFAULT_DENSE 10.0
#  define AMD_DEFAULT_AGGRESSIVE 1

/* contents of Info */
#  define AMD_STATUS 0
#  define AMD_N 1
#  define AMD_NZ 2
#  define AMD_SYMMETRY 3
#  define AMD_NZDIAG 4
#  define AMD_NZ_A_PLUS_AT 5
#  define AMD_NDENSE 6
#  define AMD_MEMORY 7
#  define AMD_NCMPA 8
#  define AMD_LNZ 9
#  define AMD_NDIV 10
#  define AMD_NMULTSUBS_LDL 11
#  define AMD_NMULTSUBS_LU 12
#  define AMD_DMAX 13

/* ------------------------------------------------------------------------- */
/* return values of AMD */
/* ------------------------------------------------------------------------- */

#  define AMD_OK 0
#  define AMD_OUT_OF_MEMORY -1
#  define AMD_INVALID -2
#  define AMD_OK_BUT_JUMBLED 1

#endif  // AMD_H header checking

#ifdef FLIP
#  undef FLIP
#endif

#ifdef MAX
#  undef MAX
#endif

#ifdef MIN
#  undef MIN
#endif

#ifdef EMPTY
#  undef EMPTY
#endif

#define FLIP(i) (-(i)-2)
#define UNFLIP(i) ((i < EMPTY) ? FLIP(i) : (i))

/* for integer MAX/MIN, or for doubles when we don't care how NaN's behave: */
#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#define MIN(a, b) (((a) < (b)) ? (a) : (b))

#define SIZE_T_MAX SIZE_MAX

#ifdef SuiteSparse_malloc
#  undef SuiteSparse_malloc
#endif

#ifdef SuiteSparse_free
#  undef SuiteSparse_free
#endif

// namespace
#ifndef PSMILU_AMD_NAMESPACE
#  define PSMILU_AMD_NAMESPACE psmilu
#endif  // PSMILU_AMD_NAMESPACE

#define PSMILU_AMD_NAMESPACE_BEGIN namespace psmilu {
#define PSMILU_AMD_NAMESPACE_END }

#define PSMILU_AMD_USING_NAMESPACE using namespace PSMILU_AMD_NAMESPACE

#endif  // _PSMILU_AMD_CONFIG_HPP