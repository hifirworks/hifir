//@HEADER
//----------------------------------------------------------------------------
//                Copyright (C) 2019 The PSMILU AUTHORS
//----------------------------------------------------------------------------
//@HEADER

/// \file psmilu_macros.hpp
/// \brief Useful preprocessing macros
/// \authors Qiao,

#ifndef _PSMILU_MACROS_HPP
#define _PSMILU_MACROS_HPP

/*!
 * \addtogroup macros
 * @{
 */

/// \def NDEBUG
/// \brief standard macro for release builds
/// \note default is off, i.e. debug mode, code runs way slower
// #define NDEBUG

/// \def PSMILU_THROW
/// \brief let psmilu use C++ exceptions intead of \a abort
/// \note default value is off
// #define PSMILU_THROW

/// \def PSMILU_LOG_PLAIN_PREFIX
/// \brief indicate psmilu to drop the ASCII color code in the logging
/// \note default value is off
// #define PSMILU_LOG_PLAIN_PREFIX

/// \def PSMILU_FC_UPPER
/// \brief Fortran name mangling uses all upper cases
/// \note default is off
// #define PSMILU_FC_UPPER

/// \def PSMILU_FC_LOWER
/// \brief Fortran name mangling uses all lower cases
/// \note default is on (implicit)
// #define PSMILU_FC_LOWER

/// \def PSMILU_FC_NC
/// \brief Fortran name mangling appends no _
/// \note default is off
// #define PSMILU_FC_NC

/// \def PSMILU_FC_APPEND__
/// \brief Fortran name mangling appends two _
/// \note default is off
// #define PSMILU_FC_APPEND__

/// \def PSMILU_FC_APPEND_
/// \brief Fortran name mangling appends single _
/// \note default is on (implicit)
// #define PSMILU_FC_APPEND_

/// \def PSMILU_NO_DROP_LE_UF
/// \brief disable applying dropping on L_E and U_F parts for computing Schur
/// \note default is off
// #define PSMILU_NO_DROP_LE_UF

/// \def PSMILU_USE_CUR_SIZES
/// \brief force psmilu uses current input sizes as thresholds for space control
/// \note default is off
// #define PSMILU_USE_CUR_SIZES

/// \def PSMILU_DISABLE_SPACE_DROP
/// \brief completely disable local space control
/// \note default is off
/// \warning Do not turn this on unless you intend to deal with relatively
///          small systems. \a PSMILU does not have other complexity control
///          mechanism except this one.
// #define PSMILU_DISABLE_SPACE_DROP

/// \def PSMILU_DISABLE_DYN_PVT_THRES
/// \brief let psmilu use static inverse norms for dropping
/// \note default is off
// #define PSMILU_DISABLE_DYN_PVT_THRES

/// \def PSMILU_DISABLE_PRE
/// \brief complete disable preprocessing
/// \note default is off
/// \warning Just stay away from this! This is for testing purpose!
// // // // // // // // #define PSMILU_DISABLE_PRE

/// \def PSMILU_ENABLE_MC64
/// \brief let psmilu enable Fortran77 MC64 support for matching/scaling
/// \note default is off
///
/// When this flag is on, the user needs to provide the library link, since
/// psmilu only provides the wrapper for calling the Fortran 77 MC64 routine
// #define PSMILU_ENABLE_MC64

/// \def PSMILU_DISABLE_BGL
/// \brief disable using Boost Graph Library (BGL)
/// \note default is off, i.e. PSMILU always tries to look for boost
///
/// We think Boost is almost available everywhere, thus having it should not
/// be a problem. If you disable using BGL, then only AMD reordering will be
/// available to you.
// #define PSMILU_DISABLE_BGL

/// \def PSMILU_RESERVE_FAC
/// \brief reserve memory factor for factorization
/// \note default is 5
///
/// We reserve \a PSMILU_RESERVE_FAC times \a nnz(A) for the \a L and \a U
/// factors for factorization. We found that this is good enough for most
/// problems without the need of a reallocation.
#ifndef PSMILU_RESERVE_FAC
#  define PSMILU_RESERVE_FAC 5
#endif  // PSMILU_RESERVE_FAC

/// \def PSMILU_LASTLEVEL_DENSE_SIZE
/// \brief dense size for last level
/// \note default is 1500
///
/// This is the threshold that if the current system is less than
/// \a PSMILU_LASTLEVEL_DENSE_SIZE, then regardless the sparsity of the
/// Schur complements, psmilu will enforce to use dense direct factorization
/// for the sake of actual runtime performance.
#ifndef PSMILU_LASTLEVEL_DENSE_SIZE
#  define PSMILU_LASTLEVEL_DENSE_SIZE 1500
#endif

/// \def PSMILU_ENABLE_MKL_PARDISO
/// \brief enable using sparse direct solver for the last level with MKL-PARDISO
/// \note default is off
///
/// Enabling a complete sparse factorization allows PSMILU to terminate the
/// process earlier. However, we may not get good performance in terms of
/// runtime and memory usage. But it potentially can help solve some extremely
/// hard problems, where we might need to stop the error accumulation from
/// level to level ASAP.
///
/// It's worth noting that enabling sparse direct factorization almost disable
/// the dense factorization!
// #define PSMILU_ENABLE_MKL_PARDISO

/// \def PSMILU_LASTLEVEL_SPARSE_SIZE
/// \brief sparse version of \ref PSMILU_LASTLEVEL_DENSE_SIZE
/// \note default is 15000
/// \warning not used if \ref PSMILU_ENABLE_MKL_PARDISO is off
#ifndef PSMILU_LASTLEVEL_SPARSE_SIZE
#  define PSMILU_LASTLEVEL_SPARSE_SIZE 15000
#endif  // PSMILU_LASTLEVEL_SPARSE_SIZE

/// \def PSMILU_FALLBACK_SPARSE_DIRECT_RATIO
/// \brief fallback to use complete factorization for a certain level
/// \note default ratio is 85%
/// \warning not used if \ref PSMILU_ENABLE_MKL_PARDISO is off
///
/// There are certain problems where almost all entries can be deferred. In
/// this case, if we have sparse direct solver enabled, we can redo the
/// factorization with a complete version as a fallback plan. This is probably
/// the best we can do, but there is not control over the sizes of systems...
#ifndef PSMILU_FALLBACK_SPARSE_DIRECT_RATIO
#  define PSMILU_FALLBACK_SPARSE_DIRECT_RATIO 85
#endif  // PSMILU_FALLBACK_SPARSE_DIRECT_RATIO

/*!
 * @}
 */

#endif  // _PSMILU_MACROS_HPP