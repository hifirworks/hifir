//@HEADER
//----------------------------------------------------------------------------
//                Copyright (C) 2019 The HILUCSI AUTHORS
//----------------------------------------------------------------------------
//@HEADER

/// \file hilucsi/macros.hpp
/// \brief Useful preprocessing macros
/// \authors Qiao,

#ifndef _HILUCSI_MACROS_HPP
#define _HILUCSI_MACROS_HPP

/*!
 * \addtogroup macros
 * @{
 */

/// \def NDEBUG
/// \brief standard macro for release builds
/// \note default is off, i.e. debug mode, code runs way slower
// #define NDEBUG

/// \def HILUCSI_THROW
/// \brief let HILUCSI use C++ exceptions intead of \a abort
/// \note default value is off
// #define HILUCSI_THROW

/// \def HILUCSI_LOG_PLAIN_PREFIX
/// \brief indicate HILUCSI to drop the ASCII color code in the logging
/// \note default value is off
// #define HILUCSI_LOG_PLAIN_PREFIX

/// \def HILUCSI_FC_UPPER
/// \brief Fortran name mangling uses all upper cases
/// \note default is off
// #define HILUCSI_FC_UPPER

/// \def HILUCSI_FC_LOWER
/// \brief Fortran name mangling uses all lower cases
/// \note default is on (implicit)
// #define HILUCSI_FC_LOWER

/// \def HILUCSI_FC_NC
/// \brief Fortran name mangling appends no _
/// \note default is off
// #define HILUCSI_FC_NC

/// \def HILUCSI_FC_APPEND__
/// \brief Fortran name mangling appends two _
/// \note default is off
// #define HILUCSI_FC_APPEND__

/// \def HILUCSI_FC_APPEND_
/// \brief Fortran name mangling appends single _
/// \note default is on (implicit)
// #define HILUCSI_FC_APPEND_

/// \def HILUCSI_NO_DROP_LE_UF
/// \brief disable applying dropping on L_E and U_F parts for computing Schur
/// \note default is off
// #define HILUCSI_NO_DROP_LE_UF

/// \def HILUCSI_DISABLE_SPACE_DROP
/// \brief completely disable local space control
/// \note default is off
/// \warning Do not turn this on unless you intend to deal with relatively
///          small systems. \a HILUCSI does not have other complexity control
///          mechanism except this one.
// #define HILUCSI_DISABLE_SPACE_DROP

/// \def HILUCSI_RESERVE_FAC
/// \brief reserve memory factor for factorization
/// \note default is 5
///
/// We reserve \a HILUCSI_RESERVE_FAC times \a nnz(A) for the \a L and \a U
/// factors for factorization. We found that this is good enough for most
/// problems without the need of a reallocation.
#ifndef HILUCSI_RESERVE_FAC
#  define HILUCSI_RESERVE_FAC 5
#endif  // HILUCSI_RESERVE_FAC

/// \def HILUCSI_LASTLEVEL_DENSE_SIZE
/// \brief dense size for last level
/// \note default is 1500
///
/// This is the threshold that if the current system is less than
/// \a HILUCSI_LASTLEVEL_DENSE_SIZE, then regardless the sparsity of the
/// Schur complements, HILUCSI will enforce to use dense direct factorization
/// for the sake of actual runtime performance.
#ifndef HILUCSI_LASTLEVEL_DENSE_SIZE
#  define HILUCSI_LASTLEVEL_DENSE_SIZE 1500
#endif

/// \def HILUCSI_ENABLE_MKL_PARDISO
/// \brief enable using sparse direct solver for the last level with MKL-PARDISO
/// \note default is off
///
/// Enabling a complete sparse factorization allows HILUCSI to terminate the
/// process earlier. However, we may not get good performance in terms of
/// runtime and memory usage. But it potentially can help solve some extremely
/// hard problems, where we might need to stop the error accumulation from
/// level to level ASAP.
///
/// It's worth noting that enabling sparse direct factorization almost disable
/// the dense factorization!
// #define HILUCSI_ENABLE_MKL_PARDISO

/// \def HILUCSI_LASTLEVEL_SPARSE_SIZE
/// \brief sparse version of \ref HILUCSI_LASTLEVEL_DENSE_SIZE
/// \note default is 15000
/// \warning not used if \ref HILUCSI_ENABLE_MKL_PARDISO is off
#ifndef HILUCSI_LASTLEVEL_SPARSE_SIZE
#  define HILUCSI_LASTLEVEL_SPARSE_SIZE 15000
#endif  // HILUCSI_LASTLEVEL_SPARSE_SIZE

/// \def HILUCSI_FALLBACK_SPARSE_DIRECT_RATIO
/// \brief fallback to use complete factorization for a certain level
/// \note default ratio is 85%
/// \warning not used if \ref HILUCSI_ENABLE_MKL_PARDISO is off
///
/// There are certain problems where almost all entries can be deferred. In
/// this case, if we have sparse direct solver enabled, we can redo the
/// factorization with a complete version as a fallback plan. This is probably
/// the best we can do, but there is not control over the sizes of systems...
#ifndef HILUCSI_FALLBACK_SPARSE_DIRECT_RATIO
#  define HILUCSI_FALLBACK_SPARSE_DIRECT_RATIO 85
#endif  // HILUCSI_FALLBACK_SPARSE_DIRECT_RATIO

/// \def HILUCSI_MIN_LOCAL_SIZE_PERCTG
/// \brief minimum percentation of the local row and column sizes wrt the
///        the averaged nnz per row/column for row and column, resp.
///
/// The default value is 85, i.e. 85% of averaged nnz is assigned as the
/// the minimum value to the user input nnz per row and column
#ifndef HILUCSI_MIN_LOCAL_SIZE_PERCTG
#  define HILUCSI_MIN_LOCAL_SIZE_PERCTG 85
#endif  // HILUCSI_MIN_LOCAL_SIZE_PERCTG

/*!
 * @}
 */

#endif  // _HILUCSI_MACROS_HPP