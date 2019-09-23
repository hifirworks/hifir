///////////////////////////////////////////////////////////////////////////////
//                This file is part of HILUCSI project                       //
///////////////////////////////////////////////////////////////////////////////

/*!
 * \file hilucsi/macros.hpp
 * \brief Useful preprocessing macros
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

#ifndef _HILUCSI_MACROS_HPP
#define _HILUCSI_MACROS_HPP

/*!
 * \addtogroup macros
 * @{
 */

// Hey! Don't define this for compiling for applications!
#ifdef ONLY_FOR_DOXYGEN

/// \def NDEBUG
/// \brief standard macro for release builds
/// \note default is off, i.e. debug mode, code runs way slower
#  define NDEBUG

/// \def HILUCSI_THROW
/// \brief let HILUCSI use C++ exceptions intead of \a abort
/// \note default value is off
#  define HILUCSI_THROW

/// \def HILUCSI_LOG_PLAIN_PREFIX
/// \brief indicate HILUCSI to drop the ASCII color code in the logging
/// \note default value is off
#  define HILUCSI_LOG_PLAIN_PREFIX

/// \def HILUCSI_FC_UPPER
/// \brief Fortran name mangling uses all upper cases
/// \note default is off
#  define HILUCSI_FC_UPPER

/// \def HILUCSI_FC_LOWER
/// \brief Fortran name mangling uses all lower cases
/// \note default is on (implicit)
#  define HILUCSI_FC_LOWER

/// \def HILUCSI_FC_NC
/// \brief Fortran name mangling appends no _
/// \note default is off
#  define HILUCSI_FC_NC

/// \def HILUCSI_FC_APPEND__
/// \brief Fortran name mangling appends two _
/// \note default is off
#  define HILUCSI_FC_APPEND__

/// \def HILUCSI_FC_APPEND_
/// \brief Fortran name mangling appends single _
/// \note default is on (implicit)
#  define HILUCSI_FC_APPEND_

/// \def HILUCSI_NO_DROP_LE_UF
/// \brief disable applying dropping on L_E and U_F parts for computing Schur
/// \note default is off
#  define HILUCSI_NO_DROP_LE_UF

/// \def HILUCSI_DISABLE_SPACE_DROP
/// \brief completely disable local space control
/// \note default is off
/// \warning Do not turn this on unless you intend to deal with relatively
///          small systems. \a HILUCSI does not have other complexity control
///          mechanism except this one.
#  define HILUCSI_DISABLE_SPACE_DROP

/// \def HILUCSI_ENABLE_MUMPS
/// \brief enabling mumps for last level solver
/// \note default is off
#  define HILUCSI_ENABLE_MUMPS

#endif  // ONLY_FOR_DOXYGEN

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

/// \def HILUCSI_LASTLEVEL_SPARSE_SIZE
/// \brief sparse version of \ref HILUCSI_LASTLEVEL_DENSE_SIZE
/// \note default is 15000
#ifndef HILUCSI_LASTLEVEL_SPARSE_SIZE
#  define HILUCSI_LASTLEVEL_SPARSE_SIZE 15000
#endif  // HILUCSI_LASTLEVEL_SPARSE_SIZE

/// \def HILUCSI_FALLBACK_SPARSE_DIRECT_RATIO
/// \brief fallback to use complete factorization for a certain level
/// \note default ratio is 85%
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