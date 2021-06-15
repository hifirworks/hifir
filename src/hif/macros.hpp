///////////////////////////////////////////////////////////////////////////////
//                  This file is part of the HIFIR library                   //
///////////////////////////////////////////////////////////////////////////////

/*!
 * \file hif/macros.hpp
 * \brief Useful preprocessing macros
 * \author Qiao Chen

\verbatim
Copyright (C) 2021 NumGeom Group at Stony Brook University

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
\endverbatim

 */

#ifndef _HIF_MACROS_HPP
#define _HIF_MACROS_HPP

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

/// \def HIF_THROW
/// \brief let HIF use C++ exceptions intead of \a abort
/// \note default value is off
#  define HIF_THROW

/// \def HIF_LOG_PLAIN_PREFIX
/// \brief indicate HIF to drop the ASCII color code in the logging
/// \note default value is off
#  define HIF_LOG_PLAIN_PREFIX

/// \def HIF_HIGH_PRECISION_SOLVE
/// \brief Use higher precision in the back solve in hif
/// \note default is off
#  define HIF_HIGH_PRECISION_SOLVE

/// \def HIF_NO_DROP_LE_UF
/// \brief disable applying dropping on L_E and U_F parts for computing Schur
/// \note default is off
#  define HIF_NO_DROP_LE_UF

/// \def HIF_DISABLE_SPACE_DROP
/// \brief completely disable local space control
/// \note default is off
/// \warning Do not turn this on unless you intend to deal with relatively
///          small systems. \a HIF does not have other complexity control
///          mechanism except this one.
#  define HIF_DISABLE_SPACE_DROP

/// \def HIF_ENABLE_MUMPS
/// \brief enabling mumps for last level solver
/// \note default is off
#  define HIF_ENABLE_MUMPS

#endif  // ONLY_FOR_DOXYGEN

/// \def HIF_RESERVE_FAC
/// \brief reserve memory factor for factorization
/// \note default is 5
///
/// We reserve \a HIF_RESERVE_FAC times \a nnz(A) for the \a L and \a U
/// factors for factorization. We found that this is good enough for most
/// problems without the need of a reallocation.
#ifndef HIF_RESERVE_FAC
#  define HIF_RESERVE_FAC 5
#endif  // HIF_RESERVE_FAC

/// \def HIF_LASTLEVEL_DENSE_SIZE
/// \brief dense size for last level
/// \note default is 1500
///
/// This is the threshold that if the current system is less than
/// \a HIF_LASTLEVEL_DENSE_SIZE, then regardless the sparsity of the
/// Schur complements, HIF will enforce to use dense direct factorization
/// for the sake of actual runtime performance.
#ifndef HIF_LASTLEVEL_DENSE_SIZE
#  define HIF_LASTLEVEL_DENSE_SIZE 2000
#endif

/// \def HIF_LASTLEVEL_SPARSE_SIZE
/// \brief sparse version of \ref HIF_LASTLEVEL_DENSE_SIZE
/// \note default is 15000
#ifndef HIF_LASTLEVEL_SPARSE_SIZE
#  define HIF_LASTLEVEL_SPARSE_SIZE 15000
#endif  // HIF_LASTLEVEL_SPARSE_SIZE

/// \def HIF_FALLBACK_SPARSE_DIRECT_RATIO
/// \brief fallback to use complete factorization for a certain level
/// \note default ratio is 85%
///
/// There are certain problems where almost all entries can be deferred. In
/// this case, if we have sparse direct solver enabled, we can redo the
/// factorization with a complete version as a fallback plan. This is probably
/// the best we can do, but there is not control over the sizes of systems...
#ifndef HIF_FALLBACK_SPARSE_DIRECT_RATIO
#  define HIF_FALLBACK_SPARSE_DIRECT_RATIO 85
#endif  // HIF_FALLBACK_SPARSE_DIRECT_RATIO

/// \def HIF_MIN_LOCAL_SIZE_PERCTG
/// \brief minimum percentation of the local row and column sizes wrt the
///        the averaged nnz per row/column for row and column, resp.
///
/// The default value is 85, i.e. 85% of averaged nnz is assigned as the
/// the minimum value to the user input nnz per row and column
#ifndef HIF_MIN_LOCAL_SIZE_PERCTG
#  define HIF_MIN_LOCAL_SIZE_PERCTG 85
#endif  // HIF_MIN_LOCAL_SIZE_PERCTG

/// \def HIF_DENSE_MODE
/// \brief backend dense kernels, if it's 1 (default), then using QRCP; other
///        values indicate using LU
#ifndef HIF_DENSE_MODE
#  define HIF_DENSE_MODE 1
#endif  // HIF_DENSE_MODE

/*!
 * @}
 */

#endif  // _HIF_MACROS_HPP