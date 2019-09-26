///////////////////////////////////////////////////////////////////////////////
//                This file is part of HILUCSI project                       //
///////////////////////////////////////////////////////////////////////////////

/*!
 * \file hilucsi/small_scale/solver.hpp
 * \brief Small scale solver main interface
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

#ifndef _HILUCSI_SMALLSCALE_SOLVER_HPP
#define _HILUCSI_SMALLSCALE_SOLVER_HPP

#include "hilucsi/small_scale/LUP.hpp"
#include "hilucsi/small_scale/QRCP.hpp"

namespace hilucsi {

/// \enum SmallScaleType
/// \brief type enumerators for small scaled solvers
/// \ingroup itr
enum SmallScaleType {
  SMALLSCALE_LUP = 0,  ///< LU with partial pivoting
  SMALLSCALE_QRCP,     ///< QR with column pivoting
  SMALLSCALE_NONE      ///< invalid flag, for debugging purpose
};

/// \class SmallScaleSolverTrait
/// \brief Trait for selecting backend solver types
/// \tparam SolverType must be the integer values in range
///         [\ref SMALLSCALE_LUP, \ref SMALLSCALE_NONE).
/// \ingroup sss
template <SmallScaleType SolverType>
class SmallScaleSolverTrait;  // trigger complition error

#ifndef DOXYGEN_SHOULD_SKIP_THIS

// LUP
template <>
class SmallScaleSolverTrait<SMALLSCALE_LUP> {
 public:
  template <class ValueType>
  using solver_type = LUP<ValueType>;
};

// QRCP
template <>
class SmallScaleSolverTrait<SMALLSCALE_QRCP> {
 public:
  template <class ValueType>
  using solver_type = QRCP<ValueType>;
};

#endif  // DOXYGEN_SHOULD_SKIP_THIS

}  // namespace hilucsi

#endif  // _HILUCSI_SMALLSCALE_SOLVER_HPP
