///////////////////////////////////////////////////////////////////////////////
//                  This file is part of the HIFIR library                   //
///////////////////////////////////////////////////////////////////////////////

/*!
 * \file hif/small_scale/solver.hpp
 * \brief Small scale solver main interface
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

#ifndef _HIF_SMALLSCALE_SOLVER_HPP
#define _HIF_SMALLSCALE_SOLVER_HPP

#include "hif/small_scale/LUP.hpp"
#include "hif/small_scale/QRCP.hpp"
#include "hif/small_scale/SYEIG.hpp"

namespace hif {

/// \class SmallScaleSolverTrait
/// \brief Trait for selecting backend solver types
/// \tparam UseQRCP Boolean flag indicating using QRCP.
/// \ingroup sss
template <bool UseQRCP>
class SmallScaleSolverTrait;  // trigger complition error

#ifndef DOXYGEN_SHOULD_SKIP_THIS

// LUP
template <>
class SmallScaleSolverTrait<false> {
 public:
  template <class ValueType>
  using solver_type = LUP<ValueType>;
};

// QRCP
template <>
class SmallScaleSolverTrait<true> {
 public:
  template <class ValueType>
  using solver_type = QRCP<ValueType>;
};

#endif  // DOXYGEN_SHOULD_SKIP_THIS

}  // namespace hif

#endif  // _HIF_SMALLSCALE_SOLVER_HPP
