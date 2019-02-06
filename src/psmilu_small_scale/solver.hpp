//@HEADER
//----------------------------------------------------------------------------
//                Copyright (C) 2019 The PSMILU AUTHORS
//----------------------------------------------------------------------------
//@HEADER

/// \file psmilu_small_scale/solver.hpp
/// \brief Small scale solver main interface
/// \authors Qiao,

#ifndef _PSMILU_SMALLSCALE_SOLVER_HPP
#define _PSMILU_SMALLSCALE_SOLVER_HPP

#include "LUP.hpp"
#include "QRCP.hpp"

namespace psmilu {

/// \enum SmallScaleType
/// \brief type enumerators for small scaled solvers
/// \ingroup sss
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

}  // namespace psmilu

#endif  // _PSMILU_SMALLSCALE_SOLVER_HPP