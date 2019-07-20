//@HEADER
//----------------------------------------------------------------------------
//                Copyright (C) 2019 The HILUCSI AUTHORS
//----------------------------------------------------------------------------
//@HEADER

/// \file hilucsi/small_scale/solver.hpp
/// \brief Small scale solver main interface
/// \authors Qiao,

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
