//@HEADER
//----------------------------------------------------------------------------
//                Copyright (C) 2019 The HILUCSI AUTHORS
//----------------------------------------------------------------------------
//@HEADER

/// \file hilucsi/ksp/common.hpp
/// \brief common interface (helpers) for KSP solvers
/// \authors Qiao,

#ifndef _HILUCSI_KSP_COMMON_HPP
#define _HILUCSI_KSP_COMMON_HPP

#include <complex>
#include <cstddef>
#include <string>

#include "hilucsi/utils/common.hpp"

namespace hilucsi {
namespace ksp {

/*!
 * \addtogroup ksp
 * @{
 */

/// \brief flags for returned information
enum {
  INVALID_ARGS  = -2,  ///< invalid function arguments
  M_SOLVE_ERROR = -1,  ///< preconditioner solve error
  SUCCESS       = 0,   ///< successful converged
  DIVERGED      = 1,   ///< iteration diverged
  STAGNATED     = 2,   ///< iteration stagnated
};

/// \brief get flag representation
/// \param[in] solver solver name
/// \param[in] flag solver returned flag
inline std::string flag_repr(const std::string &solver, const int flag) {
  switch (flag) {
    case INVALID_ARGS:
      return solver + "_" + "INVALID_ARGS";
    case M_SOLVE_ERROR:
      return solver + "_" + "M_SOLVE_ERROR";
    case SUCCESS:
      return solver + "_" + "SUCCESS";
    case DIVERGED:
      return solver + "_" + "DIVERGED";
    case STAGNATED:
      return solver + "_" + "STAGNATED";
    default:
      return solver + "_" + "UNKNOWN";
  }
}

/// \brief conjugate helper function
template <class T>
inline T conj(const T &v) {
  return std::conj(v);
}

/// \brief instantiate for \a double
template <>
inline double conj(const double &v) {
  return v;
}

/// \brief instantiate for \a flag
template <>
inline float conj(const float &v) {
  return v;
}

/// \class DefaultSettings
/// \tparam V value type
/// \brief parameters for default setting
template <class V>
class DefaultSettings {
 public:
  using scalar_type = typename ValueTypeTrait<V>::value_type;  ///< scalar
  static constexpr scalar_type rtol = sizeof(scalar_type) == 8ul ? 1e-6 : 1e-4;
  ///< default relative tolerance for residual convergence
  static constexpr std::size_t max_iters = 500u;
  ///< maximum number of iterations
};

/*!
 * @}
 */

}  // namespace ksp
}  // namespace hilucsi

#endif  // _HILUCSI_KSP_COMMON_HPP