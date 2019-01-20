//@HEADER
//----------------------------------------------------------------------------
//                Copyright (C) 2019 The PSMILU AUTHORS
//----------------------------------------------------------------------------
//@HEADER

/// \file psmilu_Crout.hpp
/// \brief Implementation of modified \a Crout update
/// \authors Qiao,

#ifndef _PSMILU_CROUT_HPP
#define _PSMILU_CROUT_HPP

#include <cstddef>

#include "psmilu_log.hpp"

namespace psmilu {

/// \class Crout
/// \brief Crout update
/// \ingroup alg
class Crout {
 public:
  typedef std::size_t size_type;  ///< size

  /// \brief default constructor, set \ref _step to -1
  Crout() : _step(static_cast<size_type>(-1)) {}

  /// \brief update step
  /// \param[in] step current step of \a Crout update
  inline void set_step(const size_type step) {
    psmilu_assert(
        step == _step + 1u,
        "%zd step should be the advanced value from previous step %zd", step,
        _step);
    _step = step;
  }

  /// \brief check current step
  inline size_type cur_step() const { return _step; }

 protected:
  size_type _step;  ///< current step
};
}  // namespace psmilu

#endif  // _PSMILU_CROUT_HPP
