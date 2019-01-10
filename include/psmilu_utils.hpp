//@HEADER
//----------------------------------------------------------------------------
//                Copyright (C) 2019 The PSMILU AUTHORS
//----------------------------------------------------------------------------
//@HEADER

/// \file psmilu_utils.hpp
/// \brief Utility/helper routines
/// \authors Qiao,

#ifndef _PSMILU_UTILS_HPP
#define _PSMILU_UTILS_HPP

#include <algorithm>

namespace psmilu {

/// \brief apply binary search on a sorted array
/// \tparam Iter iterator type
/// \tparam ValueType value type that can be casted to the derefence of Iter
/// \param[in] first starting iterator
/// \param[in] last ending iterator
/// \param[in] v target value
/// \return pair<bool, Iter> that the \a first indicates if or not we find
/// v, while the \a second is the location
/// \todo replace the implementation with one-sided binary search
template <class Iter, class ValueType>
inline std::pair<bool, Iter> find_sorted(Iter first, Iter last,
                                         const ValueType &v) {
  // NOTE, we use the c++ std binary search for now
  auto lower = std::lower_bound(first, last, v);
  return std::make_pair(lower != last && *lower == v, lower);
}

/// \brief convert a given index to c-based index
/// \tparam IndexType integer type
/// \tparam OneBased if nor not the index is Fortran based
/// \param[in] i input index
/// \return C-based index
template <class IndexType, bool OneBased>
inline constexpr IndexType to_c_idx(const IndexType i) {
  return i - static_cast<IndexType>(OneBased);
}

}  // namespace psmilu

#endif
