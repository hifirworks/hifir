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
#include <iterator>

/** \addtogroup util
 * @{
 */

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
  // WARNING! When replacing the impl, make sure the behavior aligns with
  // standard lower_bound!
  auto lower = std::lower_bound(first, last, v);
  return std::make_pair(lower != last && *lower == v, lower);
}

/// \brief rotate a subset of vector, s.t. src appears to the left most pos
/// \tparam ArrayType input and output array type
/// \param[in] n **local** size of how many items will be shifted
/// \param[in] src index in **global** range
/// \param[in,out] v input and output array
/// \note Complexity: \f$\mathcal{O}(n)\f$
/// \sa rotate_right
template <class ArrayType>
inline void rotate_left(const typename ArrayType::size_type n,
                        const typename ArrayType::size_type src, ArrayType &v) {
  auto itr_first = v.begin() + src, itr_last = itr_first + n;
  std::rotate(itr_first, itr_first + 1, itr_last);
}

/// \brief rotate a subset of vector, s.t. src appears to the right most pos
/// \tparam ArrayType input and output array type
/// \param[in] n **local** size of how many items will be shifted
/// \param[in] src index in **global** range
/// \param[in,out] v input and output array
/// \note Complexity: \f$\mathcal{O}(n)\f$
/// \sa rotate_left
///
/// For right rotation, we need to use \a reverse_iterator
template <class ArrayType>
inline void rotate_right(const typename ArrayType::size_type n,
                         const typename ArrayType::size_type src,
                         ArrayType &                         v) {
  typedef std::reverse_iterator<typename ArrayType::iterator> iterator;
  // NOTE requiring explicit construction
  iterator itr_first(v.begin() + src + 1), itr_last(itr_first + n);
  std::rotate(itr_first, itr_first + 1, itr_last);
}

/// \brief convert a given index to c-based index
/// \tparam IndexType integer type
/// \tparam OneBased if \a true not the index is Fortran based
/// \param[in] i input index
/// \return C-based index
template <class IndexType, bool OneBased>
inline constexpr IndexType to_c_idx(const IndexType i) {
  return i - static_cast<IndexType>(OneBased);
}

/// \brief convert a C-based index to original input index
/// \tparam IndexType integer type
/// \tparam OneBased if \a true not the index is Fortran based
/// \param[in] i input index
/// \return Original index
template <class IndexType, bool OneBased>
inline constexpr IndexType to_ori_idx(const IndexType i) {
  return i + static_cast<IndexType>(OneBased);
}

}  // namespace psmilu

/** @}*/  // util group

#endif
