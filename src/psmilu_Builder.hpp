//@HEADER
//----------------------------------------------------------------------------
//                Copyright (C) 2019 The PSMILU AUTHORS
//----------------------------------------------------------------------------
//@HEADER

/// \file psmilu_Builder.hpp
/// \brief Top level user class for building MILU preconditioner
/// \authors Qiao,

#ifndef _PSMILU_BUILDER_HPP
#define _PSMILU_BUILDER_HPP

namespace psmilu {

/*!
 * \addtogroup cpp
 * {@
 */

/// \class Builder
/// \tparam ValueType numerical value type, e.g. \a double
/// \tparam IndexType index type, e.g. \a int
/// \tparam OneBased if \a false (default), then assume C index system
template <class ValueType, class IndexType, bool OneBased = false>
class Builder;

/// \typedef C_Builder
/// \tparam ValueType numerical value type, e.g. \a double
/// \tparam IndexType index type, e.g. \a int
/// \sa F_Builder
///
/// This is the type wrapper for C index inputs
template <class ValueType, class IndexType>
using C_Builder = Builder<ValueType, IndexType>;

/// \typedef F_Builder
/// \tparam ValueType numerical value type, e.g. \a double
/// \tparam IndexType index type, e.g. \a int
/// \sa C_Builder
///
/// This is the type wrapper for Fortran index inputs
template <class ValueType, class IndexType>
using F_Builder = Builder<ValueType, IndexType, true>;

/// \typedef C_DefaultBuilder
/// \sa F_DefaultBuilder
///
/// This is the type wrapper for default builder for C index, using \a int as
/// index type and \a double as value type.
typedef C_Builder<double, int> C_DefaultBuilder;

/// \typedef F_DefaultBuilder
/// \sa C_DefaultBuilder
///
/// This is the type wrapper for default builder for Fortran index, using \a int
/// as index type and \a double as value type.
typedef F_Builder<double, int> F_DefaultBuilder;

/*!
 * @}
 */ // group cpp

}  // namespace psmilu

#endif  // _PSMILU_BUILDER_HPP
