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

#include <iterator>

#include "psmilu_CompressedStorage.hpp"
#include "psmilu_Options.h"
#include "psmilu_Prec.hpp"
#include "psmilu_prec_solve.hpp"

namespace psmilu {

/*!
 * \addtogroup cpp
 * {@
 */

/// \class Builder
/// \tparam ValueType numerical value type, e.g. \a double
/// \tparam IndexType index type, e.g. \a int
/// \tparam OneBased if \a false (default), then assume C index system
/// \tparam SSSType default is LU with partial pivoting
template <class ValueType, class IndexType, bool OneBased = false,
          SmallScaleType SSSType = SMALLSCALE_LUP>
class Builder {
 protected:
  const static Options _def_opts;  ///< default options
 public:
  typedef ValueType         value_type;                     ///< value type
  typedef Array<value_type> array_type;                     ///< array type
  typedef IndexType         index_type;                     ///< index type
  constexpr static bool     ONE_BASED = OneBased;           ///< index flag
  typedef CRS<value_type, index_type, ONE_BASED> crs_type;  ///< crs type
  typedef typename crs_type::other_type          ccs_type;  ///< ccs type
  constexpr static SmallScaleType sss_type = SSSType;  ///< small scale type
  typedef Precs<value_type, index_type, ONE_BASED, sss_type> precs_type;
  ///< multilevel preconditioner type
  typedef typename precs_type::value_type prec_type;  ///< single level prec
  typedef typename prec_type::size_type   size_type;  ///< size type

  /// \brief check empty or not
  inline bool empty() const { return _precs.empty(); }

  /// \brief check number of levels
  /// \note This function takes \f$\mathcal{O}(1)\f$ since C++11
  inline size_type levels() const { return _precs.size(); }

  // utilities

  /// \brief get constant reference to preconditioners
  inline const precs_type &precs() const { return _precs; }

  /// \brief get constant reference to a specific level
  /// \note This function takes linear time complexity
  inline const prec_type &prec(const size_type level) const {
    psmilu_error_if(level >= levels(), "%zd exceeds the total level number %zd",
                    level, levels());
    return *std::advance(_precs.cbegin(), level);
  }

  inline void compute(const crs_type &A, const Options &opts = _def_opts);
  inline void compute(const ccs_type &A, const Options &opts = _def_opts);
  inline void solve(const array_type &b, array_type &x) const {
    psmilu_error_if(empty(), "MILU-Prec is empty!");
    psmilu_error_if(b.size() != x.size(), "unmatched sizes");
    if (_prec_work.empty())
      _prec_work.resize(
          compute_prec_work_space(_precs.cbegin(), _precs.cend()));
    prec_solve(_precs.cbegin(), b, x, _prec_work);
  }

 protected:
  precs_type         _precs;      ///< multilevel preconditioners
  mutable array_type _prec_work;  ///< preconditioner work space for solving
};

// define default options
template <class ValueType, class IndexType, bool OneBased,
          SmallScaleType SSSType>
const Options Builder<ValueType, IndexType, OneBased, SSSType>::_def_opts =
    get_default_options();

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
