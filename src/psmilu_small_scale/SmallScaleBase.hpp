//@HEADER
//----------------------------------------------------------------------------
//                Copyright (C) 2019 The PSMILU AUTHORS
//----------------------------------------------------------------------------
//@HEADER

/// \file psmilu_small_scale/SmallScaleBase.hpp
/// \brief Small scale solver common interface
/// \authors Qiao,

#ifndef _PSMILU_SMALLSCALE_SOLVER_HPP
#define _PSMILU_SMALLSCALE_SOLVER_HPP

#include "psmilu_DenseMatrix.hpp"

namespace psmilu {
namespace internal {

/// \class SmallScaleBase
/// \tparam ValueType value type used, e.g. \a double
/// \ingroup sss
template <class ValueType>
class SmallScaleBase {
 public:
  typedef ValueType                      value_type;       ///< value type
  typedef DenseMatrix<value_type>        dense_type;       ///< dense type
  typedef typename dense_type::size_type size_type;        ///< size type
  constexpr static bool                  IS_DENSE = true;  ///< dense flag

  /// \brief default constructor
  SmallScaleBase() : _mat(), _rank(0u) {}

  // utilities
  inline bool              empty() const { return _mat.empty(); }
  inline size_type         rank() const { return _rank; }
  inline const dense_type &mat() const { return _mat; }
  inline dense_type &      mat() { return _mat; }
  inline bool              is_squared() const { return _mat.is_squared(); }
  inline bool full_rank() const { return _rank != 0u && _rank == _mat.ncols(); }

  /// \brief set operator
  /// \tparam CsType compressed storage type, see \ref CCS and \ref CRS
  /// \param[in] cs compressed storage
  template <class CsType>
  inline void set_matrix(const CsType &cs) {
    _mat = dense_type::from_sparse(cs);
  }

 protected:
  dense_type _mat;   ///< matrix
  size_type  _rank;  ///< rank
};

}  // namespace internal
}  // namespace psmilu

#endif  // _PSMILU_SMALLSCALE_SOLVER_HPP
