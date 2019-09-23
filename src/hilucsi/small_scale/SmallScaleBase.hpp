///////////////////////////////////////////////////////////////////////////////
//                This file is part of HILUCSI project                       //
///////////////////////////////////////////////////////////////////////////////

/*!
 * \file hilucsi/small_scale/SmallScaleBase.hpp
 * \brief Small scale solver common interface
 * \authors Qiao,

\verbatim
Copyright (C) 2019 NumGeom Group at Stony Brook University

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
\endverbatim

 */

#ifndef _HILUCSI_SMALLSCALE_SMALLSCALEBASE_HPP
#define _HILUCSI_SMALLSCALE_SMALLSCALEBASE_HPP

#include "hilucsi/ds/DenseMatrix.hpp"

namespace hilucsi {
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

  /// \brief set a dense operator, this is needed for H version
  /// \param[in,out] mat input matrix, the data is \b destroyed upon output
  inline void set_matrix(dense_type &&mat) { _mat = std::move(mat); }

 protected:
  dense_type _mat;   ///< matrix
  size_type  _rank;  ///< rank
};

}  // namespace internal
}  // namespace hilucsi

#endif  // _HILUCSI_SMALLSCALE_SMALLSCALEBASE_HPP
