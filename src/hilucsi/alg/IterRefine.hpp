///////////////////////////////////////////////////////////////////////////////
//                This file is part of HILUCSI project                       //
///////////////////////////////////////////////////////////////////////////////

/*!
 * \file hilucsi/alg/IterRefine.hpp
 * \brief Iterative refinement for using with HILUCSI
 * \author Qiao Chen

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

#ifndef _HILUCSI_ALG_ITERREFINE_HPP
#define _HILUCSI_ALG_ITERREFINE_HPP

#include "hilucsi/ds/Array.hpp"
#include "hilucsi/ds/CompressedStorage.hpp"

namespace hilucsi {

/// \class IterRefine
/// \tparam ValueType value type used in iterative refinement, e.g., \a double
/// \brief Iterative refinement operator
/// \ingroup ir
/// \note \a ValueType does not need to agree with that used in \a MType
template <class ValueType>
class IterRefine {
 public:
  using value_type = ValueType;                       ///< value type
  using array_type = Array<value_type>;               ///< array type
  using size_type  = typename array_type::size_type;  ///< size type

  /// \brief default constructor
  IterRefine() = default;

  /// \brief Stationary iteration used as refinement
  /// \tparam MType preconditioner, see \ref HILUCSI
  /// \tparam Matrix matrix type, see \ref CRS or \ref CCS
  /// \param[in] M MLILU preconditioner
  /// \param[in] A input matrix
  /// \param[in] b right-hand side vector
  /// \param[in] N number of iterations
  /// \param[out] x solution of Jacobi after \a N iterations
  ///
  /// This function implements the abstracted Jacobi processes as follows
  ///
  /// \f{eqnarray*}{
  ///   \mathbf{Ax}&=&\mathbf{b} \\
  ///   (\mathbf{M+A-M})\mathbf{x}&=&\mathbf{b} \\
  ///   \mathbf{Mx}&=&\mathbf{b}-(\mathbf{A-M})\mathbf{x} \\
  ///   \mathbf{x}&=&\mathbf{M}^{-1}(\mathbf{r}+\mathbf{Mx}) \\
  ///   \mathbf{x}_{k+1}&=&\mathbf{M}^{-1}\mathbf{r}+\mathbf{x}_k
  /// \f}
  template <class MType, class Matrix>
  inline void iter_refine(const MType &M, const Matrix &A, const array_type &b,
                          const size_type N, array_type &x) const {
    if (N <= 1) {
      // if iteration is less than 2, then use original interface
      M.solve(b, x);
      return;
    }
    // now, allocate space
    _init(M.ncols());
    const size_type n(b.size());                   // dimension
    std::fill(x.begin(), x.end(), value_type(0));  // init to zero
    for (size_type i(0); i < N; ++i) {
      std::copy(x.cbegin(), x.cend(), _xk.begin());  // copy rhs to x
      if (i) {
        // starting 2nd iteration
        mt::mv_nt(A, _xk, x);                                 // compute A*xk=x
        for (size_type i(0); i < n; ++i) x[i] = b[i] - x[i];  // residual
      } else
        std::copy_n(b.cbegin(), n, x.begin());
      M.solve(x, _r);  // compute inv(M)*x=r
      for (size_type i(0); i < n; ++i) x[i] = _r[i] + _xk[i];  // update
    }
  }

  /// \brief clear storage
  inline void clear() {
    array_type().swap(_xk);
    array_type().swap(_r);
  }

 protected:
  /// \brief initialize for workspace
  /// \param[in] n row size
  inline void _init(const size_type n) const {
    _xk.resize(n);
    hilucsi_error_if(_xk.empty(), "memory allocation failed");
    _r.resize(n);
    hilucsi_error_if(_r.empty(), "memory allocation failed");
  }

 protected:
  mutable array_type _xk;  ///< previous solution
  mutable array_type _r;   ///< residual
};

}  // namespace hilucsi

#endif  // _HILUCSI_ALG_ITERREFINE_HPP
