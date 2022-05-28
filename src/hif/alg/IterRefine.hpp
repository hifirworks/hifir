///////////////////////////////////////////////////////////////////////////////
//                  This file is part of the HIFIR library                   //
///////////////////////////////////////////////////////////////////////////////

/*!
 * \file hif/alg/IterRefine.hpp
 * \brief Iterative refinement for using with HIF
 * \author Qiao Chen

\verbatim
Copyright (C) 2020 NumGeom Group at Stony Brook University

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
\endverbatim

 */

#ifndef _HIF_ALG_ITERREFINE_HPP
#define _HIF_ALG_ITERREFINE_HPP

#include "hif/ds/Array.hpp"
#include "hif/ds/CompressedStorage.hpp"
#include "hif/utils/math.hpp"
#include "hif/utils/mv_helper.hpp"

namespace hif {

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

  /*!
   * \brief Stationary iteration used as refinement
   * \tparam MType preconditioner, see \ref HIF
   * \tparam Matrix matrix type, see \ref CRS or \ref CCS
   * \param[in] M HIF preconditioner
   * \param[in] A input matrix
   * \param[in] b right-hand side vector
   * \param[in] N number of iterations
   * \param[out] x solution of Jacobi after \a N iterations
   * \param[in] last_dim (optional) dimension for back solve for last level
   *                     default is its numerical rank in \a M
   * \param[in] tran (optional) transpose/Herimitian flag, default is false
   *
   * This function implements the abstracted Jacobi processes as follows
   *
   * \f{eqnarray*}{
   *   \mathbf{Ax}&=&\mathbf{b} \\
   *   (\mathbf{M+A-M})\mathbf{x}&=&\mathbf{b} \\
   *   \mathbf{Mx}&=&\mathbf{b}-(\mathbf{A-M})\mathbf{x} \\
   *   \mathbf{x}&=&\mathbf{M}^{-1}(\mathbf{r}+\mathbf{Mx}) \\
   *   \mathbf{x}_{k+1}&=&\mathbf{M}^{-1}\mathbf{r}+\mathbf{x}_k
   * \f}
   */
  template <class MType, class Matrix, class IArrayType, class OArrayType>
  inline void iter_refine(const MType &M, const Matrix &A, const IArrayType &b,
                          const size_type N, OArrayType &x,
                          const size_type last_dim = static_cast<size_type>(-1),
                          const bool      tran     = false) const {
    if (N <= 1) {
      // if iteration is less than 2, then use original interface
      M.solve(b, x, tran, last_dim);
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
        if (!tran)
          mt::multiply_nt(A, _xk, x);
        else
          multiply(A, _xk, x, true);
        for (size_type j(0); j < n; ++j) x[j] = b[j] - x[j];  // residual
      } else
        std::copy_n(b.cbegin(), n, x.begin());
      M.solve(x, _r, tran, last_dim);  // compute inv(M)*x=r
      for (size_type j(0); j < n; ++j) x[j] = _r[j] + _xk[j];  // update
    }
  }

  /// \brief Stationary iteration used as refinement
  /// \tparam MType preconditioner, see \ref HIF
  /// \tparam Matrix matrix type, see \ref CRS or \ref CCS
  /// \param[in] M HIF preconditioner
  /// \param[in] A input matrix
  /// \param[in] b right-hand side vector
  /// \param[in] N number of iterations
  /// \param[in] betas lower and upper bound of residual norms
  /// \param[out] x solution of Jacobi after \a N iterations
  /// \param[in] last_dim (optional) dimension for back solve for last level
  ///                     default is its numerical rank in \a M
  /// \param[in] tran (optional) transpose/Herimitian flag, default is false
  /// \return Number of refinements and flag. If flag == 0, then it converges;
  ///         if flag > 0, then it diverges; otherwise, it reaches maxit bound.
  template <class MType, class Matrix, class IArrayType, class OArrayType>
  inline std::pair<size_type, int> iter_refine(
      const MType &M, const Matrix &A, const IArrayType &b, const size_type N,
      const double *betas, OArrayType &x,
      const size_type last_dim = static_cast<size_type>(-1),
      const bool      tran     = false) const {
    if (N <= 1) {
      // if iteration is less than 2, then use original interface
      M.solve(b, x, tran, last_dim);
      return std::make_pair(size_type(1), -1);
    }
    // compute norm of RHS
    const double bnorm = norm2(b);
    if (bnorm == 0.0) {
      std::fill(x.begin(), x.end(), value_type(0));
      return std::make_pair(size_type(0), 0);
    }
    // now, allocate space
    _init(M.ncols());
    const size_type n(b.size());                   // dimension
    std::fill(x.begin(), x.end(), value_type(0));  // init to zero
    std::copy_n(b.cbegin(), n, _r.begin());
    size_type iters(0);
    int       flag(0);
    for (;;) {
      M.solve(_r, _xk, tran, last_dim);  // compute xk=inv(M)*x
      for (size_type j(0); j < n; ++j) x[j] += _xk[j];
      if (++iters >= N) {
        flag = -1;
        break;
      }
      if (!tran)
        mt::multiply_nt(A, x, _xk);
      else
        multiply(A, x, _xk, true);
      for (size_type j(0); j < n; ++j) _r[j] = b[j] - _xk[j];
      const double res = norm2(_r) / bnorm;
      if (res <= betas[0]) break;
      if (res > betas[1]) {
        flag = 1;
        break;
      }
    }
    return std::make_pair(iters, flag);
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
    hif_error_if(_xk.empty(), "memory allocation failed");
    _r.resize(n);
    hif_error_if(_r.empty(), "memory allocation failed");
  }

 protected:
  mutable array_type _xk;  ///< previous solution
  mutable array_type _r;   ///< residual
};

}  // namespace hif

#endif  // _HIF_ALG_ITERREFINE_HPP
