///////////////////////////////////////////////////////////////////////////////
//                  This file is part of HIF project                         //
///////////////////////////////////////////////////////////////////////////////

/*!
 * \file hif/alg/prec_prod.hpp
 * \brief Multilevel preconditioner matrix-vector products
 * \author Qiao Chen

\verbatim
Copyright (C) 2021 NumGeom Group at Stony Brook University

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

#ifndef _HIF_ALG_PRECPROD_HPP
#define _HIF_ALG_PRECPROD_HPP

#include "hif/alg/prec_solve.hpp"
#include "hif/utils/math.hpp"

namespace hif {

/*!
 * \addtogroup slv
 * @{
 */

/// \brief Given multilevel preconditioner and rhs, compute the matrix-vector
/// \tparam PrecItr this is std::list<Prec>::const_iterator
/// \tparam RhsType right hand side type, see \ref Array
/// \tparam SolType solution type, see \ref Array
/// \tparam WorkType buffer type, generic array interface, i.e. operator[]
/// \param[in] prec_itr current iterator of a single level preconditioner
/// \param[in] b right-hand side vector
/// \param[in] last_dim dimension for last level
/// \param[out] y solution vector
/// \param[out] work work space
///
/// \sa prec_solve
template <class PrecItr, class RhsType, class SolType, class WorkType>
inline void prec_prod(PrecItr prec_itr, const RhsType &b,
                      const std::size_t last_dim, SolType &y, WorkType &work) {
  using prec_type = typename std::remove_const<
      typename std::iterator_traits<PrecItr>::value_type>::type;
  using value_type = typename prec_type::value_type;
  using interface_value_type =
      typename std::remove_reference<decltype(y[0])>::type;
  using size_type            = typename prec_type::size_type;
  using array_type           = Array<value_type>;
  using interface_array_type = Array<interface_value_type>;
  using work_array_type =
      Array<typename std::remove_reference<decltype(work[0])>::type>;
  constexpr static bool WRAP = true;

  hif_assert(b.size() == y.size(), "solution and rhs sizes should match");

  // preparations
  const auto &    prec = *prec_itr;
  const size_type m = prec.m, n = prec.n, nm = n - m;
  const auto &    p_inv = prec.p_inv, &q = prec.q;
  const auto &    s = prec.s, &t = prec.t;

  // first compute the permutated vector (m+1:n), and store it to work(m+1:n)
  for (size_type i(m); i < n; ++i) work[i] = b[q[i]] / t[q[i]];
  // create an array wrapper with size of n-m, of y(m+1:n)
  auto y_mn = interface_array_type(nm, y.data() + m, WRAP);

  // handle the Schur complement
  if (prec.is_last_level()) {
    if (nm) {
      std::copy_n(&work[m], nm, y_mn.begin());
      if (prec.sparse_solver.empty()) {
        if (!prec.dense_solver.empty())
          prec.dense_solver.multiply(y_mn, last_dim);
        else
          prec.symm_dense_solver.multiply(y_mn, last_dim);
      } else
        hif_error("matrix-vector product does not support complete sparse!");
    }
  } else {
    // recursive product
    auto *work_next = &work[n];  // advance to next buffer region
    auto  work_b    = work_array_type(nm, &work[0] + m, WRAP);
    // rec call, note that y_mn should store the solution
    prec_prod(++prec_itr, work_b, last_dim, y_mn, work_next);
  }

  // compute the permuted vector (1:m), and store it to work(1:m)
  for (size_type i(0); i < m; ++i) work[i] = b[q[i]] / t[q[i]];

  // compute L*D*U*work(1:m)
  // NOTE the matrix is unit diagonal with implicit diagonal entries
  prec.U_B.multiply_nt_low(&work[0], y.data());
  // add the unit diagonal portion and scale by D_B
  for (size_type i(0); i < m; ++i) (y[i] += work[i]) *= prec.d_B[i];
  // compute L*y(1:m)
  // NOTE work(1:m) is destroyed
  prec.L_B.multiply_nt_low(y.data(), &work[0]);
  // add the unit diagonal portion
  for (size_type i(0); i < m; ++i) work[i] += y[i];

  // compute F*work(m+1:n) and store to y(1:m)
  // NOTE work(m+1:n) is still the permuted vector
  if (prec.F.ncols()) {
    prec.F.multiply_nt_low(&work[m], y.data());
    for (size_type i(0); i < m; ++i) work[i] += y[i];  // sum up to work(1:m)
  }

  // Now, work(1:m) is the output for this level

  if (nm) {
    // compute inv(LDU)*F*work(m+1:n), where F*work(m+1:n) is stored in y(1:m)
    internal::prec_solve_ldu(prec.U_B, prec.d_B, prec.L_B, y);
    // recompute the permutated vector (1:m) and sum up
    for (size_type i(0); i < m; ++i) y[i] += b[q[i]] / t[q[i]];
    // compute E*y(1:m), and stored to work(m+1:n)
    prec.E.multiply_nt_low(y.data(), &work[m]);
    // add to the solution
    for (size_type i(m); i < n; ++i) work[i] += y[i];
  }
  // std::copy_n(y.data() + m, nm, &work[m]);

  // Now, work(1:n) stores the solution w/o permutation. Let's do permutation
  for (size_type i(0); i < n; ++i) y[i] = work[p_inv[i]] / s[i];
}

/// \brief Given multilevel preconditioner and rhs, compute the matrix-vector
/// \tparam PrecItr this is std::list<Prec>::const_iterator
/// \tparam RhsType right hand side type, see \ref Array
/// \tparam SolType solution type, see \ref Array
/// \tparam WorkType buffer type, generic array interface, i.e. operator[]
/// \param[in] prec_itr current iterator of a single level preconditioner
/// \param[in] b right-hand side vector
/// \param[in] last_dim dimension for last level
/// \param[out] y solution vector
/// \param[out] work work space
///
/// \sa prec_solve
template <class PrecItr, class RhsType, class SolType, class WorkType>
inline void prec_prod_tran(PrecItr prec_itr, const RhsType &b,
                           const std::size_t last_dim, SolType &y,
                           WorkType &work) {
  using prec_type = typename std::remove_const<
      typename std::iterator_traits<PrecItr>::value_type>::type;
  using value_type = typename prec_type::value_type;
  using interface_value_type =
      typename std::remove_reference<decltype(y[0])>::type;
  using size_type            = typename prec_type::size_type;
  using array_type           = Array<value_type>;
  using interface_array_type = Array<interface_value_type>;
  using work_array_type =
      Array<typename std::remove_reference<decltype(work[0])>::type>;
  constexpr static bool WRAP = true;

  hif_assert(b.size() == y.size(), "solution and rhs sizes should match");

  // preparations
  const auto &    prec = *prec_itr;
  const size_type m = prec.m, n = prec.n, nm = n - m;
  const auto &    p = prec.p, &q_inv = prec.q_inv;
  const auto &    s = prec.s, &t = prec.t;

  // first compute the permutated vector (m+1:n), and store it to work(m+1:n)
  for (size_type i(m); i < n; ++i) work[i] = b[p[i]] / s[p[i]];
  // create an array wrapper with size of n-m, of y(m+1:n)
  auto y_mn = interface_array_type(nm, y.data() + m, WRAP);

  // handle the Schur complement
  if (prec.is_last_level()) {
    if (nm) {
      std::copy_n(&work[m], nm, y_mn.begin());
      if (prec.sparse_solver.empty()) {
        if (!prec.dense_solver.empty())
          prec.dense_solver.multiply(y_mn, last_dim, true);
        else
          prec.symm_dense_solver.multiply(y_mn, last_dim);
      } else
        hif_error("matrix-vector product does not support complete sparse!");
    }
  } else {
    // recursive product
    auto *work_next = &work[n];  // advance to next buffer region
    auto  work_b    = work_array_type(nm, &work[0] + m, WRAP);
    // rec call, note that y_mn should store the solution
    prec_prod_tran(++prec_itr, work_b, last_dim, y_mn, work_next);
  }

  // compute the permuted vector (1:m), and store it to work(1:m)
  for (size_type i(0); i < m; ++i) work[i] = b[p[i]] / s[p[i]];

  // compute U^H*conj(D)*L^H*work(1:m)
  // NOTE the matrix is unit diagonal with implicit diagonal entries
  prec.L_B.multiply_t_low(&work[0], y.data());
  // add the unit diagonal portion and scale by conj(D_B)
  for (size_type i(0); i < m; ++i) (y[i] += work[i]) *= conjugate(prec.d_B[i]);
  // compute U^H*y(1:m)
  // NOTE work(1:m) is destroyed
  prec.U_B.multiply_t_low(y.data(), &work[0]);
  // add the unit diagonal portion
  for (size_type i(0); i < m; ++i) work[i] += y[i];

  // compute E^H*work(m+1:n) and store to y(1:m)
  // NOTE work(m+1:n) is still the permuted vector
  if (nm) {
    prec.E.multiply_t_low(&work[m], y.data());
    for (size_type i(0); i < m; ++i) work[i] += y[i];  // sum up to work(1:m)
  }

  // Now, work(1:m) is the output for this level

  if (nm) {
    // compute (inv(LDU))^H*E^H*work(m+1:n), where E^H*work(m+1:n) is stored in
    // y(1:m)
    internal::prec_solve_utdlt(prec.U_B, prec.d_B, prec.L_B, y);
    // recompute the permutated vector (1:m) and sum up
    for (size_type i(0); i < m; ++i) y[i] += b[p[i]] / s[p[i]];
    // compute F^H*y(1:m), and stored to work(m+1:n)
    prec.F.multiply_t_low(y.data(), &work[m]);
    // add to the solution
    for (size_type i(m); i < n; ++i) work[i] += y[i];
  }
  // std::copy_n(y.data() + m, nm, &work[m]);

  // Now, work(1:n) stores the solution w/o permutation. Let's do permutation
  for (size_type i(0); i < n; ++i) y[i] = work[q_inv[i]] / t[i];
}

/*!
 * @}
 */

}  // namespace hif

#endif  // _HIF_ALG_PRECPROD_HPP