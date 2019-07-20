//@HEADER
//----------------------------------------------------------------------------
//                Copyright (C) 2019 The PSMILU AUTHORS
//----------------------------------------------------------------------------
//@HEADER

/// \file hilucsi/alg/prec_solve.hpp
/// \brief Multilevel preconditioner solver
/// \authors Qiao,

#ifndef _HILUCSI_ALG_PRECSOLVE_HPP
#define _HILUCSI_ALG_PRECSOLVE_HPP

// use generic programming, implicitly assume the API in Prec

#include <algorithm>
#include <iterator>
#include <list>
#include <type_traits>

#include "hilucsi/ds/Array.hpp"
#include "hilucsi/utils/common.hpp"

namespace hilucsi {
namespace internal {

/*!
 * \addtogroup slv
 * @{
 */

/// \brief triangular solving kernel in \a hilucsi_solve algorithm
/// \tparam CrsType crs storage for input U and L, see \ref CRS
/// \tparam DiagType diagonal vector type, see \ref Array
/// \tparam RhsType rhs and solution type, generial array interface
/// \param[in] U strictly upper part
/// \param[in] d diagonal vector
/// \param[in] L strictly lower part
/// \param[in,out] y input rhs and solution upon output
/// \ingroup slv
///
/// This routine is to solve:
///
/// \f[
///   \boldsymbol{y}&=\boldsymbol{U}^{-1}\boldsymbol{D}^{-1}\boldsymbol{L}^{-1}
///     \boldsymbol{y}
/// \f]
///
/// The overall complexity is linear assuming the local nnz are bounded by a
/// constant.
// The CRS version
template <class CrsType, class DiagType, class RhsType, typename T = void>
inline typename std::enable_if<CrsType::ROW_MAJOR, T>::type prec_solve_udl_inv(
    const CrsType &U, const DiagType &d, const CrsType &L, RhsType &y) {
  using size_type  = typename CrsType::size_type;
  using value_type = typename CrsType::value_type;
  static_assert(CrsType::ROW_MAJOR, "must be crs");

  const size_type m = U.nrows();
  hilucsi_assert(U.ncols() == m, "U must be squared");
  hilucsi_assert(L.nrows() == L.ncols(), "L must be squared");
  hilucsi_assert(d.size() >= m, "diagonal must be no smaller than system size");
  if (!m) return;

  // y=inv(L)*y
  for (size_type j = 1u; j < m; ++j) {
    auto       itr   = L.col_ind_cbegin(j);
    auto       v_itr = L.val_cbegin(j);
    value_type tmp(0);
    for (auto last = L.col_ind_cend(j); itr != last; ++itr, ++v_itr)
      tmp += *v_itr * y[*itr];
    y[j] -= tmp;
  }

  // y=inv(D)*y
  for (size_type i = 0u; i < m; ++i) y[i] /= d[i];

  // y = inv(U)*y
  for (size_type j = m - 1; j != 0u; --j) {
    const size_type j1    = j - 1;
    auto            itr   = U.col_ind_cbegin(j1);
    auto            v_itr = U.val_cbegin(j1);
    value_type      tmp(0);
    for (auto last = U.col_ind_cend(j1); itr != last; ++itr, ++v_itr)
      tmp += *v_itr * y[*itr];
    y[j1] -= tmp;
  }
}

/// \brief triangular solving kernel in \a hilucsi_solve algorithm
/// \tparam CcsType ccs storage for input U and L, see \ref CCS
/// \tparam DiagType diagonal vector type, see \ref Array
/// \tparam RhsType rhs and solution type, generial array interface
/// \param[in] U strictly upper part
/// \param[in] d diagonal vector
/// \param[in] L strictly lower part
/// \param[in,out] y input rhs and solution upon output
/// \ingroup slv
///
/// This routine is to solve:
///
/// \f[
///   \boldsymbol{y}&=\boldsymbol{U}^{-1}\boldsymbol{D}^{-1}\boldsymbol{L}^{-1}
///     \boldsymbol{y}
/// \f]
///
/// The overall complexity is linear assuming the local nnz are bounded by a
/// constant.
// CCS version
template <class CcsType, class DiagType, class RhsType, typename T = void>
inline typename std::enable_if<!CcsType::ROW_MAJOR, T>::type prec_solve_udl_inv(
    const CcsType &U, const DiagType &d, const CcsType &L, RhsType &y) {
  using size_type = typename CcsType::size_type;
  static_assert(!CcsType::ROW_MAJOR, "must be ccs");
  using rev_iterator   = std::reverse_iterator<decltype(U.row_ind_cbegin(0))>;
  using rev_v_iterator = std::reverse_iterator<decltype(U.val_cbegin(0))>;

  const size_type m = U.nrows();
  hilucsi_assert(U.ncols() == m, "U must be squared");
  hilucsi_assert(L.nrows() == L.ncols(), "L must be squared");
  hilucsi_assert(d.size() >= m, "diagonal must be no smaller than system size");

  // y=inv(L)*y
  for (size_type j = 0u; j < m; ++j) {
    const auto y_j = y[j];
    auto       itr = L.row_ind_cbegin(j);
#ifndef NDEBUG
    if (itr != L.row_ind_cend(j))
      hilucsi_error_if(size_type(*itr) <= j, "must be strictly lower part!");
#endif
    auto v_itr = L.val_cbegin(j);
    for (auto last = L.row_ind_cend(j); itr != last; ++itr, ++v_itr) {
      hilucsi_assert(size_type(*itr) < m, "%zd exceeds system size %zd",
                     size_type(*itr), m);
      y[*itr] -= *v_itr * y_j;
    }
  }

  // y=inv(D)*y
  for (size_type i = 0u; i < m; ++i) y[i] /= d[i];

  // y = inv(U)*y, NOTE since U is unit diagonal, no need to handle j == 0
  if (m)
    for (size_type j = m - 1; j != 0u; --j) {
      const auto y_j = y[j];
      auto       itr = rev_iterator(U.row_ind_cend(j));
#ifndef NDEBUG
      if (itr != rev_iterator(U.row_ind_cbegin(j)))
        hilucsi_error_if(size_type(*itr) >= j, "must be strictly upper part");
#endif
      auto v_itr = rev_v_iterator(U.val_cend(j));
      for (auto last = rev_iterator(U.row_ind_cbegin(j)); itr != last;
           ++itr, ++v_itr)
        y[*itr] -= *v_itr * y_j;
    }
}

/*!
 * @}
 */

}  // namespace internal

/*!
 * \addtogroup slv
 * @{
 */

/// \brief Given multilevel preconditioner and rhs, solve the solution
/// \tparam PrecItr this is std::list<Prec>::const_iterator
/// \tparam WorkType buffer type, generic array interface, i.e. operator[]
/// \param[in] prec_itr current iterator of a single level preconditioner
/// \param[in] b right-hand side vector
/// \param[out] y solution vector
/// \param[out] work work space
///
/// Regarding preconditioner, C++ generic programming is used, i.e. the member
/// interfaces of \ref Prec are implicitly assumed. Also, for \b most cases,
/// the work array should be size of n, which is the system size of first level.
/// However, if we have too many levels, the size of \a work may not be system
/// size. Since we use a \b generic array interface for the buffer type, it's
/// not easy to check out-of-bound accessing, as a result, it's critical to
/// ensure the size of work buffer. Make sure work size is no smaller than
/// the size given by \ref compute_prec_work_space
///
/// \sa compute_prec_work_space
template <class PrecItr, class WorkType>
inline void prec_solve(
    PrecItr prec_itr,
    const Array<typename std::remove_const<
        typename std::iterator_traits<PrecItr>::value_type>::type::value_type>
        &b,
    Array<typename std::remove_const<
        typename std::iterator_traits<PrecItr>::value_type>::type::value_type>
        &     y,
    WorkType &work) {
  using prec_type = typename std::remove_const<
      typename std::iterator_traits<PrecItr>::value_type>::type;
  using value_type           = typename prec_type::value_type;
  using size_type            = typename prec_type::size_type;
  using array_type           = Array<value_type>;
  constexpr static bool WRAP = true;

  static_assert(
      std::is_same<typename std::remove_reference<decltype(work[0])>::type,
                   value_type>::value,
      "value type must be same for buffer and input data");

  hilucsi_assert(b.size() == y.size(), "solution and rhs sizes should match");

  // preparations
  const auto &    prec = *prec_itr;
  const size_type m = prec.m, n = prec.n, nm = n - m;
  const auto &    p = prec.p, &q_inv = prec.q_inv;
  const auto &    s = prec.s, &t = prec.t;

  // first compute the permutated vector (1:m), and store it to work(1:m)
  for (size_type i = 0u; i < m; ++i) work[i] = s[p[i]] * b[p[i]];

  // compute the E part only if E is not empty
  if (nm) {
    // solve for work(1:m)=inv(U)*inv(B)*inv(L)*work(1:m), this is done inplace
    internal::prec_solve_udl_inv(prec.U_B, prec.d_B, prec.L_B, work);
    // then compute y(m+1:n) = E*work(1:m)
    prec.E.mv_nt_low(&work[0], y.data() + m);
    // then subtract b from the y(m+1:n)
    for (size_type i = m; i < n; ++i) y[i] = s[p[i]] * b[p[i]] - y[i];
  }

  if (prec.is_last_level()) {
    if (nm) {
      // create an array wrapper with size of n-m, of y(m+1:n)
      auto y_mn = array_type(nm, y.data() + m, WRAP);
      if (prec.sparse_solver.empty())
        prec.dense_solver.solve(y_mn);
      else {
        prec.sparse_solver.solve(y_mn);
        hilucsi_error_if(prec.sparse_solver.info(), "%s returned error %d",
                         prec.sparse_solver.backend(),
                         prec.sparse_solver.info());
      }
    }
  } else {
    auto        y_mn      = array_type(nm, y.data() + m, WRAP);
    value_type *work_next = &work[n];  // advance to next buffer region
    auto        work_b    = array_type(nm, &work[0] + m, WRAP);
    std::copy(y_mn.cbegin(), y_mn.cend(), work_b.begin());
    // rec call, note that y_mn should store the solution
    prec_solve(++prec_itr, work_b, y_mn, work_next);
  }

  // copy y(m+1:n) to work(m+1:n)
  std::copy(y.cbegin() + m, y.cend(), &work[0] + m);

  // only handle the F part if it's not empty
  if (prec.F.ncols()) {
    // compute F*y(m+1:n) and store it to work(1:m)
    prec.F.mv_nt_low(y.data() + m, &work[0]);
    // subtract b(1:m) from work(1:m)
    for (size_type i = 0u; i < m; ++i) work[i] = s[p[i]] * b[p[i]] - work[i];
  } else if (nm) {
    // should not happen for square systems
    for (size_type i = 0u; i < m; ++i) work[i] = s[p[i]] * b[p[i]];
  }

  // solve for work(1:m)=inv(U)*inv(B)*inv(L)*work(1:m), inplace
  internal::prec_solve_udl_inv(prec.U_B, prec.d_B, prec.L_B, work);

  // Now, we have work(1:n) storing the complete solution before final scaling
  // and permutation

  for (size_type i = 0u; i < n; ++i) y[i] = t[i] * work[q_inv[i]];
}

/// \brief compute the \b safe buffer size for \ref prec_solve
/// \tparam PrecItr this is std::list<Prec>::const_iterator
/// \param[in] first begin iterator
/// \param[in] last end iterator
/// \return size of buffer one should allocate for the work space
/// \sa prec_solve
template <class PrecItr>
inline std::size_t compute_prec_work_space(PrecItr first, PrecItr last) {
  if (first == last) return 0u;
  std::size_t n(0u);
  for (; first != last; ++first) n += first->n;
  return n;
}

/*!
 * @}
 */

}  // namespace hilucsi

#endif  // _HILUCSI_ALG_PRECSOLVE_HPP