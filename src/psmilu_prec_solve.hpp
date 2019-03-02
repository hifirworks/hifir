//@HEADER
//----------------------------------------------------------------------------
//                Copyright (C) 2019 The PSMILU AUTHORS
//----------------------------------------------------------------------------
//@HEADER

/// \file psmilu_prec_solve.hpp
/// \brief Multilevel preconditioner solver
/// \authors Qiao,

#ifndef _PSMILU_PRECSOLVE_HPP
#define _PSMILU_PRECSOLVE_HPP

// use generic programming, implicitly assume the API in Prec

#include <algorithm>
#include <iterator>
#include <list>
#include <type_traits>

#include "psmilu_Array.hpp"
#include "psmilu_utils.hpp"

namespace psmilu {
namespace internal {

/// \brief triangular solving kernel in \a psmilu_solve algorithm
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
template <class CcsType, class DiagType, class RhsType>
inline void prec_solve_udl_inv(const CcsType &U, const DiagType &d,
                               const CcsType &L, RhsType &y) {
  using size_type                 = typename CcsType::size_type;
  constexpr static bool ONE_BASED = CcsType::ONE_BASED;
  static_assert(!CcsType::ROW_MAJOR, "must be ccs");
  const auto c_idx = [](const size_type i) {
    return to_c_idx<size_type, ONE_BASED>(i);
  };
  using rev_iterator   = std::reverse_iterator<decltype(U.row_ind_cbegin(0))>;
  using rev_v_iterator = std::reverse_iterator<decltype(U.val_cbegin(0))>;

  const size_type m = U.nrows();
  psmilu_assert(U.ncols() == m, "U must be squared");
  psmilu_assert(L.nrows() == L.ncols(), "L must be squared");
  psmilu_assert(d.size() >= m, "diagonal must be no smaller than system size");

  // y=inv(L)*y
  for (size_type j = 0u; j < m; ++j) {
    const auto y_j = y[j];
    auto       itr = L.row_ind_cbegin(j);
#ifndef NDEBUG
    if (itr != L.row_ind_cend(j))
      psmilu_error_if(c_idx(*itr) <= j, "must be strictly lower part!");
#endif
    auto v_itr = L.val_cbegin(j);
    for (auto last = L.row_ind_cend(j); itr != last; ++itr, ++v_itr) {
      const auto i = c_idx(*itr);
      psmilu_assert(i < m, "%zd exceeds system size %zd", i, m);
      y[i] -= *v_itr * y_j;
    }
  }

  // y=inv(D)*y
  for (size_type i = 0u; i < m; ++i) y[i] /= d[i];

  // y = inv(U)*y, NOTE since U is unit diagonal, no need to handle j == 0
  for (size_type j = m - 1; j != 0u; --j) {
    const auto y_j = y[j];
    auto       itr = rev_iterator(U.row_ind_cend(j));
#ifndef NDEBUG
    if (itr != rev_iterator(U.row_ind_cbegin(j)))
      psmilu_error_if(c_idx(*itr) >= j, "must be strictly upper part");
#endif
    auto v_itr = rev_v_iterator(U.val_cend(j));
    for (auto last = rev_iterator(U.row_ind_cbegin(j)); itr != last;
         ++itr, ++v_itr)
      y[c_idx(*itr)] -= *v_itr * y_j;
  }
}

}  // namespace internal

/// \brief Given multilevel preconditioner and rhs, solve the solution
/// \tparam PrecItr this is std::list<Prec>::const_iterator
/// \tparam WorkType buffer type, generic array interface, i.e. operator[]
/// \param[in] prec_itr current iterator of a single level preconditioner
/// \param[in] b right-hand side vector
/// \param[out] y solution vector
/// \param[out] work work space
/// \ingroup slv
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

  psmilu_assert(b.size() == y.size(), "solution and rhs sizes should match");

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
      prec.dense_solver.solve(y_mn);  // solve inplace!
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
/// \ingroup slv
template <class PrecItr>
inline std::size_t compute_prec_work_space(PrecItr first, PrecItr last) {
  if (first == last) return 0u;
  std::size_t n(0u);
  for (; first != last; ++first) n += first->n;
  return n;
}

}  // namespace psmilu

#endif  // _PSMILU_PRECSOLVE_HPP