//@HEADER
//----------------------------------------------------------------------------
//                Copyright (C) 2019 The PSMILU AUTHORS
//----------------------------------------------------------------------------
//@HEADER

/// \file psmilu_diag_pivot.hpp
/// \brief Condition number estimation used in for diagonal pivoting
/// \authors Qiao,

#ifndef _PSMILU_DIAGPIVOTCOND_HPP
#define _PSMILU_DIAGPIVOTCOND_HPP

#include <algorithm>
#include <type_traits>

#include "psmilu_log.hpp"
#include "psmilu_utils.hpp"

namespace psmilu {

/// \brief update the kappa l at step
/// \tparam L_AugCcsType augmented ccs for L, see \ref AugCCS
/// \tparam KappaL_Type array used to store kappa, see \ref Array
/// \param[in] step current Crout step
/// \param[in] L lower part at current step
/// \param[in,out] kappa_l previous solutions in and current solution out
/// \return if \a true, then the current rhs is one; ow, it's negative one
/// \ingroup alg
/// \note Recall that the lower part is unit diagonal without explicitly
///       storing diagonal entries
///
/// This routine is to bound the \f$\infty\f$-norm of
/// \f$\left\vert\boldsymbol{L}^{-1}\right\vert\f$ by using a greedy strategy,
/// i.e. bounding the
/// \f$\left\vert\boldsymbol{L}^{-1}\boldsymbol{c}\right\vert\f$, where
/// \f$c_i=\pm 1\f$ for \f$i\f$ in \f$1,...,n\f$, s.t.
/// \f$\left|\boldsymbol{e}_i^T\boldsymbol{L}^{-1}\boldsymbol{c}\right|\f$ is
/// maximized. Here, to achieve linear time complexity, we maintain an array
/// that stores all previous solutions, i.e. \f$\kappa_l\f$, and use it to
/// update the current one.
///
/// Complexity is linear, i.e.
/// \f$\mathcal{O}(\textrm{nnz}(\boldsymbol{L}_{k,:}))\f$
template <class L_AugCcsType, class KappaL_Type>
inline bool update_kappa_l(const typename L_AugCcsType::size_type step,
                           const L_AugCcsType &L, KappaL_Type &kappa_l) {
  static_assert(!L_AugCcsType::ROW_MAJOR, "must be column major storage");
  using value_type          = typename L_AugCcsType::value_type;
  using index_type          = typename L_AugCcsType::index_type;
  using size_type           = typename L_AugCcsType::size_type;
  constexpr static bool ONE = true, NEG_ONE = false;
  if (!step) {
    // for the first step, we set the rhs (c) to be 1 and let the solution
    // to be 1
    kappa_l[0] = 1;
    return ONE;
  }
  // we need to loop thru all entries in row step
  value_type sum(0);

  // start augment id
  index_type aug_id = L.start_row_id(step);
  while (!L.is_nil(aug_id)) {
    const index_type col_idx = L.col_idx(aug_id);
    psmilu_assert((size_type)col_idx < kappa_l.size(),
                  "%zd exceeds the solution size", (size_type)col_idx);
    psmilu_assert((size_type)col_idx < step,
                  "the matrix L should only contain the strict lower case");
    sum += kappa_l[col_idx] * L.val_from_row_id(aug_id);
    // advance to next augment handle
    aug_id = L.next_row_id(aug_id);
  }
  const value_type k1 = 1. - sum, k2 = -1. - sum;
  if (std::abs(k1) < std::abs(k2)) {
    kappa_l[step] = k2;
    return NEG_ONE;
  }
  kappa_l[step] = k1;
  return ONE;
}

/// \brief update the kappa for U transpose inverse
/// \tparam IsSymm if \a true, then assume a leading symmetric block
/// \tparam U_AugCrsType augmented data structure for U, see \ref AugCRS
/// \tparam KappaL_Type array type used for storing kappa l, see \ref Array
/// \tparam KappaU_Type array type used for storing kappa u, see \ref Array
/// \tparam T dummy type (\a bool) for \a SFINAE
/// \param[in] step current Crout step
/// \param[in] kappa_l kappa for lower part estimation
/// \param[out] kappa_u kappa vector for u
/// \return to keep the API consistent, we return \a true here
/// \ingroup alg
///
/// Notice that this routine is \a SFINAE-able by \a IsSymm, and this is for
/// the \a true case, i.e. symmetric leading block. In this case, the solution
/// vector of \f$\boldsymbol{U}^T\f$ is the same as that of \f$\boldsymbol{L}\f$
template <bool IsSymm, class U_AugCrsType, class KappaL_Type, class KappaU_Type,
          typename T = bool>
inline typename std::enable_if<IsSymm, T>::type update_kappa_ut(
    const typename U_AugCrsType::size_type step, const U_AugCrsType & /* U */,
    const KappaL_Type &kappa_l, KappaU_Type &kappa_u) {
  // symmetric case
  kappa_u[step] = kappa_l[step];
  return true;
}

/// \brief update kappa u for general case
/// \tparam IsSymm if \a true, then assume a leading symmetric block
/// \tparam U_AugCrsType augmented crs type for U, see \ref AugCRS
/// \tparam KappaL_Type array type used for storing kappa l, see \ref Array
/// \tparam KappaU_Type array type used for storing kappa u, see \ref Array
/// \tparam T dummy type (\a bool) for \a SFINAE
/// \param[in] step current Crout step
/// \param[in] U upper part
/// \param[in,out] kappa_u older slutions in and new solution out
/// \return if \a true, then the current rhs is one; ow, it's negative one
/// \ingroup alg
///
/// Notice that this routine is \a SFINAE-able by parameter \a IsSymm, and this
/// case is for \a IsSymm is \a false, i.e. asymmetric leading blocks.
/// This routine is to bound the \f$\infty\f$-norm of
/// \f$\left\vert\boldsymbol{U}^{-T}\right\vert\f$ by using a greedy strategy,
/// i.e. bounding the
/// \f$\left\vert\boldsymbol{U}^{-T}\boldsymbol{c}\right\vert\f$, where
/// \f$c_i=\pm 1\f$ for \f$i\f$ in \f$1,...,n\f$, s.t.
/// \f$\left|\boldsymbol{e}_i^T\boldsymbol{U}^{-T}\boldsymbol{c}\right|\f$ is
/// maximized. Here, to achieve linear time complexity, we maintain an array
/// that stores all previous solutions, i.e. \f$\kappa_u\f$, and use it to
/// update the current one.
///
/// Complexity is linear, i.e.
/// \f$\mathcal{O}(\textrm{nnz}(\boldsymbol{U}_{:,k}))\f$
template <bool IsSymm, class U_AugCrsType, class KappaL_Type, class KappaU_Type,
          typename T = bool>
inline typename std::enable_if<!IsSymm, T>::type update_kappa_ut(
    const typename U_AugCrsType::size_type step, const U_AugCrsType &U,
    const KappaL_Type & /* kappa_l */, KappaU_Type &                 kappa_u) {
  static_assert(U_AugCrsType::ROW_MAJOR, "must be row major storage");
  using value_type          = typename U_AugCrsType::value_type;
  using index_type          = typename U_AugCrsType::index_type;
  using size_type           = typename U_AugCrsType::size_type;
  constexpr static bool ONE = true, NEG_ONE = false;
  if (!step) {
    // for first step, just assign one and return one
    kappa_u[0] = 1;
    return ONE;
  }

  // we need to loop thru all entries in row step
  value_type sum(0);

  // start augment id
  index_type aug_id = U.start_col_id(step);
  while (!U.is_nil(aug_id)) {
    const index_type row_idx = U.row_idx(aug_id);
    psmilu_assert((size_type)row_idx < kappa_u.size(),
                  "%zd exceeds the solution size", (size_type)row_idx);
    psmilu_assert((size_type)row_idx < step,
                  "the matrix U should only contain the strict upper case");
    sum += kappa_u[row_idx] * U.val_from_col_id(aug_id);
    // advance augment handle
    aug_id = U.next_col_id(aug_id);
  }
  const value_type k1 = 1. - sum, k2 = -1. - sum;
  if (std::abs(k1) < std::abs(k2)) {
    kappa_u[step] = k2;
    return NEG_ONE;
  }
  kappa_u[step] = k1;
  return ONE;
}
}  // namespace psmilu

#endif  // _PSMILU_DIAGPIVOTCOND_HPP
