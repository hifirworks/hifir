//@HEADER
//----------------------------------------------------------------------------
//                Copyright (C) 2019 The PSMILU AUTHORS
//----------------------------------------------------------------------------
//@HEADER

/// \file psmilu_Crout.hpp
/// \brief Implementation of modified \a Crout update
/// \authors Qiao,

#ifndef _PSMILU_CROUT_HPP
#define _PSMILU_CROUT_HPP

#include <cstddef>
#include <type_traits>

#include "psmilu_log.hpp"
#include "psmilu_utils.hpp"

namespace psmilu {

/// \class Crout
/// \brief Crout update
/// \ingroup alg
class Crout {
 public:
  typedef std::size_t size_type;  ///< size

  /// \brief default constructor
  Crout() : _step(0) {}

  // default copy and assign
  Crout(const Crout &) = default;
  Crout &operator=(const Crout &) = default;

  /// \brief assign to a step
  /// \param[in] step step number
  const Crout &operator=(const size_type step) const {
    _step = step;
    return *this;
  }

  /// \brief increment Crout step
  inline const Crout &operator++() const {
    ++_step;
    return *this;
  }

  /// \brief increment Crout step, suffix
  inline Crout operator++(int) const {
    const Crout tmp(*this);
    ++_step;
    return tmp;
  }

  /// \brief implicitly casting to size_type
  inline operator size_type() const { return _step; }

  template <class A_CcsType, class PermType, class L_CcsType, class L_StartType,
            class DiagType, class U_AugCrsType, class SpVecType>
  void compute_l(const A_CcsType &ccs_A, const PermType &p, const size_type qk,
                 const L_CcsType &L, const L_StartType &L_start,
                 const DiagType &d, const U_AugCrsType &U, SpVecType &l) const {
    // compilation checking
    static_assert(!A_CcsType::ROW_MAJOR, "input A must be CCS for loading l");
    static_assert(!(A_CcsType::ONE_BASED ^ L_CcsType::ONE_BASED),
                  "ionconsistent one-based");
    static_assert(!(A_CcsType::ONE_BASED ^ U_AugCrsType::ONE_BASED),
                  "ionconsistent one-based");
    using index_type           = typename U_AugCrsType::index_type;
    constexpr static bool base = A_CcsType::ONE_BASED;

    //------------
    // run time
    //------------

    // first load the A
    _load_A2l(ccs_A, p, qk, l);
    // if not the first step
    // compute -L(k+1:n,1:k-1)d(1:k-1,1:k-1)U(1:k-1,k)
    if (_step) {
      // get the leading value and index iterators for L
      auto L_v_first = L.vals().cbegin();
      auto L_i_first = L.row_ind().cbegin();
      // get the leading value pos for U
      auto       U_v_first = U.vals().cbegin();
      index_type aug_id    = U.start_col_id(_step);  // get starting column ID
      while (!U.is_nil(aug_id)) {
        // get the row index (C-based)
        // NOTE the row index is that of the column in L
        const index_type r_idx = U.row_idx(aug_id);
        psmilu_assert((size_type)r_idx < d.size(),
                      "%zd exceeds the diagonal vector size", (size_type)r_idx);
        psmilu_assert((size_type)r_idx < L_start.size(),
                      "%zd exceeds the L_start size", (size_type)r_idx);
        // get the value position
        // NOTE this needs to be changed when we discard val_pos impl (U_start)
        // compute D*U first for this entry
        const auto du = d[r_idx] * *(U_v_first + U.val_pos_idx(aug_id));
        // get the starting position from L_start
        auto L_v_itr  = L_v_first + L_start[r_idx];
        auto L_i_itr  = L_i_first + L_start[r_idx],
             L_i_last = L.row_ind_cend(r_idx);
        // for this local range
        for (; L_i_itr != L_i_last; ++L_i_itr, ++L_v_itr) {
          // convert to c index
          const auto c_idx = to_c_idx<size_type, base>(*L_i_itr);
          // compute this entry, if index does not exist, assign new value to
          // -L*d*u, else, -= L*d*u
          l.push_back(*L_i_itr, _step) ? l.vals()[c_idx] = -*L_v_itr * du
                                       : l.vals()[c_idx] -= *L_v_itr * du;
        }
        // advance augmented handle
        aug_id = U.next_col_id(aug_id);
      }  // while
    }
    // scale the inverse of diagonal
    // TODO need to check singularity
    const auto      dk_inv = 1. / d[_step];
    const size_type n      = l.size();
    auto &          vals   = l.vals();
    for (size_type i = 0u; i < n; ++i) vals[l.c_idx(i)] *= dk_inv;
  }

  template <bool IsSymm, class LeftDiagType, class A_CrsType,
            class RightDiagType, class PermType, class L_augCcsType,
            class DiagType, class U_CrsType, class U_StartType, class SpVecType>
  void compute_ut(const LeftDiagType &s, const A_CrsType &crs_A,
                  const RightDiagType &t, const PermType &q, const size_type pk,
                  const size_type m, const L_augCcsType &L, const DiagType &d,
                  const U_CrsType &U, const U_StartType &U_start,
                  SpVecType &ut) const {
    // compilation checking
    static_assert(A_CrsType::ROW_MAJOR, "input A must be CRS for loading ut");
    static_assert(!(A_CrsType::ONE_BASED ^ L_augCcsType::ONE_BASED),
                  "ionconsistent one-based");
    static_assert(!(A_CrsType::ONE_BASED ^ U_CrsType::ONE_BASED),
                  "ionconsistent one-based");
    using index_type           = typename L_augCcsType::index_type;
    constexpr static bool base = A_CrsType::ONE_BASED;

    //------------
    // run time
    //------------

    // first load the A
    _load_A2ut<IsSymm>(s, crs_A, t, q, pk, m, ut);
    // if not the first step
    // compute -L(k,1:k-1)d(1:k-1,1:k-1)U(1:k-1,k+1:n)
    if (_step) {
      // get leading value and index iterators for U
      auto U_v_first = U.vals().cbegin();
      auto U_i_first = U.col_ind().cbegin();
      // get the leading value pos for L
      auto L_v_first = L.vals().cbegin();
      // get starting row ID
      index_type aug_id = L.start_row_id(_step);
      while (!L.is_nil(aug_id)) {
        // get the column index (C-based)
        // NOTE the column index is that of the row in U
        const index_type c_idx = L.col_idx(aug_id);
        psmilu_assert((size_type)c_idx < d.size(),
                      "%zd exceeds the diagonal vector size", (size_type)c_idx);
        psmilu_assert((size_type)c_idx < U_start.size(),
                      "%zd exceeds the U_start size", (size_type)c_idx);
        // NOTE once we drop val-pos impl, change this accordingly (L_start)
        // compute L*D
        const auto ld = *(L_v_first + L.val_pos_idx(aug_id)) * d[c_idx];
        // get the starting position from U_start
        auto U_v_itr  = U_v_first + U_start[c_idx];
        auto U_i_itr  = U_i_first + U_start[c_idx],
             U_i_last = U.col_ind_cend(c_idx);
        // for this local range
        for (; U_i_itr != U_i_last; ++U_i_itr, ++U_v_itr) {
          // convert to c index
          const auto c_idx = to_c_idx<size_type, base>(*U_i_itr);
          ut.push_back(*U_i_itr, _step) ? ut.vals()[c_idx] = -ld * *U_v_itr
                                        : ut.vals()[c_idx] -= ld * *U_v_itr;
        }
        // advance augmented handle
        aug_id = L.next_row_id(aug_id);
      }  // while
    }
    // scale the inverse of dk
    // TODO need to check singularity
    const auto      dk_inv = 1. / d[_step];
    const size_type n      = ut.size();
    auto &          vals   = ut.vals();
    for (size_type i = 0u; i < n; ++i) vals[ut.c_idx(i)] *= dk_inv;
  }

  template <class L_augCcsType, class L_StartType>
  inline void update_L_start(const L_augCcsType &L,
                             L_StartType &       L_start) const {
    static_assert(!L_augCcsType::ROW_MAJOR, "L must be AugCCS");
    using index_type = typename L_augCcsType::index_type;

    // NOTE this routine should be called at the beginning of the crout update
    // this is easy to handle
    if (!_step) return;
    // we just added a new column to L, which is _step-1
    // we, then, assign the starting ID should be its start_ind
    L_start[_step - 1] = L.col_start()[_step - 1];
    // get the aug handle
    index_type aug_id = L.start_row_id(_step);
    // loop through current row, O(l_k')
    while (!L.is_nil(aug_id)) {
      // get the column index, C based
      const index_type c_idx = L.col_idx(aug_id);
      // for each of this starting inds, advance one
      if (L_start[c_idx] < L.col_start()[c_idx + 1]) ++L_start[c_idx];
      // advance augmented handle
      aug_id = L.next_row_id(aug_id);
    }
  }

  template <bool IsSymm, class U_augCrsType, class U_StartType>
  inline void update_U_start(const U_augCrsType &U, const size_type m,
                             U_StartType &U_start) const {
    static_assert(U_augCrsType::ROW_MAJOR, "U must be AugCRS");
    using index_type = typename U_augCrsType::index_type;

    // NOTE this routine should be called at the beginning of Crout update
    if (!_step) return;
    // We need to handle the previous (just added) row
    U_start[_step - 1] = U.row_start()[_step - 1];
    // get the aug handle
    const size_type start  = IsSymm ? m : _step;
    index_type      aug_id = U.start_col_id(start);
    // loop through current column, O(u_{k,m})
    // NOTE that since m is dynamic, we must start from beginning even for symm
    // case
    while (!U.is_nil(aug_id)) {
      // get the row index, C based
      const index_type c_idx = U.row_idx(aug_id);
      U_start[c_idx]         = U.val_pos_idx(aug_id);
      if (U_start[c_idx] < U.row_start()[c_idx + 1]) ++U_start[c_idx];
      // advance augmented handle
      aug_id = U.next_col_id(aug_id);
    }
  }

  template <bool IsSymm, class SpVecType, class DiagType>
  inline typename std::enable_if<!IsSymm>::type update_B_diag(
      const SpVecType &l, const SpVecType &ut, const size_type m,
      DiagType &d) const {
    // get the current diagonal entry
    const auto dk = d[_step];
    // NOTE since l and ut are in general unsorted, we need to loop through
    // one of them and determine the existance of the other entry, this can
    // be done by utilizing the dense tags in sparse vector
    const auto &    l_d_tags = l.dense_tags();
    const size_type n        = ut.size();
    // NOTE we choose to loop out ut, cuz in general length(ut) <= length(l)
    // otherwise, another checking is needed for index bound
    for (size_type i = 0u; i < n; ++i) {
      const auto c_idx = ut.c_idx(i);
      psmilu_assert(
          (size_type)c_idx >= _step,
          "should only contain the upper part of ut, (c_idx,step)=(%zd,%zd)!",
          (size_type)c_idx, _step);
      // if the dense tags of this entry points to this step, we know l has
      // an element in this slot
      if (c_idx < m && (size_type)l_d_tags[c_idx] == _step)
        d[c_idx] -= dk * ut.val(i) * l.vals()[c_idx];
    }
  }

  template <bool IsSymm, class SpVecType, class DiagType>
  inline typename std::enable_if<IsSymm>::type update_B_diag(
      const SpVecType &l, const SpVecType & /* ut */, const size_type m,
      DiagType &d) const {
    psmilu_assert(m, "fatal, symmetric block cannot be empty!");
    // get the current diagonal entry
    const auto dk = d[_step];
    // NOTE that the diagonal is updated before doing any dropping/sorting,
    // thus the indices are, in general, unsorted
    const size_type n = l.size();
    // get the c index
    for (size_type i = 0u; i < n; ++i) {
      const auto c_idx = l.c_idx(i);
      psmilu_assert(
          (size_type)c_idx >= _step,
          "should only contain the lower part of l, (c_idx,step)=(%zd,%zd)!",
          (size_type)c_idx, _step);
      if (c_idx < m) d[c_idx] -= dk * l.val(i) * l.val(i);
    }
  }

 protected:
  /// \brief load A column to l buffer
  /// \tparam CcsType ccs matrix of input A, see \ref CCS
  /// \tparam PermType permutation vector type, see \ref BiPermMatrix
  /// \tparam SpVecType sparse vector type, see \ref SparseVector
  /// \param[in] ccs_A input matrix in CCS scheme
  /// \param[in] p left-hand side row permutation matrix
  /// \param[in] qk column index (C-based) in permutated matrix
  /// \param[out] l output sparse vector of column vector for L
  /// \note Complexity is \f$\mathcal{O}(nnz(\boldsymbol{A}(:,qk)))\f$
  template <class CcsType, class PermType, class SpVecType>
  inline void _load_A2l(const CcsType &ccs_A, const PermType &p,
                        const size_type qk, SpVecType &l) const {
    // compilation consistency checking
    static_assert(!(CcsType::ONE_BASED ^ SpVecType::ONE_BASED),
                  "inconsistent one-based in ccs and sparse vector");
    constexpr static bool base = CcsType::ONE_BASED;
    // l should be empty
    psmilu_assert(l.empty(), "l should be empty while loading A");
    // qk is c index
    auto v_itr = ccs_A.val_cbegin(qk);
    auto i_itr = ccs_A.row_ind_cbegin(qk);
    for (auto last = ccs_A.row_ind_cend(qk); i_itr != last; ++i_itr, ++v_itr) {
      const auto c_idx = p.inv(to_c_idx<size_type, base>(*i_itr));
      // push to the sparse vector only if its in range _step+1:n
      if ((size_type)c_idx > _step) {
#ifndef NDEBUG
        const bool val_must_not_exit =
#endif
            l.push_back(to_ori_idx<size_type, base>(c_idx), _step);
        psmilu_assert(val_must_not_exit,
                      "see prefix, failed on Crout step %zd for l", _step);
        l.vals()[c_idx] = *v_itr;
      }
    }
  }

  /// \brief load A row to ut buffer
  /// \tparam IsSymm if \a true, then only load the offset
  /// \tparam LeftDiagType diagonal matrix from left-hand side
  /// \tparam CrsType crs matrix of input A, see \ref CRS
  /// \tparam RightDiagType diagonal matrix from right-hand side
  /// \tparam PermType permutation vector type, see \ref BiPermMatrix
  /// \tparam SpVecType sparse vector type, see \ref SparseVector
  /// \param[in] s diagonal matrix scaling from left-hand side
  /// \param[in] crs_A input matrix in CRS scheme
  /// \param[in] t diagonal matrix scaling from right-hand side
  /// \param[in] q right-hand side column permutation matrix
  /// \param[in] pk row index
  /// \param[in] m leading size
  /// \param[out] ut output sparse vector of row vector for U
  /// \note Complexity is \f$\mathcol{O}(nnz(\boldsymbol{A}(pk,:)))\f$
  template <bool IsSymm, class LeftDiagType, class CrsType, class RightDiagType,
            class PermType, class SpVecType>
  inline void _load_A2ut(const LeftDiagType &s, const CrsType &crs_A,
                         const RightDiagType &t, const PermType &q,
                         const size_type pk, const size_type m,
                         SpVecType &ut) const {
    // compilation consistency checking
    static_assert(!(CrsType::ONE_BASED ^ SpVecType::ONE_BASED),
                  "inconsistent one-based in ccs and sparse vector");
    constexpr static bool base = CrsType::ONE_BASED;
    // ut should be empty
    psmilu_assert(ut.empty(), "ut should be empty while loading A");
    // note m == 0 should be checked b4 calling this routine
    const size_type thres = IsSymm ? m - 1u : _step;
    // pk is c index
    auto       v_itr = crs_A.val_cbegin(pk);
    auto       i_itr = crs_A.col_ind_cbegin(pk);
    const auto s_pk  = s[pk];
    for (auto last = crs_A.col_ind_cend(pk); i_itr != last; ++i_itr, ++v_itr) {
      const auto c_idx = q.inv(to_c_idx<size_type, base>(*i_itr));
      if ((size_type)c_idx > thres) {
#ifndef NDEBUG
        const bool val_must_not_exit =
#endif
            ut.push_back(to_ori_idx<size_type, base>(c_idx), _step);
        psmilu_assert(val_must_not_exit,
                      "see prefix, failed on Crout step %zd for ut", _step);
        // null statement for opt and getting rid of unused warning
        (void)val_must_not_exit;
        ut.vals()[c_idx] = s_pk * *v_itr * t[c_idx];  // scale here
      }
    }
  }

 protected:
  mutable size_type _step;  ///< current step
};
}  // namespace psmilu

#endif  // _PSMILU_CROUT_HPP
