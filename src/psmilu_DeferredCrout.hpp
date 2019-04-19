//@HEADER
//----------------------------------------------------------------------------
//                Copyright (C) 2019 The PSMILU AUTHORS
//----------------------------------------------------------------------------
//@HEADER

/// \file psmilu_DeferredCrout.hpp
/// \brief Implementation of modified \a Crout update in deferred fashion
/// \authors Qiao,

#ifndef _PSMILU_DEFERRED_CROUT_HPP
#define _PSMILU_DEFERRED_CROUT_HPP

#include "psmilu_Crout.hpp"

namespace psmilu {

/// \class DeferredCrout
/// \brief Crout update in deferred fashion
/// \ingroup crout
class DeferredCrout : public Crout {
 public:
  using size_type = Crout::size_type;  ///< size type

  /// \brief default constructor
  DeferredCrout() : Crout(), _defers(0) {}

  // default copy and assign
  DeferredCrout(const DeferredCrout &) = default;
  DeferredCrout &operator=(const DeferredCrout &) = default;

  /// \brief get the current defers
  inline size_type defers() const { return _defers; }

  /// \brief get the deferred step
  inline size_type deferred_step() const { return _defers + _step; }

  /// \brief increment the defer counter
  inline void increment_defer_counter() { ++_defers; }

  template <class ScaleArray, class CrsType, class PermType, class Ori2Def,
            class U_CrsType, class U_StartType, class DiagType,
            class L_AugCcsType, class SpVecType>
  void compute_ut(const ScaleArray &s, const CrsType &crs_A,
                  const ScaleArray &t, const size_type pk, const PermType &q,
                  const Ori2Def &ori2def, const L_AugCcsType &L,
                  const DiagType &d, const U_CrsType &U,
                  const U_StartType &U_start, SpVecType &ut) const {
    // compilation checking
    static_assert(CrsType::ROW_MAJOR, "input A must be CRS for loading ut");
    static_assert(!(CrsType::ONE_BASED ^ L_AugCcsType::ONE_BASED),
                  "ionconsistent one-based");
    static_assert(!(CrsType::ONE_BASED ^ U_CrsType::ONE_BASED),
                  "ionconsistent one-based");
    using index_type           = typename L_AugCcsType::index_type;
    constexpr static bool base = CrsType::ONE_BASED;

    // reset sparse buffer
    ut.reset_counter();

    // first load the A row
    _load_A2ut(s, crs_A, t, pk, q, ori2def, ut);

    // if not first step
    if (_step) {
      // get the leading value and index iterators for U
      auto U_v_first = U.vals().cbegin();
      auto U_i_first = U.col_ind().cbegin();
      // get the starting row ID with deferring
      index_type aug_id = L.start_row_id(_step);
      while (!L.is_nil(aug_id)) {
        // get the column index
        const size_type col_idx = L.col_idx(aug_id);
        psmilu_assert(
            col_idx < _step,
            "compute_ut column index %zd should not exceed step %zd for L",
            col_idx, _step);
        psmilu_assert(col_idx < U_start.size(), "%zd exceeds the U_start size",
                      col_idx);
        // compute L*d
        const auto ld = L.val_from_row_id(aug_id) * d[col_idx];
        // get the starting position from U_start
        auto U_v_itr = U_v_first + U_start[col_idx];
        auto U_i_itr = U_i_first + U_start[col_idx];
#ifndef NDEBUG
        if (U_i_itr != U.col_ind_cbegin(col_idx)) {
          const auto prev_idx = to_c_idx<size_type, base>(*(U_i_itr - 1));
          psmilu_error_if(prev_idx > _step, "U_start error!");
        }
#endif
        // for loop to go thru all entries in U
        for (auto U_i_last = U.col_ind_cend(col_idx); U_i_itr != U_i_last;
             ++U_i_itr, ++U_v_itr) {
          // convert to c index
          const auto c_idx = to_c_idx<size_type, base>(*U_i_itr);
          psmilu_assert(c_idx > deferred_step(),
                        "U index %zd in computing ut should greater than step "
                        "%zd(defers:%zd)",
                        c_idx, _step, _defers);
          if (ut.push_back(*U_i_itr, _step))
            ut.vals()[c_idx] = -ld * *U_v_itr;
          else
            ut.vals()[c_idx] -= ld * *U_v_itr;
        }
        // advance augmented handle
        aug_id = L.next_row_id(aug_id);
      }  // while
    }
  }

  template <bool IsSymm, class ScaleArray, class CcsType, class PermType,
            class Ori2Def, class L_CcsType, class L_StartType, class DiagType,
            class U_AugCrsType, class SpVecType>
  void compute_l(const ScaleArray &s, const CcsType &ccs_A, const ScaleArray &t,
                 const PermType &p, const size_type qk, const size_type m,
                 const Ori2Def &ori2def, const L_CcsType &L,
                 const L_StartType &L_start, const DiagType &d,
                 const U_AugCrsType &U, SpVecType &l) const {
    // compilation checking
    static_assert(!CcsType::ROW_MAJOR, "input A must be CCS for loading l");
    static_assert(!(CcsType::ONE_BASED ^ L_CcsType::ONE_BASED),
                  "ionconsistent one-based");
    static_assert(!(CcsType::ONE_BASED ^ U_AugCrsType::ONE_BASED),
                  "ionconsistent one-based");
    using index_type           = typename U_AugCrsType::index_type;
    constexpr static bool base = CcsType::ONE_BASED;

    // clear sparse counter
    l.reset_counter();

    // load A column
    _load_A2l<IsSymm>(s, ccs_A, t, p, qk, m, ori2def, l);

    // if not first step
    if (_step) {
      // get the leading value and index iterators for L
      auto L_v_first = L.vals().cbegin();
      auto L_i_first = L.row_ind().cbegin();
      // get the deferred column handle
      index_type aug_id = U.start_col_id(_step);
      while (!U.is_nil(aug_id)) {
        // get the row index
        const size_type r_idx = U.row_idx(aug_id);
        if (!IsSymm)
          psmilu_assert(
              r_idx < _step,
              "compute_ut row index %zd should not exceed step %zd for U",
              r_idx, _step);
        psmilu_assert(r_idx < L_start.size(), "%zd exceeds the L_start size",
                      r_idx);
        // compute d*U
        const auto du = d[r_idx] * U.val_from_col_id(aug_id);
        // get the starting position from L_start
        auto L_v_itr = L_v_first + L_start[r_idx];
        auto L_i_itr = L_i_first + L_start[r_idx];
#ifndef NDEBUG
        if (!IsSymm)
          if (L_i_itr != L.row_ind_cbegin(r_idx)) {
            const auto prev_idx = to_c_idx<size_type, base>(*(L_i_itr - 1));
            psmilu_error_if(prev_idx > _step, "L_start error!");
          }
#endif
        for (auto L_i_last = L.row_ind_cend(r_idx); L_i_itr != L_i_last;
             ++L_i_itr, ++L_v_itr) {
          // convert to c index
          const auto c_idx = to_c_idx<size_type, base>(*L_i_itr);
          psmilu_assert(c_idx > deferred_step(),
                        "L index %zd in computing l should greater than step "
                        "%zd(defers:%zd)",
                        c_idx, _step, _defers);
          // compute this entry, if index does not exist, assign new value to
          // -L*d*u, else, -= L*d*u
          if (l.push_back(*L_i_itr, _step))
            l.vals()[c_idx] = -du * *L_v_itr;
          else
            l.vals()[c_idx] -= *L_v_itr * du;
        }
        // advance augmented handle
        aug_id = U.next_col_id(aug_id);
      }  // while
    }
  }

  template <class ArrayType>
  inline void compress_array(ArrayType &v) const {
    v[_step] = v[deferred_step()];
  }

  template <class ArrayIn, class ArrayOut>
  inline void assign_gap_array(const ArrayIn &r, ArrayOut &l) const {
    l[_step] = r[deferred_step()];
  }

  template <class U_AugCrsType, class U_StartType>
  inline void update_U_start_and_compress_U(U_AugCrsType &U,
                                            U_StartType & U_start) const {
    static_assert(U_AugCrsType::ROW_MAJOR, "U must be AugCRS");
    using index_type                 = typename U_AugCrsType::index_type;
    constexpr static bool ONE_BASED  = U_AugCrsType::ONE_BASED;
    const static auto     comp_index = [](index_type &i, const size_type dfrs
#ifndef NDEBUG
                                      ,
                                      const size_type step
#endif
                                   ) {
      i -= dfrs;
      psmilu_assert(i - ONE_BASED == (index_type)step, "should be step %zd",
                    step);
    };
    const static auto comp_index_dummy = [](index_type &, const size_type

#ifndef NDEBUG
                                            ,
                                            const size_type
#endif
                                         ) {};

    if (!_defers)
      _update_U_start(U, U_start, comp_index_dummy);
    else {
      _update_U_start(U, U_start, comp_index);
      U.col_start()[_step] = U.col_start()[deferred_step()];
      U.col_end()[_step]   = U.col_end()[deferred_step()];
    }
  }

  template <bool IsSymm, class L_AugCcsType, class L_StartType>
  inline void update_L_start_and_compress_L(L_AugCcsType &L, const size_type m,
                                            L_StartType &L_start) const {
    static_assert(!L_AugCcsType::ROW_MAJOR, "L must be AugCCS");
    using index_type                 = typename L_AugCcsType::index_type;
    constexpr static bool ONE_BASED  = L_AugCcsType::ONE_BASED;
    const static auto     comp_index = [](index_type &i, const size_type dfrs
#ifndef NDEBUG
                                      ,
                                      const size_type step
#endif
                                   ) {
      i -= dfrs;
      psmilu_assert(i - ONE_BASED == (index_type)step, "should be step %zd",
                    step);
    };
    const static auto comp_index_dummy = [](index_type &, const size_type

#ifndef NDEBUG
                                            ,
                                            const size_type
#endif
                                         ) {};

    if (IsSymm || !_defers)
      _update_L_start<IsSymm>(L, m, L_start, comp_index_dummy);
    else {
      _update_L_start<IsSymm>(L, m, L_start, comp_index);
      L.row_start()[_step] = L.row_start()[deferred_step()];
      L.row_end()[_step]   = L.row_end()[deferred_step()];
    }
  }

  template <bool IsSymm, class SpVecType, class DiagType>
  inline typename std::enable_if<!IsSymm>::type update_B_diag(
      const SpVecType &l, const SpVecType &ut, const size_type m,
      DiagType &d) const {
    // NOTE that we need the internal dense tag from sparse vector
    // thus, we can either:
    //    1) make this function a friend of SparseVec, or
    //    2) create a caster class
    using extractor = internal::SpVInternalExtractor<SpVecType>;

    // assume l is not scaled by the diagonal

    // get the current diagonal entry
    // const auto dk = d[_step];
    if (ut.size() <= l.size()) {
      const auto &    l_d_tags = static_cast<const extractor &>(l).dense_tags();
      const size_type n        = ut.size();
      const auto &    l_vals   = l.vals();
      for (size_type i = 0u; i < n; ++i) {
        const size_type c_idx = ut.c_idx(i);
        psmilu_assert(c_idx > deferred_step(),
                      "should only contain the upper part of ut, "
                      "(c_idx,step(defer))=(%zd,%zd(%zd))!",
                      c_idx, _step, _defers);
        // if the dense tags of this entry points to this step, we know l has
        // an element in this slot
        if (c_idx < m && (size_type)l_d_tags[c_idx] == _step)
          d[c_idx] -= ut.val(i) * l_vals[c_idx];
      }
    } else {
      const auto &ut_d_tags  = static_cast<const extractor &>(ut).dense_tags();
      const size_type n      = l.size();
      const auto &    u_vals = ut.vals();
      for (size_type i = 0u; i < n; ++i) {
        const size_type c_idx = l.c_idx(i);
        psmilu_assert(c_idx > deferred_step(),
                      "should only contain the lower part of l, "
                      "(c_idx,step(defer))=(%zd,%zd(%zd))!",
                      c_idx, _step, _defers);
        // if the dense tags of this entry points to this step, we know ut has
        // an element in this slot
        if (c_idx < m && (size_type)ut_d_tags[c_idx] == _step)
          d[c_idx] -= u_vals[c_idx] * l.val(i);
      }
    }
  }

  template <bool IsSymm, class SpVecType, class DiagType>
  inline typename std::enable_if<IsSymm>::type update_B_diag(
      const SpVecType & /* l */, const SpVecType &ut, const size_type m,
      DiagType &d) const {
    psmilu_assert(m, "fatal, symmetric block cannot be empty!");
    // get the current diagonal entry
    const auto dk = d[_step];
    // we only need to deal with ut
    const size_type n = ut.size();
    for (size_type i(0); i < n; ++i) {
      const size_type c_idx = ut.c_idx(i);
      psmilu_assert(c_idx > deferred_step(),
                    "ut should only contain the upper part, "
                    "(c_idx,step(defer))=(%zd,%zd(%zd))!",
                    c_idx, _step, _defers);
      if (c_idx < m) d[c_idx] -= dk * ut.val(i) * ut.val(i);
    }
  }

 protected:
  /// \brief load a row of A to ut buffer
  /// \tparam ScaleArray scaling from left/right-hand sides, see \ref Array
  /// \tparam CrsType crs matrix of input A, see \ref CRS
  /// \tparam PermType permutation vector type, see \ref BiPermMatrix
  /// \tparam Ori2Def original to deferred mapping type
  /// \tparam SpVecType sparse vector type, see \ref SparseVector
  /// \param[in] s row scaling vector
  /// \param[in] crs_A input matrix in CRS scheme
  /// \param[in] t column scaling vector
  /// \param[in] pk row permuted index
  /// \param[in] q column permutation matrix
  /// \param[in] ori2def original to deferred system
  /// \param[out] ut output sparse vector of row vector for A
  /// \sa _load_A2l
  template <class ScaleArray, class CrsType, class PermType, class Ori2Def,
            class SpVecType>
  inline void _load_A2ut(const ScaleArray &s, const CrsType &crs_A,
                         const ScaleArray &t, const size_type &pk,
                         const PermType &q, const Ori2Def &ori2def,
                         SpVecType &ut) const {
    // compilation consistency checking
    static_assert(!(CrsType::ONE_BASED ^ SpVecType::ONE_BASED),
                  "inconsistent one-based in ccs and sparse vector");
    constexpr static bool base = CrsType::ONE_BASED;

    // ut should be empty
    psmilu_assert(ut.empty(), "ut should be empty while loading A");
    const size_type defer_thres = deferred_step();
    // pk is c index
    auto       v_itr = crs_A.val_cbegin(pk);
    auto       i_itr = crs_A.col_ind_cbegin(pk);
    const auto s_pk  = s[pk];
    for (auto last = crs_A.col_ind_cend(pk); i_itr != last; ++i_itr, ++v_itr) {
      const auto      A_idx = to_c_idx<size_type, base>(*i_itr);
      const size_type c_idx = ori2def[q.inv(A_idx)];
      if (c_idx > defer_thres) {
        // get the gapped index
#ifndef NDEBUG
        const bool val_must_not_exit =
#endif
            ut.push_back(to_ori_idx<size_type, base>(c_idx), _step);
        psmilu_assert(val_must_not_exit,
                      "see prefix, failed on Crout step %zd for ut", _step);
        ut.vals()[c_idx] = s_pk * *v_itr * t[A_idx];  // scale here
      }
    }
  }

  /// \brief load a column of A to l buffer
  /// \tparam IsSymm if \a true, then only load the offset
  /// \tparam ScaleArray scaling from left/right-hand sides, see \ref Array
  /// \tparam CcsType ccs matrix of input A, see \ref CCS
  /// \tparam PermType permutation vector type, see \ref BiPermMatrix
  /// \tparam Ori2Def original to deferred mapping type
  /// \tparam SpVecType sparse vector type, see \ref SparseVector
  /// \param[in] s row scaling vector
  /// \param[in] ccs_A input matrix in CCS scheme
  /// \param[in] t column scaling vector
  /// \param[in] p row permutation matrix
  /// \param[in] qk permuted column index
  /// \param[in] m leading size
  /// \param[in] ori2def original to deferred system
  /// \param[out] l output sparse vector of column vector for A
  /// \sa _load_A2ut
  template <bool IsSymm, class ScaleArray, class CcsType, class PermType,
            class Ori2Def, class SpVecType>
  inline void _load_A2l(const ScaleArray &s, const CcsType &ccs_A,
                        const ScaleArray &t, const PermType &p,
                        const size_type &qk, const size_type m,
                        const Ori2Def &ori2def, SpVecType &l) const {
    // compilation consistency checking
    static_assert(!(CcsType::ONE_BASED ^ SpVecType::ONE_BASED),
                  "inconsistent one-based in ccs and sparse vector");
    constexpr static bool base = CcsType::ONE_BASED;

    // runtime
    psmilu_assert(l.empty(), "l should be empty while loading A");
    const size_type defer_thres = deferred_step();
    const size_type thres       = IsSymm ? m - 1 : defer_thres;
    // qk is c index
    auto       v_itr = ccs_A.val_cbegin(qk);
    auto       i_itr = ccs_A.row_ind_cbegin(qk);
    const auto t_qk  = t[qk];
    for (auto last = ccs_A.row_ind_cend(qk); i_itr != last; ++i_itr, ++v_itr) {
      const auto      A_idx = to_c_idx<size_type, base>(*i_itr);
      const size_type c_idx = ori2def[p.inv(A_idx)];
      // push to the sparse vector only if its in range _step+1:n
      if (c_idx > thres) {
#ifndef NDEBUG
        const bool val_must_not_exit =
#endif
            l.push_back(to_ori_idx<size_type, base>(c_idx), _step);
        psmilu_assert(val_must_not_exit,
                      "see prefix, failed on Crout step %zd for l", _step);
        l.vals()[c_idx] = s[A_idx] * *v_itr * t_qk;  // scale here
      }
    }
  }

  template <class U_AugCrsType, class U_StartType, class CompIndexKernel>
  inline void _update_U_start(U_AugCrsType &U, U_StartType &U_start,
                              const CompIndexKernel &comp_index) const {
    static_assert(U_AugCrsType::ROW_MAJOR, "U must be AugCRS");
    using index_type                = typename U_AugCrsType::index_type;
    constexpr static bool ONE_BASED = U_AugCrsType::ONE_BASED;
    const auto            c_idx     = [](const size_type i) {
      return to_c_idx<size_type, ONE_BASED>(i);
    };

    if (!_step) return;

    U_start[_step - 1] = U.row_start()[_step - 1] - ONE_BASED;
    // get the aug handle wrt defers
    index_type aug_id = U.start_col_id(deferred_step());
    // go through all non zeros entries
    while (!U.is_nil(aug_id)) {
      const index_type row = U.row_idx(aug_id);
      if (U_start[row] < U.row_start()[row + 1] - ONE_BASED &&
          c_idx(*(U.col_ind().cbegin() + U_start[row])) <= deferred_step())
        ++U_start[row];
      comp_index(*(U.col_ind().begin() + U.val_pos_idx(aug_id)), _defers
#ifndef NDEBUG
                 ,
                 _step
#endif
      );
      // advance augmented handle
      aug_id = U.next_col_id(aug_id);
    }
  }

  template <bool IsSymm, class L_AugCcsType, class L_StartType,
            class CompIndexKernel>
  inline typename std::enable_if<!IsSymm>::type _update_L_start(
      L_AugCcsType &L, const size_type /* m */, L_StartType &L_start,
      const CompIndexKernel &comp_index) const {
    static_assert(!L_AugCcsType::ROW_MAJOR, "L must be AugCCS");
    using index_type                = typename L_AugCcsType::index_type;
    constexpr static bool ONE_BASED = L_AugCcsType::ONE_BASED;
    const auto            c_idx     = [](const size_type i) {
      return to_c_idx<size_type, ONE_BASED>(i);
    };

    if (!_step) return;

    L_start[_step - 1] = L.col_start()[_step - 1] - ONE_BASED;
    // get aug handle wrp the defer step
    index_type aug_id = L.start_row_id(deferred_step());
    // go thru all entries
    while (!L.is_nil(aug_id)) {
      // get column index
      const index_type col = L.col_idx(aug_id);
      // for each of this starting inds, advance one
      if (L_start[col] < L.col_start()[col + 1] - ONE_BASED &&
          c_idx(*(L.row_ind().cbegin() + L_start[col])) <= deferred_step())
        ++L_start[col];
      comp_index(*(L.row_ind().begin() + L.val_pos_idx(aug_id)), _defers
#ifndef NDEBUG
                 ,
                 _step
#endif
      );
      // advance augmented handle
      aug_id = L.next_row_id(aug_id);
    }
  }

  template <bool IsSymm, class L_AugCcsType, class L_StartType,
            class CompIndexKernel>
  inline typename std::enable_if<IsSymm>::type _update_L_start(
      L_AugCcsType &L, const size_type m, L_StartType &L_start,
      const CompIndexKernel & /* comp_index */) const {
    using index_type                = typename L_AugCcsType::index_type;
    constexpr static bool ONE_BASED = L_AugCcsType::ONE_BASED;
    const auto            ori_idx   = [](const size_type i) {
      return to_ori_idx<size_type, ONE_BASED>(i);
    };

    psmilu_assert(m, "cannot have empty leading block for symmetric case!");

    if (!_step) return;

    if (m >= L.nrows() && !_defers) {
      // if we have leading block that indicates fully symmetric system
      // then the ending position is stored
      // col_start has an extra element, thus this is valid accessing
      // or maybe we can use {Array,vector}::back??
      L_start[_step - 1] = L.col_start()[_step] - ONE_BASED;
      return;
    }
    // binary search to point to start of newly added row
    auto info          = find_sorted(L.row_ind_cbegin(_step - 1),
                            L.row_ind_cend(_step - 1), ori_idx(m));
    L_start[_step - 1] = info.second - L.row_ind().cbegin();
  }

 protected:
  size_type _defers;  ///< deferring counter

 private:
  using Crout::update_L_start;
  using Crout::update_U_start;
};
}  // namespace psmilu

#endif  // _PSMILU_DEFERRED_CROUT_HPP