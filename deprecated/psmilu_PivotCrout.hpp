//@HEADER
//----------------------------------------------------------------------------
//                Copyright (C) 2019 The PSMILU AUTHORS
//----------------------------------------------------------------------------
//@HEADER

/// \file psmilu_PivotCrout.hpp
/// \brief Implementation of modified \a Crout update in deferred fashion, plus
///        with pivoting enabled
/// \authors Qiao,

#ifndef _PSMILU_PIVOT_CROUT_HPP
#define _PSMILU_PIVOT_CROUT_HPP

#include "psmilu_DeferredCrout.hpp"

namespace psmilu {

/// \class PivotCrout
/// \brief Crout update in deferred fashion
/// \ingroup crout
class PivotCrout : public DeferredCrout {
 public:
  using size_type = DeferredCrout::size_type;  ///< size type

  /// \brief default constructor
  PivotCrout() = default;

  // default copy and assign
  PivotCrout(const PivotCrout &) = default;
  PivotCrout &operator=(const PivotCrout &) = default;

  template <class ScaleArray, class CrsType, class PermType, class U_CrsType,
            class U_StartType, class DiagType, class L_AugCcsType,
            class SpVecType>
  void compute_ut_diag(const ScaleArray &s, const CrsType &crs_A,
                       const ScaleArray &t, const size_type pk,
                       const PermType &q, const L_AugCcsType &L,
                       const DiagType &d, const U_CrsType &U,
                       const U_StartType &U_start, SpVecType &ut,
                       typename SpVecType::value_type &dk) const {
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
    _load_A2ut_diag(s, crs_A, t, pk, q, ut, dk);

    // if not first step
    if (_step) {
      // get the leading value and index iterators for U
      auto U_v_first = U.vals().cbegin();
      auto U_i_first = U.col_ind().cbegin();
      // get the starting row ID with deferring
      index_type aug_id = L.start_row_id(deferred_step());
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
        // handle diagonal entry
        if (U_i_itr != U.col_ind_cbegin(col_idx)) {
          const auto prev_idx = to_c_idx<size_type, base>(*(U_i_itr - 1));
          if (prev_idx == _step) dk -= ld * *(U_v_itr - 1);
        }
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

  template <class ScaleArray, class CcsType, class PermType, class L_CcsType,
            class L_StartType, class DiagType, class U_AugCrsType,
            class SpVecType>
  void compute_l_diag(const ScaleArray &s, const CcsType &ccs_A,
                      const ScaleArray &t, const PermType &p,
                      const size_type qk, const L_CcsType &L,
                      const L_StartType &L_start, const DiagType &d,
                      const U_AugCrsType &U, SpVecType &l,
                      typename SpVecType::value_type &dk) const {
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
    _load_A2l_diag(s, ccs_A, t, p, qk, l, dk);

    // if not first step
    if (_step) {
      // get the leading value and index iterators for L
      auto L_v_first = L.vals().cbegin();
      auto L_i_first = L.row_ind().cbegin();
      // get the deferred column handle
      index_type aug_id = U.start_col_id(deferred_step());
      while (!U.is_nil(aug_id)) {
        // get the row index
        const size_type r_idx = U.row_idx(aug_id);
        psmilu_assert(
            r_idx < _step,
            "compute_ut row index %zd should not exceed step %zd for U", r_idx,
            _step);
        psmilu_assert(r_idx < L_start.size(), "%zd exceeds the L_start size",
                      r_idx);
        // compute d*U
        const auto du = d[r_idx] * U.val_from_col_id(aug_id);
        // get the starting position from L_start
        auto L_v_itr = L_v_first + L_start[r_idx];
        auto L_i_itr = L_i_first + L_start[r_idx];
        // handle diagonal entry
        if (L_i_itr != L.row_ind_cbegin(r_idx)) {
          const auto prev_idx = to_c_idx<size_type, base>(*(L_i_itr - 1));
          if (prev_idx == _step) dk -= du * *(L_v_itr - 1);
        }
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

  template <class U_AugCrsType, class U_StartType>
  inline void update_U_start_and_compress_U_wbak(
      U_AugCrsType &U, U_StartType &U_start, U_StartType &bak,
      typename U_StartType::size_type &nbaks) const {
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
      _update_U_start(U, U_start, bak, nbaks, comp_index_dummy);
    else
      _update_U_start(U, U_start, bak, nbaks, comp_index);
    U.col_start()[_step]  = U.col_start()[deferred_step()];
    U.col_end()[_step]    = U.col_end()[deferred_step()];
    U.col_counts()[_step] = U.col_counts()[deferred_step()];
  }

  template <class U_AugCrsType>
  inline void uncompress_U(U_AugCrsType &U, const size_type start,
                           const size_type end, const size_type cts) const {
    using index_type               = typename U_AugCrsType::index_type;
    const static auto uncomp_index = [](index_type &i, const size_type dfrs) {
      i += dfrs;
    };

    if (!_defers) {
      index_type aug_id = U.start_col_id(_step);
      while (!U.is_nil(aug_id)) {
        uncomp_index(*(U.col_ind().begin() + U.val_pos_idx(aug_id)), _defers);
        aug_id = U.next_col_id(aug_id);
      }
      U.col_start()[_step]  = start;
      U.col_end()[_step]    = end;
      U.col_counts()[_step] = cts;
    }
  }

  template <class L_AugCcsType, class L_StartType>
  inline void update_L_start_and_compress_L_wbak(
      L_AugCcsType &L, L_StartType &L_start, L_StartType &bak,
      typename L_StartType::size_type &nbaks) const {
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

    if (!_defers)
      _update_L_start(L, L_start, bak, nbaks, comp_index_dummy);
    else
      _update_L_start(L, L_start, bak, nbaks, comp_index);
    L.row_start()[_step]  = L.row_start()[deferred_step()];
    L.row_end()[_step]    = L.row_end()[deferred_step()];
    L.row_counts()[_step] = L.row_counts()[deferred_step()];
  }

  template <class L_AugCcsType>
  inline void uncompress_L(L_AugCcsType &L, const size_type start,
                           const size_type end, const size_type cts) const {
    using index_type               = typename L_AugCcsType::index_type;
    const static auto uncomp_index = [](index_type &i, const size_type dfrs) {
      i += dfrs;
    };

    if (!_defers) {
      index_type aug_id = L.start_row_id(_step);
      while (!L.is_nil(aug_id)) {
        uncomp_index(*(L.row_ind().begin() + L.val_pos_idx(aug_id)), _defers);
        aug_id = L.next_row_id(aug_id);
      }
      L.row_start()[_step]  = start;
      L.row_end()[_step]    = end;
      L.row_counts()[_step] = cts;
    }
  }

 protected:
  template <class ScaleArray, class CrsType, class PermType, class SpVecType>
  inline void _load_A2ut_diag(const ScaleArray &s, const CrsType &crs_A,
                              const ScaleArray &t, const size_type &pk,
                              const PermType &q, SpVecType &ut,
                              typename SpVecType::value_type &d) const {
    // compilation consistency checking
    static_assert(!(CrsType::ONE_BASED ^ SpVecType::ONE_BASED),
                  "inconsistent one-based in ccs and sparse vector");
    constexpr static bool base = CrsType::ONE_BASED;
    using value_type           = typename SpVecType::value_type;

    // ut should be empty
    psmilu_assert(ut.empty(), "ut should be empty while loading A");
    const size_type defer_thres = deferred_step();
    d                           = value_type(0);  // set diagonal to be zero
    // pk is c index
    auto       v_itr = crs_A.val_cbegin(pk);
    auto       i_itr = crs_A.col_ind_cbegin(pk);
    const auto s_pk  = s[pk];
    for (auto last = crs_A.col_ind_cend(pk); i_itr != last; ++i_itr, ++v_itr) {
      const auto      A_idx = to_c_idx<size_type, base>(*i_itr);
      const size_type c_idx = q[A_idx];
      if (c_idx >= defer_thres) {
        if (c_idx > defer_thres) {
          // get the gapped index
#ifndef NDEBUG
          const bool val_must_not_exit =
#endif
              ut.push_back(to_ori_idx<size_type, base>(c_idx), _step);
          psmilu_assert(val_must_not_exit,
                        "see prefix, failed on Crout step %zd for ut", _step);
          ut.vals()[c_idx] = s_pk * *v_itr * t[A_idx];  // scale here
        } else
          d = s_pk * *v_itr * t[A_idx];
      }
    }
  }

  template <class ScaleArray, class CcsType, class PermType, class SpVecType>
  inline void _load_A2l_diag(const ScaleArray &s, const CcsType &ccs_A,
                             const ScaleArray &t, const PermType &p,
                             const size_type &qk, SpVecType &l,
                             typename SpVecType::value_type &d) const {
    // compilation consistency checking
    static_assert(!(CcsType::ONE_BASED ^ SpVecType::ONE_BASED),
                  "inconsistent one-based in ccs and sparse vector");
    constexpr static bool base = CcsType::ONE_BASED;
    using value_type           = typename SpVecType::value_type;

    // runtime
    psmilu_assert(l.empty(), "l should be empty while loading A");
    const size_type defer_thres = deferred_step();
    d                           = value_type(0);
    // qk is c index
    auto       v_itr = ccs_A.val_cbegin(qk);
    auto       i_itr = ccs_A.row_ind_cbegin(qk);
    const auto t_qk  = t[qk];
    for (auto last = ccs_A.row_ind_cend(qk); i_itr != last; ++i_itr, ++v_itr) {
      const auto      A_idx = to_c_idx<size_type, base>(*i_itr);
      const size_type c_idx = p[A_idx];
      // push to the sparse vector only if its in range _step+1:n
      if (c_idx >= defer_thres) {
        if (c_idx > defer_thres) {
#ifndef NDEBUG
          const bool val_must_not_exit =
#endif
              l.push_back(to_ori_idx<size_type, base>(c_idx), _step);
          psmilu_assert(val_must_not_exit,
                        "see prefix, failed on Crout step %zd for l", _step);
          l.vals()[c_idx] = s[A_idx] * *v_itr * t_qk;  // scale here
        } else
          d = s[A_idx] * *v_itr * t_qk;
      }
    }
  }

  template <class U_AugCrsType, class U_StartType, class CompIndexKernel>
  inline void _update_U_start(U_AugCrsType &U, U_StartType &U_start,
                              U_StartType &                    bak,
                              typename U_StartType::size_type &nbaks,
                              const CompIndexKernel &comp_index) const {
    static_assert(U_AugCrsType::ROW_MAJOR, "U must be AugCRS");
    using index_type                = typename U_AugCrsType::index_type;
    constexpr static bool ONE_BASED = U_AugCrsType::ONE_BASED;
    const auto            c_idx     = [](const size_type i) {
      return to_c_idx<size_type, ONE_BASED>(i);
    };

    nbaks = 0;

    if (!_step) return;

    U_start[_step - 1] = U.row_start()[_step - 1] - ONE_BASED;
    // get the aug handle wrt defers
    index_type aug_id = U.start_col_id(deferred_step());
    // go through all non zeros entries
    while (!U.is_nil(aug_id)) {
      const index_type row = U.row_idx(aug_id);
      if (U_start[row] < U.row_start()[row + 1] - ONE_BASED &&
          c_idx(*(U.col_ind().cbegin() + U_start[row])) <= deferred_step()) {
        ++U_start[row];
        bak[nbaks++] = row;
      }
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

  template <class L_AugCcsType, class L_StartType, class CompIndexKernel>
  inline void _update_L_start(L_AugCcsType &L, L_StartType &L_start,
                              L_StartType &                    bak,
                              typename L_StartType::size_type &nbaks,
                              const CompIndexKernel &comp_index) const {
    static_assert(!L_AugCcsType::ROW_MAJOR, "L must be AugCCS");
    using index_type                = typename L_AugCcsType::index_type;
    constexpr static bool ONE_BASED = L_AugCcsType::ONE_BASED;
    const auto            c_idx     = [](const size_type i) {
      return to_c_idx<size_type, ONE_BASED>(i);
    };

    nbaks = 0;

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
          c_idx(*(L.row_ind().cbegin() + L_start[col])) <= deferred_step()) {
        ++L_start[col];
        bak[nbaks++] = col;
      }
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
};
}  // namespace psmilu

#endif  // _PSMILU_PIVOT_CROUT_HPP