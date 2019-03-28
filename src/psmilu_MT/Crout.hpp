//@HEADER
//----------------------------------------------------------------------------
//                Copyright (C) 2019 The PSMILU AUTHORS
//----------------------------------------------------------------------------
//@HEADER

/// \file psmilu_MT/Crout.hpp
/// \brief Implementation of modified \a Crout update in MT fashion
/// \authors Qiao,

#ifndef _PSMILU_MT_CROUT_HPP
#define _PSMILU_MT_CROUT_HPP

#include "psmilu_Crout.hpp"

namespace psmilu {
namespace mt {

/// \class Crout_MT
/// \ingroup mt
/// \brief detailed implementation of \a Crout updates in MT
class Crout_MT : public Crout {
 public:
  using size_type = Crout::size_type;  ///< size type

  /// \brief determine partition of threads
  /// \tparam IsSymm if \a true, the assume the leading block \a m is symmetric
  /// \param[in] n global size
  /// \param[in] m leading block size
  /// \param[in] threads number of threads
  /// \return starting thread ID for L part
  template <bool IsSymm>
  inline static int partition(const size_type n, const size_type m,
                              const int threads) {
    if (IsSymm) {
      const double r          = (double)m / n;
      const int    L_start_id = r * threads;
      return L_start_id;
    }
    return threads / 2;
  }

  /// \brief default constructor
  Crout_MT() = default;

  // default copy and assign
  Crout_MT(const Crout_MT &) = default;
  Crout_MT &operator=(const Crout_MT &) = default;

  template <bool IsSymm, class LeftDiagType, class CrsType, class CcsType,
            class RightDiagType, class PermType, class SpVecType>
  void load_ut_l(const LeftDiagType &s, const CrsType &crs_A,
                 const CcsType &ccs_A, const RightDiagType &t,
                 const PermType &p, const PermType &q, const size_type m,
                 const int ut_start_id, const int l_start_id, const int thread,
                 SpVecType &ut, SpVecType &l) {
    // compilation checking
    static_assert(!CcsType::ROW_MAJOR, "input A must be CCS for loading l");
    static_assert(CrsType::ROW_MAJOR, "input A must be CRS for loading ut");

    if (ut_start_id == thread) {
      // for ut group
      // clear sparse buffer
      ut.reset_counter();

      // first load the A
      Crout::_load_A2ut(s, crs_A, t, p[_step], q, ut);
    } else if (l_start_id == thread) {
      // for l group
      // clear sparse buffer
      l.reset_counter();

      // first load the A
      Crout::template _load_A2l<IsSymm>(s, ccs_A, t, p, q[_step], m, l);
    }
  }

  template <class U_CrsType, class U_PosArray, class DiagType,
            class L_AugCcsType, class SpVecType>
  void compute_ut(const L_AugCcsType &L, const DiagType &d, const U_CrsType &U,
                  const U_PosArray &                     U_start,
                  const typename U_PosArray::value_type *U_end, SpVecType &ut) {
    using index_type                = typename L_AugCcsType::index_type;
    constexpr static bool ONE_BASED = L_AugCcsType::ONE_BASED;

    if (_step) {
      // get the leading value and index iterators for L
      auto       U_v_first = U.vals().cbegin();
      auto       U_i_first = U.col_ind().cbegin();
      index_type aug_id    = L.start_row_id(_step);  // get starting column ID
      while (!L.is_nil(aug_id)) {
        // get the column index(C - based)
        // NOTE the column index is that of the row in U
        const index_type c_idx = L.col_idx(aug_id);
        psmilu_assert((size_type)c_idx < _step,
                      "compute_ut column index %zd should not exceed step %zd "
                      "for L",
                      (size_type)c_idx, _step);
        // NOTE once we drop val-pos impl, change this accordingly (L_start)
        // compute L*D
        const auto ld = L.val_from_row_id(aug_id) * d[c_idx];
        // get the starting position from U_start
        auto U_v_itr  = U_v_first + U_start[c_idx];
        auto U_i_itr  = U_i_first + U_start[c_idx],
             U_i_last = U_i_first + U_end[c_idx];
        // for this local range
        for (; U_i_itr < U_i_last; ++U_i_itr, ++U_v_itr) {
          // convert to c index
          const auto c_idx = to_c_idx<size_type, ONE_BASED>(*U_i_itr);
          ut.push_back(*U_i_itr, _step) ? ut.vals()[c_idx] = -ld * *U_v_itr
                                        : ut.vals()[c_idx] -= ld * *U_v_itr;
        }
        // advance augmented handle
        aug_id = L.next_row_id(aug_id);
      }
    }
  }

  template <class L_CcsType, class L_PosArray, class DiagType,
            class U_AugCrsType, class SpVecType>
  void compute_l(const L_CcsType &L, const L_PosArray &L_start,
                 const typename L_PosArray::value_type *L_end,
                 const DiagType &d, const U_AugCrsType &U, SpVecType &l) {
    using index_type                = typename U_AugCrsType::index_type;
    constexpr static bool ONE_BASED = L_CcsType::ONE_BASED;
    if (_step) {
      // get the leading value and index iterators for L
      auto       L_v_first = L.vals().cbegin();
      auto       L_i_first = L.row_ind().cbegin();
      index_type aug_id    = U.start_col_id(_step);  // get starting column ID
      // TODO the following ugly if-else can be clean-up by extracting the
      // base (Crout::) implementation out as an independent protected routine
      while (!U.is_nil(aug_id)) {
        // get the row index (C-based)
        // NOTE the row index is that of the column in L
        const index_type r_idx = U.row_idx(aug_id);
        // get the value position
        // NOTE this needs to be changed when we discard val_pos impl
        // (U_start) compute D*U first for this entry
        const auto du = d[r_idx] * U.val_from_col_id(aug_id);
        // get the starting position from L_start
        auto L_v_itr  = L_v_first + L_start[r_idx];
        auto L_i_itr  = L_i_first + L_start[r_idx],
             L_i_last = L_i_first + L_end[r_idx];
        // for this local range
        for (; L_i_itr < L_i_last; ++L_i_itr, ++L_v_itr) {
          // convert to c index
          const auto c_idx = to_c_idx<size_type, ONE_BASED>(*L_i_itr);
          // compute this entry, if index does not exist, assign new value to
          // -L*d*u, else, -= L*d*u
          l.push_back(*L_i_itr, _step) ? l.vals()[c_idx] = -*L_v_itr * du
                                       : l.vals()[c_idx] -= *L_v_itr * du;
        }
        // advance augmented handle
        aug_id = U.next_col_id(aug_id);
      }  // while
    }
  }

  template <class U_AugCrsType, class L_AugCcsType, class PosArray>
  inline void update_U_pos(const U_AugCrsType &U, const L_AugCcsType &L,
                           const size_type stride, const int start_id,
                           const int thread, PosArray &U_pos) {
    static_assert(U_AugCrsType::ROW_MAJOR, "U must be AugCRS");
    static_assert(!L_AugCcsType::ROW_MAJOR, "L must be AugCCS");
    using index_type                = typename U_AugCrsType::index_type;
    constexpr static bool ONE_BASED = U_AugCrsType::ONE_BASED;

    if (!_step) return;

    const auto c_idx = [](const size_type i) {
      return to_c_idx<size_type, ONE_BASED>(i);
    };

    // U_pos[_step - 1] = c_idx(U.row_start()[_step - 1]);
    auto            U_first = U.col_ind().cbegin();
    const int       abs_id  = thread - start_id;
    const size_type col_idx = _step + abs_id * stride;

    auto info        = find_sorted(U.col_ind_cbegin(_step - 1),
                            U.col_ind_cend(_step - 1), col_idx + ONE_BASED);
    U_pos[_step - 1] = info.second - U_first;

    index_type aug_id = L.start_row_id(_step);
    while (!L.is_nil(aug_id)) {
      const index_type row_idx = L.col_idx(aug_id);
      auto             last    = U.col_ind_cend(row_idx);
      auto             itr     = U_first + U_pos[row_idx];
      // amortized constant, no need to use binary search considering the
      // overhead
      for (; itr < last; ++itr)
        if (c_idx(*itr) >= col_idx) break;
      U_pos[row_idx] = itr - U_first;
      // advance augmented handle
      aug_id = L.next_row_id(aug_id);
    }
  }

  template <bool IsSymm, class L_AugCcsType, class U_AugCrsType, class PosArray>
  inline void update_L_pos(const L_AugCcsType &L, const U_AugCrsType &U,
                           const size_type m, const size_type stride,
                           const int start_id, const int thread,
                           PosArray &L_pos) {
    static_assert(U_AugCrsType::ROW_MAJOR, "U must be AugCRS");
    static_assert(!L_AugCcsType::ROW_MAJOR, "L must be AugCCS");
    using index_type                = typename U_AugCrsType::index_type;
    constexpr static bool ONE_BASED = U_AugCrsType::ONE_BASED;

    if (!_step) return;

    const auto c_idx = [](const size_type i) {
      return to_c_idx<size_type, ONE_BASED>(i);
    };

    const size_type leading = IsSymm ? m - 1u : _step;
    auto            L_first = L.row_ind().cbegin();
    const int       abs_id  = thread - start_id;
    const size_type row_idx = leading + abs_id * stride;

    auto info        = find_sorted(L.row_ind_cbegin(_step - 1),
                            L.row_ind_cend(_step - 1), leading + ONE_BASED);
    L_pos[_step - 1] = info.second - L_first;

    index_type aug_id = U.start_col_id(_step);
    while (!U.is_nil(aug_id)) {
      const index_type col_idx = U.row_idx(aug_id);
      auto             last    = L.row_ind_cend(col_idx);
      auto             itr     = L_first + L_pos[col_idx];
      for (; itr < last; ++itr)
        if (c_idx(*itr) >= row_idx) break;
      L_pos[col_idx] = itr - L_first;
      // advance augmented handle
      aug_id = U.next_col_id(aug_id);
    }
  }

 protected:
  using Crout::_step;  ///< bring in step attribute
};

}  // namespace mt
}  // namespace psmilu

#endif  // _PSMILU_MT_CROUT_HPP
