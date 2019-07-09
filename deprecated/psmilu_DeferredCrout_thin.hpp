//@HEADER
//----------------------------------------------------------------------------
//                Copyright (C) 2019 The PSMILU AUTHORS
//----------------------------------------------------------------------------
//@HEADER

/// \file psmilu_DeferredCrout_thin.hpp
/// \brief Implementation of modified \a Crout update in deferred fashion
///        by using compressed data structures
/// \authors Qiao,

#ifndef _PSMILU_DEFERRED_CROUT_THIN_HPP
#define _PSMILU_DEFERRED_CROUT_THIN_HPP

#include "psmilu_DeferredCrout.hpp"
#include "psmilu_utils.hpp"

namespace psmilu {

/// \class DeferredCrout_thin
/// \brief Crout update in deferred fashion
/// \ingroup crout
///
/// The main difference between this version and DeferredCrout is the way to
/// treat the starting positions, i.e. L_start and U_start. In this version,
/// The starting positions store the value relative to current row/column,
/// whereas the original version stores the global positions.
///
/// In terms of data structure wise change, the thin version no longer requires
/// using augmented storage, which enables linked lists for all entries but
/// requires a significantly amount of memory. In addition, augmented data
/// structures are not needed for strictly deferred factorization.
class DeferredCrout_thin : public DeferredCrout {
 public:
  using size_type = DeferredCrout::size_type;  ///< size type

  /// \brief default constructor
  DeferredCrout_thin() = default;

  // default copy and assign
  DeferredCrout_thin(const DeferredCrout_thin &) = default;
  DeferredCrout_thin &operator=(const DeferredCrout_thin &) = default;

  /// \brief dual of the \ref compute_l, i.e. computing the row of U
  /// \tparam ScaleArray row scaling diagonal matrix type, see \ref Array
  /// \tparam CrsType crs format, see \ref CRS
  /// \tparam PermType permutation matrix type, see \ref Array
  /// \tparam PosArray position array type, see \ref Array
  /// \tparam DiagType diagonal matrix type, see \ref Array
  /// \tparam CcsType ccs format, see \ref CCS
  /// \tparam SpVecType work array for current row vector, see \ref SparseVector
  /// \param[in] s row scaling matrix from preprocessing
  /// \param[in] crs_A input matrix in crs scheme
  /// \param[in] t column scaling matrix from preprocessing
  /// \param[in] pk permutated row index
  /// \param[in] q column inverse permutation (deferred)
  /// \param[in] L lower part
  /// \param[in] L_start starting positions for current step
  /// \param[in] L_list linked list of column indices for current step
  /// \param[in] d diagonal vector
  /// \param[in] U upper part
  /// \param[in] U_start leading row positions of current \ref _step
  /// \param[out] ut current row vector of U
  /// \sa compute_l
  ///
  /// This routine computes the current row vector of \f$\boldsymbol{U}\f$
  /// (w/o diagonal scaling). Mathematically, this routine is to compute:
  ///
  /// \f[
  ///   \boldsymbol{u}_{k}^{T}=
  ///     \hat{\boldsymbol{A}}[p_{k},\boldsymbol{q}_{k+1:n}]-
  ///     \boldsymbol{L}_{k,1:k-1}\boldsymbol{D}_{k-1}
  ///       \boldsymbol{U}_{1:k-1,k+1:n}
  /// \f]
  ///
  /// It's worth noting that, conceptally, the formula above is nothing but
  /// a vector matrix operation. However, standard implementation won't give
  /// good performance (especially with consideration of cache performance),
  /// this is because \f$\boldsymbol{L}\f$ is stored in column major whereas
  /// row major for \f$\boldsymbol{U}\f$. Therefore, the actual implementation
  /// is in the fashion that loops through \f$\boldsymbol{U}\f$ in row major
  /// while keeping \f$\boldsymbol{L}\f$ as much static as possible.
  ///
  /// Regarding the complexity cost, it takes
  /// \f$\matchcal{O}(\textrm{nnz}(\boldsymbol{A}_{p_k,:}))\f$ to first load the
  /// row of A to \f$\boldsymbol{u}_k^T\f$, and takes
  /// \f$\mathcal{O}(\cup_j \textrm{nnz}(\boldsymbol{U}_{j,:}))\f$, where
  /// \f$l_{kj}\neq 0\f$ for computing the vector matrix operation.
  template <class ScaleArray, class CrsType, class PermType, class PosArray,
            class DiagType, class CcsType, class SpVecType>
  void compute_ut(const ScaleArray &s, const CrsType &crs_A,
                  const ScaleArray &t, const size_type pk, const PermType &q,
                  const CcsType &L, const PosArray &L_start,
                  const PosArray &L_list, const DiagType &d, const CrsType &U,
                  const PosArray &U_start, SpVecType &ut) const {
    // compilation checking
    static_assert(CrsType::ROW_MAJOR, "input A must be CRS for loading ut");
    using index_type                 = typename PosArray::value_type;
    constexpr static index_type nil  = static_cast<index_type>(-1);
    constexpr static bool       base = CrsType::ONE_BASED;

    // reset sparse buffer
    ut.reset_counter();

    // first load the A row
    _load_A2ut(s, crs_A, t, pk, q, ut);

    // if not first step
    if (_step) {
      index_type L_col = L_list[deferred_step()];
      while (L_col != nil) {
        psmilu_assert(
            (size_type)L_col < _step,
            "compute_ut column index %zd should not exceed step %zd for L",
            (size_type)L_col, _step);
        // compute L*d
        const auto ld      = *(L.val_cbegin(L_col) + L_start[L_col]) * d[L_col];
        auto       U_v_itr = U.val_cbegin(L_col) + U_start[L_col];
        auto       U_i_itr = U.col_ind_cbegin(L_col) + U_start[L_col],
             U_last        = U.col_ind_cend(L_col);
        if (U_i_itr != U_last &&
            to_c_idx<size_type, base>(*U_i_itr) == deferred_step()) {
          // advance to offsets
          ++U_i_itr;
          ++U_v_itr;
        }
        for (; U_i_itr != U_last; ++U_i_itr, ++U_v_itr) {
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
        // get to next column index
        L_col = L_list[L_col];
      }  // while
    }
  }

  /// \brief compute the column vector of L of current step
  /// \tparam IsSymm if \a true, then a symmetric leading block is assumed
  /// \tparam ScaleArray row/column scaling diagonal matrix type, see \ref Array
  /// \tparam CcsType ccs format, see \ref CCS
  /// \tparam PermType permutation matrix type, see \ref Array
  /// \tparam PosArray position array type, see \ref Array
  /// \tparam DiagType array used for storing diagonal array, see \ref Array
  /// \tparam CrsType crs format, see \ref CRS
  /// \tparam SpVecType sparse vector for storing l, see \ref SparseVector
  /// \param[in] s row scaling matrix from preprocessing
  /// \param[in] ccs_A input matrix in ccs scheme
  /// \param[in] t column scaling matrix from preprocessing
  /// \param[in] p row inverse permutation matrix (deferred)
  /// \param[in] qk permutated column index
  /// \param[in] m leading block size
  /// \param[in] L lower part of decomposition
  /// \param[in] L_start position array storing the starting locations of \a L
  /// \param[in] d diagonal entries
  /// \param[in] U augmented U matrix
  /// \param[in] U_start starting positions of U
  /// \param[in] U_list linked list of row indices of current step in U
  /// \param[out] l column vector of L at current \ref _step
  /// \sa compute_ut
  ///
  /// This routine computes the current column vector of \f$\boldsymbol{L}\f$
  /// (w/o diagonal scaling). Mathematically, this routine is to compute:
  ///
  /// \f[
  ///   \boldsymbol{\ell}_{k}=
  ///     \hat{\boldsymbol{A}}[\boldsymbol{p}_{k+1:n},q_{k}]-
  ///     \boldsymbol{L}_{k+1:n,1:k-1}\boldsymbol{D}_{k-1}
  ///       \boldsymbol{U}_{1:k-1,k}
  /// \f]
  ///
  /// It's worth noting that, conceptally, the formula above is nothing but
  /// a matrix vector operation. However, standard implementation won't give
  /// good performance (especially with consideration of cache performance),
  /// this is because \f$\boldsymbol{L}\f$ is stored in column major whereas
  /// row major for \f$\boldsymbol{U}\f$. Therefore, the actual implementation
  /// is in the fashion that loops through \f$\boldsymbol{L}\f$ in column major
  /// while keeping \f$\boldsymbol{U}\f$ as much static as possible.
  ///
  /// Regarding the complexity cost, it takes
  /// \f$\matchcal{O}(\textrm{nnz}(\boldsymbol{A}_{:,q_k}))\f$ to first load the
  /// column of A to \f$\boldsymbol{\ell}_k\f$, and takes
  /// \f$\mathcal{O}(\cup_i \textrm{nnz}(\boldsymbol{L}_{:,i}))\f$, where
  /// \f$u_{ik}\neq 0\f$ for computing the matrix vector operation.
  template <bool IsSymm, class ScaleArray, class CcsType, class PermType,
            class PosArray, class DiagType, class CrsType, class SpVecType>
  void compute_l(const ScaleArray &s, const CcsType &ccs_A, const ScaleArray &t,
                 const PermType &p, const size_type qk, const size_type m,
                 const CcsType &L, const PosArray &L_start, const DiagType &d,
                 const CrsType &U, const PosArray &U_start,
                 const PosArray &U_list, SpVecType &l) const {
    // compilation checking
    static_assert(!CcsType::ROW_MAJOR, "input A must be CCS for loading l");
    using index_type                 = typename PosArray::value_type;
    constexpr static index_type nil  = static_cast<index_type>(-1);
    constexpr static bool       base = CcsType::ONE_BASED;

    // clear sparse counter
    l.reset_counter();

    if (!IsSymm || (_defers || m != ccs_A.nrows())) {
      // load A column
      _load_A2l<IsSymm>(s, ccs_A, t, p, qk, m, l);

      // if not first step
      if (_step) {
        index_type U_row = U_list[deferred_step()];
        while (U_row != nil) {
          if (!IsSymm)
            psmilu_assert(
                (size_type)U_row < _step,
                "compute_ut row index %zd should not exceed step %zd for U",
                (size_type)U_row, _step);
          // compute d*U
          const auto du = d[U_row] * *(U.val_cbegin(U_row) + U_start[U_row]);
          auto       L_v_itr = L.val_cbegin(U_row) + L_start[U_row];
          auto       L_i_itr = L.row_ind_cbegin(U_row) + L_start[U_row],
               L_last        = L.row_ind_cend(U_row);
          if (L_i_itr != L_last &&
              to_c_idx<size_type, base>(*L_i_itr) == deferred_step()) {
            ++L_i_itr;
            ++L_v_itr;
          }
          for (; L_i_itr != L_last; ++L_i_itr, ++L_v_itr) {
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
          // update U linked list
          U_row = U_list[U_row];
        }  // while
      }
    }
  }

  /// \brief compress L and U and update their corresponding starting positions
  ///        and linked lists
  /// \tparam CsType either \ref CRS or \ref CCS
  /// \tparam PosArray position array type, see \ref Array
  /// \param[in,out] T strictly lower (L) or upper (U) matrices
  /// \param[in,out] list linked list of current step with primary indices
  /// \param[in,out] start local positions
  ///
  /// The overall idea involves the following steps: (1) loop through the
  /// current secondary axis to get the primary indices in \a list, then (2)
  /// from these primary indices combining with starting positions, \a start,
  /// to get the index and value arrays position, (3) next step is to compress
  /// the index wrt to the current overall deferals, finally (4) advance the
  /// starting position and update the linked list accordingly.
  ///
  /// Clearly, all these steps must be done concurrently with in a single loop.
  template <class CsType, class PosArray>
  inline void update_compress(CsType &T, PosArray &list,
                              PosArray &start) const {
    using index_type                 = typename PosArray::value_type;
    constexpr static index_type nil  = static_cast<index_type>(-1);
    constexpr static bool       base = CsType::ONE_BASED;

    // now update k
    index_type idx = list[deferred_step()];
    while (idx != nil) {
      auto itr = T.ind_begin(idx) + start[idx];
      *itr -= _defers;  // compress
      ++start[idx];     // increment next position
      const auto next = list[idx];
      if (++itr != T.ind_end(idx)) {
        const size_type j = to_c_idx<size_type, base>(*itr);
        list[idx]         = list[j];
        list[j]           = idx;
      } else
        list[idx] = nil;
      idx = next;  // update linked list
    }              // while
    list[_step] = nil;
    // need to update newly added entry at the end
    start[_step] = 0;
    if (T.nnz_in_primary(_step)) {
      const size_type first = to_c_idx<size_type, base>(*T.ind_cbegin(_step));
      if (list[first] != nil) list[_step] = list[first];
      list[first] = _step;
      psmilu_assert(first != deferred_step(),
                    "newly added node cannot have diagonal entry");
    }
  }

  /// \brief another starting array needed in symmetric computation to get
  ///        the offsets
  /// \tparam CcsType ccs type for L, see \ref CCS
  /// \tparam PosArray position array, see \ref Array
  /// \param[in] L lower factor
  /// \param[in] m leading block size
  /// \param[out] L_start offset starting positions
  template <class CcsType, class PosArray>
  inline void update_L_start_offset_symm(const CcsType &                   L,
                                         const typename CcsType::size_type m,
                                         PosArray &L_start) const {
    static_assert(!CcsType::ROW_MAJOR, "must be CCS");
    constexpr static bool base = CcsType::ONE_BASED;

    auto info =
        find_sorted(L.row_ind_cbegin(_step), L.row_ind_cend(_step), m + base);
    L_start[_step] = info.second - L.row_ind_cbegin(_step);
  }

  /// \brief estimate the inverse norm w/o augmented data structures
  /// \tparam CsType either \ref CRS (U) or \ref CCS (L)
  /// \tparam PosArray position array type, see \ref Array
  /// \tparam KappaArray array type for storing inverse norms, see \ref Array
  /// \param[in] T lower or upper factors
  /// \param[in] list linked list of primary indices of current step
  /// \param[in] start starting positions in index and value arrays
  /// \param[in,out] kappa inverse norm solutions
  template <class CsType, class PosArray, class KappaArray>
  inline bool update_kappa(const CsType &T, const PosArray &list,
                           const PosArray &start, KappaArray &kappa) const {
    using value_type                = typename CsType::value_type;
    using index_type                = typename PosArray::value_type;
    constexpr static index_type nil = static_cast<index_type>(-1);
    constexpr static bool       one = true, neg_one = false;
    if (!_step) {
      kappa[0] = value_type(1);
      return one;
    }

    // need to sum all values
    value_type sum(0);
    index_type idx = list[deferred_step()];
    while (idx != nil) {
      psmilu_assert((size_type)idx < _step,
                    "the triangular matrices should be strict lower/upper, "
                    "idx=%zd/step=%zd",
                    (size_type)idx, _step);
      sum += kappa[idx] * *(T.val_cbegin(idx) + start[idx]);
      idx = list[idx];  // traverse to next node
    }
    const value_type k1 = value_type(1) - sum, k2 = -value_type(1) - sum;
    if (std::abs(k1) < std::abs(k2)) {
      kappa[_step] = k2;
      return neg_one;
    }
    kappa[_step] = k1;
    return one;
  }

  /// \brief defer an secondary entry
  /// \tparam CsType either \ref CRS (U) or \ref CCS (L)
  /// \tparam PosArray position array, see \ref Array
  /// \param[in] to_idx deferred destination
  /// \param[in] start starting positions
  /// \param[in,out] T lower or upper factors
  /// \param[in,out] list linked list of primary indices
  template <class CsType, class PosArray>
  inline void defer_entry(const size_type to_idx, const PosArray &start,
                          CsType &T, PosArray &list) const {
    using index_type                 = typename PosArray::value_type;
    constexpr static index_type nil  = static_cast<index_type>(-1);
    constexpr static bool       base = CsType::ONE_BASED;

    psmilu_assert(list[to_idx] == nil, "deferred location %zd must be nil",
                  to_idx);

    index_type idx = list[deferred_step()];
    while (idx != nil) {
      auto itr = T.ind_begin(idx) + start[idx];
      psmilu_assert(itr != T.ind_end(idx), "fatal");
      const size_type n  = T.ind_end(idx) - itr,
                      vp = T.ind_start()[idx] + start[idx] - base;
      *itr               = to_idx + base;  // set deferred index
      rotate_left(n, vp, T.inds());
      rotate_left(n, vp, T.vals());
      // update linked list
      itr                  = T.ind_begin(idx) + start[idx];
      const size_type j    = *itr - base;
      const auto      next = list[idx];
      list[idx]            = list[j];
      list[j]              = idx;
      idx                  = next;  // next node
    }
    list[_step] = nil;
  }

  /// \brief for symmetric leading block, we need to fix the offset positions
  ///        while we do deferring
  /// \tparam CcsType CCS storage type for L
  /// \tparam PosArray position array, see \ref Array
  /// \param[in] to_idx deferred destination
  /// \param[in] L_start starting positions
  /// \param[in,out] L lower factor L
  /// \param[in,out] L_list linked list of column indices
  /// \param[out] L_offsets offsets of asymmetric potions
  template <class CcsType, class PosArray>
  inline void defer_L_and_fix_offsets(const size_type to_idx,
                                      const PosArray &L_start, CcsType &L,
                                      PosArray &L_list,
                                      PosArray &L_offsets) const {
    static_assert(!CcsType::ROW_MAJOR, "must be CCS");
    using index_type                 = typename PosArray::value_type;
    constexpr static index_type nil  = static_cast<index_type>(-1);
    constexpr static bool       base = CcsType::ONE_BASED;

    index_type idx = L_list[deferred_step()];
    while (idx != nil) {
      auto            itr = L.row_ind_begin(idx) + L_start[idx];
      const size_type n   = L.row_ind_end(idx) - itr,
                      vp  = L.col_start()[idx] + L_start[idx] - base;
      *itr                = to_idx + base;
      rotate_left(n, vp, L.inds());
      rotate_left(n, vp, L.vals());
      itr                  = L.row_ind_begin(idx) + L_start[idx];
      const size_type j    = *itr - base;
      const auto      next = L_list[idx];
      L_list[idx]          = L_list[j];
      L_list[j]            = idx;
      --L_offsets[idx];
      idx = next;
    }
    L_list[_step] = nil;
  }
};
}  // namespace psmilu

#endif  // _PSMILU_DEFERRED_CROUT_THIN_HPP