///////////////////////////////////////////////////////////////////////////////
//                This file is part of HILUCSI project                       //
///////////////////////////////////////////////////////////////////////////////

/*!
 * \file hilucsi/alg/Crout.hpp
 * \brief Implementation of modified \a Crout update in deferred fashion
 *        by using compressed data structures
 * \author Qiao Chen

\verbatim
Copyright (C) 2019 NumGeom Group at Stony Brook University

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

#ifndef _HILUCSI_ALG_CROUT_HPP
#define _HILUCSI_ALG_CROUT_HPP

#include <algorithm>
#include <cstddef>
#include <type_traits>

#include "hilucsi/utils/common.hpp"
#include "hilucsi/utils/log.hpp"

namespace hilucsi {

/// \class Crout
/// \brief Crout update in deferred fashion
/// \ingroup crout
///
/// The idea of this class is to implement Crout kernels that wrap around a
/// common factor---step. In this case, the class Crout is as simple (in terms
/// of data size) as an integer. It also supports implicitly casting to integer
/// as well as increment operations (both suffix and prefix versions). The whole
/// disign can be demonstrate by the following C++ pseudocode:
///
/// \code{.cpp}
/// for (Crout crout; crout < m; ++crout) {
///   crout.do_work1(...);
///   crout.do_work2(...);
///   ...
/// }
/// \endcode
class Crout {
 public:
  using size_type = std::size_t;  ///< size type

  /// \brief default constructor
  Crout() : _step(0), _defers(0) {}

  // default copy and assign
  Crout(const Crout &) = default;
  Crout &operator=(const Crout &) = default;

  /// \brief assign to a step
  /// \param[in] step step number
  Crout &operator=(const size_type step) {
    _step = step;
    return *this;
  }

  /// \brief increment Crout step
  inline const Crout &operator++() {
    ++_step;
    return *this;
  }

  /// \brief increment Crout step, suffix
  inline Crout operator++(int) {
    const Crout tmp(*this);
    ++_step;
    return tmp;
  }

  /// \brief implicitly casting to size_type
  inline operator size_type() const { return _step; }

  /// \brief get the current defers
  inline size_type defers() const { return _defers; }

  /// \brief get the deferred step
  inline size_type deferred_step() const { return _defers + _step; }

  /// \brief increment the defer counter
  inline void increment_defer_counter() { ++_defers; }

  /// \brief compress an array without deferrals (gaps)
  /// \tparam ArrayType in and out array type
  /// \param[in,out] v v[k]=v[k+d]
  template <class ArrayType>
  inline void compress_array(ArrayType &v) const {
    v[_step] = v[deferred_step()];
  }

  /// \brief compress an array and assign the no-gapped value to another one
  /// \tparam ArrayIn input array type
  /// \tparam ArrayOut output array type
  /// \param[in] r rhs array
  /// \param[in] l lhs array
  /// \note l[k]=r[k+d]
  template <class ArrayIn, class ArrayOut>
  inline void assign_gap_array(const ArrayIn &r, ArrayOut &l) const {
    l[_step] = r[deferred_step()];
  }

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
  /// This routine computes the current row vector of \f$\mathbf{U}\f$
  /// (w/o diagonal scaling). Mathematically, this routine is to compute:
  ///
  /// \f[
  ///   \mathbf{u}_{k}^{T}=
  ///     \hat{\mathbf{A}}[p_{k},\mathbf{q}_{k+1:n}]-
  ///     \mathbf{L}_{k,1:k-1}\mathbf{D}_{k-1}
  ///       \mathbf{U}_{1:k-1,k+1:n}
  /// \f]
  ///
  /// It's worth noting that, conceptally, the formula above is nothing but
  /// a vector matrix operation. However, standard implementation won't give
  /// good performance (especially with consideration of cache performance),
  /// this is because \f$\mathbf{L}\f$ is stored in column major whereas
  /// row major for \f$\mathbf{U}\f$. Therefore, the actual implementation
  /// is in the fashion that loops through \f$\mathbf{U}\f$ in row major
  /// while keeping \f$\mathbf{L}\f$ as much static as possible.
  ///
  /// Regarding the complexity cost, it takes
  /// \f$\mathcal{O}(\textrm{nnz}(\mathbf{A}_{p_k,:}))\f$ to first load the
  /// row of A to \f$\mathbf{u}_k^T\f$, and takes
  /// \f$\mathcal{O}(\cup_j \textrm{nnz}(\mathbf{U}_{j,:}))\f$, where
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
    using index_type                = typename PosArray::value_type;
    constexpr static index_type nil = static_cast<index_type>(-1);

    // reset sparse buffer
    ut.reset_counter();

    // first load the A row
    _load_A2ut(s, crs_A, t, pk, q, ut);

    // if not first step
    if (_step) {
      index_type L_col = L_list[deferred_step()];
      while (L_col != nil) {
        hilucsi_assert(
            (size_type)L_col < _step,
            "compute_ut column index %zd should not exceed step %zd for L",
            (size_type)L_col, _step);
        // compute L*d
        const auto ld      = *(L.val_cbegin(L_col) + L_start[L_col]) * d[L_col];
        auto       U_v_itr = U.val_cbegin(L_col) + U_start[L_col];
        auto       U_i_itr = U.col_ind_cbegin(L_col) + U_start[L_col],
             U_last        = U.col_ind_cend(L_col);
        if (U_i_itr != U_last && size_type(*U_i_itr) == deferred_step()) {
          // advance to offsets
          ++U_i_itr;
          ++U_v_itr;
        }
        for (; U_i_itr != U_last; ++U_i_itr, ++U_v_itr) {
          // convert to c index
          const size_type idx = *U_i_itr;
          hilucsi_assert(idx > deferred_step(),
                         "U index %zd in computing ut should greater than step "
                         "%zd(defers:%zd)",
                         idx, _step, _defers);
          if (ut.push_back(idx, _step))
            ut.vals()[idx] = -ld * *U_v_itr;
          else
            ut.vals()[idx] -= ld * *U_v_itr;
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
  /// This routine computes the current column vector of \f$\mathbf{L}\f$
  /// (w/o diagonal scaling). Mathematically, this routine is to compute:
  ///
  /// \f[
  ///   \mathbf{\ell}_{k}=
  ///     \hat{\mathbf{A}}[\mathbf{p}_{k+1:n},q_{k}]-
  ///     \mathbf{L}_{k+1:n,1:k-1}\mathbf{D}_{k-1}
  ///       \mathbf{U}_{1:k-1,k}
  /// \f]
  ///
  /// It's worth noting that, conceptally, the formula above is nothing but
  /// a matrix vector operation. However, standard implementation won't give
  /// good performance (especially with consideration of cache performance),
  /// this is because \f$\mathbf{L}\f$ is stored in column major whereas
  /// row major for \f$\mathbf{U}\f$. Therefore, the actual implementation
  /// is in the fashion that loops through \f$\mathbf{L}\f$ in column major
  /// while keeping \f$\mathbf{U}\f$ as much static as possible.
  ///
  /// Regarding the complexity cost, it takes
  /// \f$\mathcal{O}(\textrm{nnz}(\mathbf{A}_{:,q_k}))\f$ to first load the
  /// column of A to \f$\mathbf{\ell}_k\f$, and takes
  /// \f$\mathcal{O}(\cup_i \textrm{nnz}(\mathbf{L}_{:,i}))\f$, where
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
    using index_type                = typename PosArray::value_type;
    constexpr static index_type nil = static_cast<index_type>(-1);

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
            hilucsi_assert(
                (size_type)U_row < _step,
                "compute_ut row index %zd should not exceed step %zd for U",
                (size_type)U_row, _step);
          // compute d*U
          const auto du = d[U_row] * *(U.val_cbegin(U_row) + U_start[U_row]);
          auto       L_v_itr = L.val_cbegin(U_row) + L_start[U_row];
          auto       L_i_itr = L.row_ind_cbegin(U_row) + L_start[U_row],
               L_last        = L.row_ind_cend(U_row);
          if (L_i_itr != L_last && size_type(*L_i_itr) == deferred_step()) {
            ++L_i_itr;
            ++L_v_itr;
          }
          for (; L_i_itr != L_last; ++L_i_itr, ++L_v_itr) {
            // convert to c index
            const auto idx = *L_i_itr;
            hilucsi_assert(
                idx > deferred_step(),
                "L index %zd in computing l should greater than step "
                "%zd(defers:%zd)",
                idx, _step, _defers);
            // compute this entry, if index does not exist, assign new value to
            // -L*d*u, else, -= L*d*u
            if (l.push_back(idx, _step))
              l.vals()[idx] = -du * *L_v_itr;
            else
              l.vals()[idx] -= *L_v_itr * du;
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
    using index_type                = typename PosArray::value_type;
    constexpr static index_type nil = static_cast<index_type>(-1);

    // now update k
    index_type idx = list[deferred_step()];
    while (idx != nil) {
      auto itr = T.ind_begin(idx) + start[idx];
      *itr -= _defers;  // compress
      ++start[idx];     // increment next position
      const auto next = list[idx];
      if (++itr != T.ind_end(idx)) {
        list[idx]  = list[*itr];
        list[*itr] = idx;
      } else
        list[idx] = nil;
      idx = next;  // update linked list
    }              // while
    list[_step] = nil;
    // need to update newly added entry at the end
    start[_step] = 0;
    if (T.nnz_in_primary(_step)) {
      const size_type first = *T.ind_cbegin(_step);
      if (list[first] != nil) list[_step] = list[first];
      list[first] = _step;
      hilucsi_assert(first != deferred_step(),
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

    auto info = find_sorted(L.row_ind_cbegin(_step), L.row_ind_cend(_step), m);
    L_start[_step] = info.second - L.row_ind_cbegin(_step);
  }

  /// \brief estimate the inverse norm
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
      hilucsi_assert((size_type)idx < _step,
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

  /// \brief update the current diagonal entry
  /// \tparam IsSymm if \a true, then assuming we have a leading block
  /// \tparam SpVecType sparse vector for storing l/U, see \ref SparseVector
  /// \tparam DiagType diagonal vector array type, see \ref Array
  /// \param[in] l newly computed column vector of L
  /// \param[in] ut newly computed row vector of U
  /// \param[in] m leading block size (not necessary for symmetric)
  /// \param[in,out] d diagonal vector
  ///
  /// This routine is \a SFINAE-able by \a IsSymm and it's for asymmetric cases.
  /// This routine is to compute the last step in Crout updates, i.e. updating
  /// the diagonal entries:
  ///
  /// \f[
  ///   \mathbf{D}_{k+1:m,k+1:m}=\mathbf{D}_{k+1:m,k+1:m}-
  ///     \mathbf{D}_{k,k}
  ///     \left(\mathbf{\ell}_k\otimes\mathbf{u}_k\right)_{k+1:m,k+1:m}
  /// \f]
  ///
  /// The complexity is bounded by:
  ///
  /// \f[
  ///   \mathcal{O}(\min(\textrm{nnz}(\mathbf{\ell}_k),
  ///     \textrm{nnz}(\mathbf{u}_k)))
  ///\f]
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
        const size_type idx = ut.idx(i);
        hilucsi_assert(idx > deferred_step(),
                       "should only contain the upper part of ut, "
                       "(idx,step(defer))=(%zd,%zd(%zd))!",
                       idx, _step, _defers);
        // if the dense tags of this entry points to this step, we know l has
        // an element in this slot
        if (idx < m && (size_type)l_d_tags[idx] == _step)
          d[idx] -= ut.val(i) * l_vals[idx];
      }
    } else {
      const auto &ut_d_tags  = static_cast<const extractor &>(ut).dense_tags();
      const size_type n      = l.size();
      const auto &    u_vals = ut.vals();
      for (size_type i = 0u; i < n; ++i) {
        const size_type idx = l.idx(i);
        hilucsi_assert(idx > deferred_step(),
                       "should only contain the lower part of l, "
                       "(idx,step(defer))=(%zd,%zd(%zd))!",
                       idx, _step, _defers);
        // if the dense tags of this entry points to this step, we know ut has
        // an element in this slot
        if (idx < m && (size_type)ut_d_tags[idx] == _step)
          d[idx] -= u_vals[idx] * l.val(i);
      }
    }
  }

  /// \brief update the current diagonal entry
  /// \tparam IsSymm if \a true, then assuming we have a leading block
  /// \tparam SpVecType sparse vector for storing l, see \ref SparseVector
  /// \tparam DiagType diagonal vector array type, see \ref Array
  /// \param[in] ut newly computed row vector of U
  /// \param[in] m leading block size (not necessary for symmetric)
  /// \param[in,out] d diagonal vector
  ///
  /// This routine is \a SFINAE-able by \a IsSymm and it's for symmetric cases.
  /// This routine is to compute the last step in Crout updates, i.e. updating
  /// the diagonal entries:
  ///
  /// \f[
  ///   \mathbf{D}_{k+1:m,k+1:m}=\mathbf{D}_{k+1:m,k+1:m}-
  ///     \mathbf{D}_{k,k}
  ///     \left(\mathbf{\ell}_k\otimes\mathbf{u}_k\right)_{k+1:m,k+1:m}
  /// \f]
  ///
  /// The complexity is bounded by:
  ///
  /// \f[
  ///   \mathcal{O}(\textrm{nnz}(\mathbf{u}_k))
  ///\f]
  template <bool IsSymm, class SpVecType, class DiagType>
  inline typename std::enable_if<IsSymm>::type update_B_diag(
      const SpVecType & /* l */, const SpVecType &ut, const size_type m,
      DiagType &d) const {
    hilucsi_assert(m, "fatal, symmetric block cannot be empty!");
    // get the current diagonal entry
    const auto dk = d[_step];
    // we only need to deal with ut
    const size_type n = ut.size();
    for (size_type i(0); i < n; ++i) {
      const size_type idx = ut.idx(i);
      hilucsi_assert(idx > deferred_step(),
                     "ut should only contain the upper part, "
                     "(idx,step(defer))=(%zd,%zd(%zd))!",
                     idx, _step, _defers);
      if (idx < m) d[idx] -= dk * ut.val(i) * ut.val(i);
    }
  }

  /// \brief scale the computed row/column vectors by current diagonal inverse
  /// \tparam DiagType diagonal matrix type, see \ref Array
  /// \tparam SpVecType sparse vector for storing l/U, see \ref SparseVector
  /// \param[in] d diagonal vector
  /// \param[in,out] v vector to be scaled
  /// \return The returned boolean flag indices a singularity occurs if \a true
  ///
  /// This routine computes \f$\mathbf{v}=\mathbf{v}/d_k\f$; firstly,
  /// singularity is checked to ensure the validation of computation. Then,
  /// we test if it's safe to invert \f$d_k\f$ so that we can use multiplication
  /// instead of division.
  ///
  /// Regarding complexity, this routine takes
  /// \f$\mathcal{O}(\textrm{nnz}(\mathbf{v}))\f$
  template <class DiagType, class SpVecType>
  inline bool scale_inv_diag(const DiagType &d, SpVecType &v) const {
    using value_t                     = typename DiagType::value_type;
    constexpr static value_t zero     = Const<value_t>::ZERO;
    constexpr static value_t safe_min = Const<value_t>::MIN;
    constexpr static bool    okay     = false;

    const value_t dk = d[_step];
    // first, if exactly zero, return fail
    if (dk == zero) return !okay;

    const size_type n    = v.size();
    auto &          vals = v.vals();

    if (std::abs(dk) > safe_min) {
      // take the inverse, do multiply
      const value_t dk_inv = Const<value_t>::ONE / dk;
      for (size_type i = 0u; i < n; ++i) vals[v.idx(i)] *= dk_inv;
    } else
      for (size_type i = 0u; i < n; ++i) vals[v.idx(i)] /= dk;

    return okay;
  }

  /// \brief defer an secondary entry
  /// \tparam CsType either \ref CRS (U) or \ref CCS (L)
  /// \tparam PosArray position array, see \ref Array
  /// \param[in] to_idx deferred destination
  /// \param[in] start starting positions
  /// \param[in,out] T lower or upper factors
  /// \param[in,out] list linked list of primary indices
  /// \ingroup defer
  template <class CsType, class PosArray>
  inline void defer_entry(const size_type to_idx, const PosArray &start,
                          CsType &T, PosArray &list) const {
    using index_type                = typename PosArray::value_type;
    constexpr static index_type nil = static_cast<index_type>(-1);

    hilucsi_assert(list[to_idx] == nil, "deferred location %zd must be nil",
                   to_idx);

    index_type idx = list[deferred_step()];
    while (idx != nil) {
      auto itr = T.ind_begin(idx) + start[idx];
      hilucsi_assert(itr != T.ind_end(idx), "fatal");
      const size_type n  = T.ind_end(idx) - itr,
                      vp = T.ind_start()[idx] + start[idx];
      *itr               = to_idx;  // set deferred index
      rotate_left(n, vp, T.inds());
      rotate_left(n, vp, T.vals());
      // update linked list
      itr                  = T.ind_begin(idx) + start[idx];
      const size_type j    = *itr;
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
  /// \ingroup defer
  template <class CcsType, class PosArray>
  inline void defer_L_and_fix_offsets(const size_type to_idx,
                                      const PosArray &L_start, CcsType &L,
                                      PosArray &L_list,
                                      PosArray &L_offsets) const {
    static_assert(!CcsType::ROW_MAJOR, "must be CCS");
    using index_type                = typename PosArray::value_type;
    constexpr static index_type nil = static_cast<index_type>(-1);

    index_type idx = L_list[deferred_step()];
    while (idx != nil) {
      auto            itr = L.row_ind_begin(idx) + L_start[idx];
      const size_type n   = L.row_ind_end(idx) - itr,
                      vp  = L.col_start()[idx] + L_start[idx];
      *itr                = to_idx;
      rotate_left(n, vp, L.inds());
      rotate_left(n, vp, L.vals());
      itr                  = L.row_ind_begin(idx) + L_start[idx];
      const size_type j    = *itr;
      const auto      next = L_list[idx];
      L_list[idx]          = L_list[j];
      L_list[j]            = idx;
      --L_offsets[idx];
      idx = next;
    }
    L_list[_step] = nil;
  }

 protected:
  /// \brief load a row of A to ut buffer
  /// \tparam ScaleArray scaling from left/right-hand sides, see \ref Array
  /// \tparam CrsType crs matrix of input A, see \ref CRS
  /// \tparam PermType permutation vector type, see \ref BiPermMatrix
  /// \tparam SpVecType sparse vector type, see \ref SparseVector
  /// \param[in] s row scaling vector
  /// \param[in] crs_A input matrix in CRS scheme
  /// \param[in] t column scaling vector
  /// \param[in] pk row permuted index
  /// \param[in] q_inv column inverse permutation matrix
  /// \param[out] ut output sparse vector of row vector for A
  /// \sa _load_A2l
  template <class ScaleArray, class CrsType, class PermType, class SpVecType>
  inline void _load_A2ut(const ScaleArray &s, const CrsType &crs_A,
                         const ScaleArray &t, const size_type &pk,
                         const PermType &q_inv, SpVecType &ut) const {
    // ut should be empty
    hilucsi_assert(ut.empty(), "ut should be empty while loading A");
    const size_type defer_thres = deferred_step();
    // pk is c index
    auto       v_itr = crs_A.val_cbegin(pk);
    auto       i_itr = crs_A.col_ind_cbegin(pk);
    const auto s_pk  = s[pk];
    for (auto last = crs_A.col_ind_cend(pk); i_itr != last; ++i_itr, ++v_itr) {
      const auto      A_idx = *i_itr;
      const size_type idx   = q_inv[A_idx];
      if (idx > defer_thres) {
        // get the gapped index
#ifndef NDEBUG
        const bool val_must_not_exit =
#endif
            ut.push_back(idx, _step);
        hilucsi_assert(val_must_not_exit,
                       "see prefix, failed on Crout step %zd for ut", _step);
        ut.vals()[idx] = s_pk * *v_itr * t[A_idx];  // scale here
      }
    }
  }

  /// \brief load a column of A to l buffer
  /// \tparam IsSymm if \a true, then only load the offset
  /// \tparam ScaleArray scaling from left/right-hand sides, see \ref Array
  /// \tparam CcsType ccs matrix of input A, see \ref CCS
  /// \tparam PermType permutation vector type, see \ref BiPermMatrix
  /// \tparam SpVecType sparse vector type, see \ref SparseVector
  /// \param[in] s row scaling vector
  /// \param[in] ccs_A input matrix in CCS scheme
  /// \param[in] t column scaling vector
  /// \param[in] p_inv row permutation matrix
  /// \param[in] qk permuted column index
  /// \param[in] m leading size
  /// \param[out] l output sparse vector of column vector for A
  /// \sa _load_A2ut
  template <bool IsSymm, class ScaleArray, class CcsType, class PermType,
            class SpVecType>
  inline void _load_A2l(const ScaleArray &s, const CcsType &ccs_A,
                        const ScaleArray &t, const PermType &p_inv,
                        const size_type &qk, const size_type m,
                        SpVecType &l) const {
    // runtime
    hilucsi_assert(l.empty(), "l should be empty while loading A");
    const size_type defer_thres = deferred_step();
    const size_type thres       = IsSymm ? m - 1 : defer_thres;
    // qk is c index
    auto       v_itr = ccs_A.val_cbegin(qk);
    auto       i_itr = ccs_A.row_ind_cbegin(qk);
    const auto t_qk  = t[qk];
    for (auto last = ccs_A.row_ind_cend(qk); i_itr != last; ++i_itr, ++v_itr) {
      const auto      A_idx = *i_itr;
      const size_type idx   = p_inv[A_idx];
      // push to the sparse vector only if its in range _step+1:n
      if (idx > thres) {
#ifndef NDEBUG
        const bool val_must_not_exit =
#endif
            l.push_back(idx, _step);
        hilucsi_assert(val_must_not_exit,
                       "see prefix, failed on Crout step %zd for l", _step);
        l.vals()[idx] = s[A_idx] * *v_itr * t_qk;  // scale here
      }
    }
  }

 protected:
  size_type _step;    ///< current step
  size_type _defers;  ///< deferring counter
};
}  // namespace hilucsi

#endif  // _HILUCSI_ALG_CROUT_HPP