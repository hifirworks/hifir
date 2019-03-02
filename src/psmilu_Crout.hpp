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

  /// \brief dual of the \ref compute_l, i.e. computing the row of U
  /// \tparam LeftDiagType row scaling diagonal matrix type, see \ref Array
  /// \tparam A_CrsType crs format for input matrix, see \ref CRS
  /// \tparam RightDiagType column scaling diagonal matrix type, see \ref Array
  /// \tparam PermType permutation matrix type, see \ref BiPermMatrix
  /// \tparam L_AugCcsType augmented storage type for L, see \ref AugCCS
  /// \tparam DiagType diagonal matrix type, see \ref Array
  /// \tparam U_CrsType crs storage format for U, see \ref AugCRS
  /// \tparam U_StartType array type storing U_start, see \ref Array
  /// \tparam SpVecType work array for current row vector, see \ref SparseVector
  /// \param[in] s row scaling matrix from preprocessing
  /// \param[in] crs_A input matrix in crs scheme
  /// \param[in] t column scaling matrix from preprocessing
  /// \param[in] pk permutated row index
  /// \param[in] q column permutation matrix (right-hand side)
  /// \param[in] L lower part
  /// \param[in] d diagonal vector
  /// \param[in] U upper part
  /// \param[in] U_start leading row positions of current \ref _step
  /// \param[out] ut current row vector of U
  /// \note For arrays, this routine is compatible with both \ref Array and
  ///       \a std::vector
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
  template <class LeftDiagType, class A_CrsType, class RightDiagType,
            class PermType, class U_CrsType, class U_StartType, class DiagType,
            class L_AugCcsType, class SpVecType>
  void compute_ut(const LeftDiagType &s, const A_CrsType &crs_A,
                  const RightDiagType &t, const size_type pk, const PermType &q,
                  const L_AugCcsType &L, const DiagType &d, const U_CrsType &U,
                  const U_StartType &U_start, SpVecType &ut) {
    // compilation checking
    static_assert(A_CrsType::ROW_MAJOR, "input A must be CRS for loading ut");
    static_assert(!(A_CrsType::ONE_BASED ^ L_AugCcsType::ONE_BASED),
                  "ionconsistent one-based");
    static_assert(!(A_CrsType::ONE_BASED ^ U_CrsType::ONE_BASED),
                  "ionconsistent one-based");
    using index_type           = typename L_AugCcsType::index_type;
    constexpr static bool base = A_CrsType::ONE_BASED;

    //------------
    // run time
    //------------

    // clear sparse buffer
    ut.reset_counter();

    // first load the A
    _load_A2ut(s, crs_A, t, pk, q, ut);

    // if not the first step
    // compute -L(k,1:k-1)d(1:k-1,1:k-1)U(1:k-1,k+1:n)
    if (_step) {
      // get leading value and index iterators for U
      auto U_v_first = U.vals().cbegin();
      auto U_i_first = U.col_ind().cbegin();
      // get starting row ID
      index_type aug_id = L.start_row_id(_step);
      while (!L.is_nil(aug_id)) {
        // get the column index (C-based)
        // NOTE the column index is that of the row in U
        const index_type c_idx = L.col_idx(aug_id);
        psmilu_assert(
            (size_type)c_idx < _step,
            "compute_ut column index %zd should not exceed step %zd for L",
            (size_type)c_idx, _step);
        psmilu_assert((size_type)c_idx < d.size(),
                      "%zd exceeds the diagonal vector size", (size_type)c_idx);
        psmilu_assert((size_type)c_idx < U_start.size(),
                      "%zd exceeds the U_start size", (size_type)c_idx);
        // NOTE once we drop val-pos impl, change this accordingly (L_start)
        // compute L*D
        const auto ld = L.val_from_row_id(aug_id) * d[c_idx];
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
  }

  /// \brief compute the column vector of L of current step
  /// \tparam IsSymm if \a true, then a symmetric leading block is assumed
  /// \tparam LeftDiagType row scaling diagonal matrix type, see \ref Array
  /// \tparam A_CcsType ccs format of input A, see \ref CCS
  /// \tparam RightDiagType column scaling diagonal matrix type, see \ref Array
  /// \tparam PermType permutation matrix type, see \ref BiPermMatrix
  /// \tparam L_CcsType ccs format of L matrix, see \ref AugCCS
  /// \tparam L_StartType array used for storing L_start, see \ref Array
  /// \tparam DiagType array used for storing diagonal array, see \ref Array
  /// \tparam U_AugCrsType augmented crs for U matrix, see \ref AugCRS
  /// \tparam SpVecType sparse vector for storing l, see \ref SparseVector
  /// \param[in] s row scaling matrix from preprocessing
  /// \param[in] ccs_A input matrix in ccs scheme
  /// \param[in] t column scaling matrix from preprocessing
  /// \param[in] p row permutation matrix
  /// \param[in] qk permutated column index
  /// \param[in] m leading block size
  /// \param[in] L lower part of decomposition
  /// \param[in] L_start position array storing the starting locations of \a L
  /// \param[in] d diagonal entries
  /// \param[in] U augmented U matrix
  /// \param[out] l column vector of L at current \ref _step
  /// \note For both \a d and \a L_start, \a std::vector can be used
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
  template <bool IsSymm, class LeftDiagType, class A_CcsType,
            class RightDiagType, class PermType, class L_CcsType,
            class L_StartType, class DiagType, class U_AugCrsType,
            class SpVecType>
  inline void compute_l(const LeftDiagType &s, const A_CcsType &ccs_A,
                        const RightDiagType &t, const PermType &p,
                        const size_type qk, const size_type m,
                        const L_CcsType &L, const L_StartType &L_start,
                        const DiagType &d, const U_AugCrsType &U,
                        SpVecType &l) {
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

    // clear sparse buffer
    l.reset_counter();

    // first load the A
    _load_A2l<IsSymm>(s, ccs_A, t, p, qk, m, l);
    // if not the first step
    // compute -L(k+1:n,1:k-1)d(1:k-1,1:k-1)U(1:k-1,k)
    if (_step) {
      // get the leading value and index iterators for L
      auto       L_v_first = L.vals().cbegin();
      auto       L_i_first = L.row_ind().cbegin();
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
        const auto du = d[r_idx] * U.val_from_col_id(aug_id);
        // get the starting position from L_start
        auto L_v_itr  = L_v_first + L_start[r_idx];
        auto L_i_itr  = L_i_first + L_start[r_idx],
             L_i_last = L.row_ind_cend(r_idx);
        // for this local range
        for (; L_i_itr != L_i_last; ++L_i_itr, ++L_v_itr) {
          // convert to c index
          const auto c_idx = to_c_idx<size_type, base>(*L_i_itr);
#ifndef NDEBUG
          if (IsSymm)
            psmilu_error_if(
                c_idx < _step,
                "%zd step symmetric l should not contain upper part (%zd)",
                _step, c_idx);
#endif
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

  /// \brief scale the computed row/column vectors by current diagonal inverse
  /// \tparam DiagType diagonal matrix type, see \ref Array
  /// \tparam SpVecType sparse vector for storing l/U, see \ref SparseVector
  /// \param[in] d diagonal vector
  /// \param[in,out] v vector to be scaled
  /// \return The returned boolean flag indices a singularity occurs if \a true
  ///
  /// This routine computes \f$\boldsymbol{v}=\boldsymbol{v}/d_k\f$; firstly,
  /// singularity is checked to ensure the validation of computation. Then,
  /// we test if it's safe to invert \f$d_k\f$ so that we can use multiplication
  /// instead of division.
  ///
  /// Regarding complexity, this routine takes
  /// \f$\mathcal{O}(\textrm{nnz}(\boldsymbol{v}))\f$
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
      for (size_type i = 0u; i < n; ++i) vals[v.c_idx(i)] *= dk_inv;
    } else
      for (size_type i = 0u; i < n; ++i) vals[v.c_idx(i)] /= dk;

    return okay;
  }

  /// \brief updating position array for leading column locations in U
  /// \tparam U_AugCrsType augmented crs for U matrix, see \ref AugCRS
  /// \tparam U_StartType array type for U_start, see \ref Array
  /// \param[in] U upper part
  /// \param[in,out] U_start array storing the leading locations
  /// \note Complexity: \f$\mathcal{O}(\textrm{nnz}(\boldsymbol{U}(:,k)))\f$
  template <class U_AugCrsType, class U_StartType>
  inline void update_U_start(const U_AugCrsType &U,
                             U_StartType &       U_start) const {
    static_assert(U_AugCrsType::ROW_MAJOR, "U must be AugCRS");
    using index_type                = typename U_AugCrsType::index_type;
    constexpr static bool ONE_BASED = U_AugCrsType::ONE_BASED;
    const auto            c_idx     = [](const size_type i) {
      return to_c_idx<size_type, ONE_BASED>(i);
    };

    if (!_step) return;

    // we just added a new row to U, which is _step
    // we, then, assign the starting ID to be its start_ind
    U_start[_step - 1] = U.row_start()[_step - 1] - ONE_BASED;
    // get the aug handle
    index_type aug_id = U.start_col_id(_step);
    // loop thru current column, O(u_k)
    while (!U.is_nil(aug_id)) {
      // get the row index, C based
      const index_type row = U.row_idx(aug_id);
      if (U_start[row] < U.row_start()[row + 1] - ONE_BASED &&
          c_idx(*(U.col_ind().cbegin() + U_start[row])) <= _step)
        ++U_start[row];
      // advance augmented handle
      aug_id = U.next_col_id(aug_id);
    }
  }

  /// \brief updating position array for leading row locations in L
  /// \tparam IsSymm if \a true, then assuming a symmetric leading block
  /// \tparam L_AugCcsType augmented ccs type for L, see \ref AugCCS
  /// \tparam L_StartType array type storing leading positions, see \ref Array
  /// \param[in] L lower part
  /// \param[in,out] L_start leading location array
  /// \note Complexity: \f$\mathcal{O}(\textrm{nnz}(\boldsymbol{L}(k,:)))\f$
  ///
  /// This routine is \a SFINAE-able by \a IsSymm, where this is for \a false
  /// cases, i.e. no leading symmetric block
  template <bool IsSymm, class L_AugCcsType, class L_StartType>
  inline typename std::enable_if<!IsSymm>::type update_L_start(
      const L_AugCcsType &L, const size_type /* m */, L_StartType &L_start,
      bool /* no_pivot */ = false) const {
    static_assert(!L_AugCcsType::ROW_MAJOR, "L must be AugCCS");
    using index_type                = typename L_AugCcsType::index_type;
    constexpr static bool ONE_BASED = L_AugCcsType::ONE_BASED;
    const auto            c_idx     = [](const size_type i) {
      return to_c_idx<size_type, ONE_BASED>(i);
    };

    if (!_step) return;

    // we just added a new column to L, which is _step
    // we, then, assign the starting ID should be its start_ind
    L_start[_step - 1] = L.col_start()[_step - 1] - ONE_BASED;
    // get the aug handle
    index_type aug_id = L.start_row_id(_step);
    // loop through current row, O(l_k')
    while (!L.is_nil(aug_id)) {
      // get the column index, C based
      const index_type col = L.col_idx(aug_id);
      // for each of this starting inds, advance one
      if (L_start[col] < L.col_start()[col + 1] - ONE_BASED &&
          c_idx(*(L.row_ind().cbegin() + L_start[col])) <= _step)
        ++L_start[col];
      // advance augmented handle
      aug_id = L.next_row_id(aug_id);
    }
  }

  /// \brief updating position array for leading row locations in L
  /// \tparam IsSymm if \a true, then assuming a symmetric leading block
  /// \tparam L_AugCcsType augmented ccs type for L, see \ref AugCCS
  /// \tparam L_StartType array type storing leading positions, see \ref Array
  /// \param[in] L lower part
  /// \param[in] m leading block size
  /// \param[in,out] L_start leading location array
  /// \note Complexity: \f$\mathcal{O}(\textrm{nnz}(\boldsymbol{L}(k,:)))\f$
  ///
  /// This routine is \a SFINAE-able by \a IsSymm, where this is for \a true
  /// cases. For the symmetric case, we just need to search the starting
  /// position for the newly added column. However, of course, this is not
  /// enough, we also need to update the entries whenever we perform the row
  /// and column interchanages, which can't be done in this routine.
  template <bool IsSymm, class L_AugCcsType, class L_StartType>
  inline typename std::enable_if<IsSymm>::type update_L_start(
      const L_AugCcsType &L, const size_type m, L_StartType &L_start,
      bool = false) const {
    static_assert(!L_AugCcsType::ROW_MAJOR, "L must be AugCCS");
    using index_type                = typename L_AugCcsType::index_type;
    constexpr static bool ONE_BASED = L_AugCcsType::ONE_BASED;
    const auto            ori_idx   = [](const size_type i) {
      return to_ori_idx<size_type, ONE_BASED>(i);
    };

    psmilu_assert(m, "cannot have empty leading block for symmetric case!");

    if (!_step) return;

    // NOTE that from step to step, m can only decrease!

    if (m >= L.nrows()) {
      // if we have leading block that indicates fully symmetric system
      // then the ending position is stored
      // col_start has an extra element, thus this is valid accessing
      // or maybe we can use {Array,vector}::back??
      L_start[_step - 1] = L.col_start()[_step];
      return;
    }
    // binary search to point to start of newly added row
    auto info          = find_sorted(L.row_ind_cbegin(_step - 1),
                            L.row_ind_cend(_step - 1), ori_idx(m));
    L_start[_step - 1] = info.second - L.row_ind().cbegin();
    return;
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
  ///   \boldsymbol{D}_{k+1:m,k+1:m}=\boldsymbol{D}_{k+1:m,k+1:m}-
  ///     \boldsymbol{D}_{k,k}
  ///     \left(\boldsymbol{\ell}_k\otimes\boldsymbol{u}_k\right)_{k+1:m,k+1:m}
  /// \f]
  ///
  /// The complexity is bounded by:
  ///
  /// \f[
  ///   \mathcal{O}(\min(\textrm{nnz}(\boldsymbol{\ell}_k),
  ///     \textrm{nnz}(\boldsymbol{u}_k)))
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

    // get the current diagonal entry
    const auto dk = d[_step];
    if (ut.size() <= l.size()) {
      const auto &    l_d_tags = static_cast<const extractor &>(l).dense_tags();
      const size_type n        = ut.size();
      const auto &    l_vals   = l.vals();
      for (size_type i = 0u; i < n; ++i) {
        const size_type c_idx = ut.c_idx(i);
        psmilu_assert(
            c_idx > _step,
            "should only contain the upper part of ut, (c_idx,step)=(%zd,%zd)!",
            c_idx, _step);
        // if the dense tags of this entry points to this step, we know l has
        // an element in this slot
        if (c_idx < m && (size_type)l_d_tags[c_idx] == _step)
          d[c_idx] -= dk * ut.val(i) * l_vals[c_idx];
      }
    } else {
      const auto &ut_d_tags  = static_cast<const extractor &>(ut).dense_tags();
      const size_type n      = l.size();
      const auto &    u_vals = ut.vals();
      for (size_type i = 0u; i < n; ++i) {
        const size_type c_idx = l.c_idx(i);
        psmilu_assert(
            c_idx > _step,
            "should only contain the lower part of l, (c_idx,step)=(%zd,%zd)!",
            c_idx, _step);
        // if the dense tags of this entry points to this step, we know ut has
        // an element in this slot
        if (c_idx < m && (size_type)ut_d_tags[c_idx] == _step)
          d[c_idx] -= dk * u_vals[c_idx] * l.val(i);
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
  ///   \boldsymbol{D}_{k+1:m,k+1:m}=\boldsymbol{D}_{k+1:m,k+1:m}-
  ///     \boldsymbol{D}_{k,k}
  ///     \left(\boldsymbol{\ell}_k\otimes\boldsymbol{\u}_k\right)_{k+1:m,k+1:m}
  /// \f]
  ///
  /// The complexity is bounded by:
  ///
  /// \f[
  ///   \mathcal{O}(\textrm{nnz}(\boldsymbol{\u}_k))
  ///\f]
  template <bool IsSymm, class SpVecType, class DiagType>
  inline typename std::enable_if<IsSymm>::type update_B_diag(
      const SpVecType & /* l */, const SpVecType &ut, const size_type m,
      DiagType &d) const {
    psmilu_assert(m, "fatal, symmetric block cannot be empty!");
    // get the current diagonal entry
    const auto dk = d[_step];
    // NOTE that the diagonal is updated before doing any dropping/sorting,
    // thus the indices are, in general, unsorted
    const size_type n = ut.size();
    // get the c index
    for (size_type i = 0u; i < n; ++i) {
      const size_type c_idx = ut.c_idx(i);
      psmilu_assert(
          c_idx > _step,
          "ut should only contain the upper part, (c_idx,step)=(%zd,%zd)!",
          c_idx, _step);
      if (c_idx < m) d[c_idx] -= dk * ut.val(i) * ut.val(i);
    }
  }

 protected:
  /// \brief load a row of A to ut buffer
  /// \tparam LeftDiagType diagonal matrix from left-hand side, see \ref Array
  /// \tparam CrsType crs matrix of input A, see \ref CRS
  /// \tparam RightDiagType diagonal matrix from right-hand side
  /// \tparam PermType permutation vector type, see \ref BiPermMatrix
  /// \tparam SpVecType sparse vector type, see \ref SparseVector
  /// \param[in] s row scaling vector
  /// \param[in] crs_A input matrix in CRS scheme
  /// \param[in] t column scaling vector
  /// \param[in] pk permutated row index
  /// \param[in] q column permutation matrix
  /// \param[out] ut output sparse vector of row vector for U
  /// \note Complexity is \f$\mathcal{O}(\textrm{nnz}(\boldsymbol{A}(pk,:)))\f$
  /// \sa _load_A2l
  template <class LeftDiagType, class CrsType, class RightDiagType,
            class PermType, class SpVecType>
  inline void _load_A2ut(const LeftDiagType &s, const CrsType &crs_A,
                         const RightDiagType &t, const size_type pk,
                         const PermType &q, SpVecType &ut) const {
    // compilation consistency checking
    static_assert(!(CrsType::ONE_BASED ^ SpVecType::ONE_BASED),
                  "inconsistent one-based in ccs and sparse vector");
    constexpr static bool base = CrsType::ONE_BASED;
    // ut should be empty
    psmilu_assert(ut.empty(), "ut should be empty while loading A");
    // pk is c index
    auto       v_itr = crs_A.val_cbegin(pk);
    auto       i_itr = crs_A.col_ind_cbegin(pk);
    const auto s_pk  = s[pk];
    for (auto last = crs_A.col_ind_cend(pk); i_itr != last; ++i_itr, ++v_itr) {
      const auto A_idx = to_c_idx<size_type, base>(*i_itr);
      const auto c_idx = q.inv(A_idx);
      if ((size_type)c_idx > _step) {
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
  /// \tparam LeftDiagType diagonal matrix from left-hand side
  /// \tparam CcsType ccs matrix of input A, see \ref CCS
  /// \tparam RightDiagType diagonal matrix from right-hand side
  /// \tparam PermType permutation vector type, see \ref BiPermMatrix
  /// \tparam SpVecType sparse vector type, see \ref SparseVector
  /// \param[in] s row scaling vector
  /// \param[in] ccs_A input matrix in CCS scheme
  /// \param[in] t column scaling vector
  /// \param[in] p row permutation matrix
  /// \param[in] qk permuted column index
  /// \param[in] m leading size
  /// \param[out] l output sparse vector of column vector for L
  /// \note Complexity is \f$\mathcol{O}(\textrm{nnz}(\boldsymbol{A}(:,qk)))\f$
  /// \sa _load_A2ut
  template <bool IsSymm, class LeftDiagType, class CcsType, class RightDiagType,
            class PermType, class SpVecType>
  inline void _load_A2l(const LeftDiagType &s, const CcsType &ccs_A,
                        const RightDiagType &t, const PermType &p,
                        const size_type qk, const size_type m,
                        SpVecType &l) const {
    // compilation consistency checking
    static_assert(!(CcsType::ONE_BASED ^ SpVecType::ONE_BASED),
                  "inconsistent one-based in ccs and sparse vector");
    constexpr static bool base = CcsType::ONE_BASED;

    // runtime
    psmilu_assert(l.empty(), "l should be empty while loading A");
    const size_type thres = IsSymm ? m - 1u : _step;
    // qk is c index
    auto       v_itr = ccs_A.val_cbegin(qk);
    auto       i_itr = ccs_A.row_ind_cbegin(qk);
    const auto t_qk  = t[qk];
    for (auto last = ccs_A.row_ind_cend(qk); i_itr != last; ++i_itr, ++v_itr) {
      const auto A_idx = to_c_idx<size_type, base>(*i_itr);
      const auto c_idx = p.inv(A_idx);
      // push to the sparse vector only if its in range _step+1:n
      if ((size_type)c_idx > thres) {
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

 protected:
  mutable size_type _step;  ///< current step
};
}  // namespace psmilu

#endif  // _PSMILU_CROUT_HPP
