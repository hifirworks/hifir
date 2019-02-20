//@HEADER
//----------------------------------------------------------------------------
//                Copyright (C) 2019 The PSMILU AUTHORS
//----------------------------------------------------------------------------
//@HEADER

/// \file psmilu_Prec.hpp
/// \brief Multilevel preconditioner interface
/// \authors Qiao,

#ifndef _PSMILU_PREC_HPP
#define _PSMILU_PREC_HPP

#include <list>
#include <utility>

#include "psmilu_CompressedStorage.hpp"
#include "psmilu_small_scale/solver.hpp"

namespace psmilu {

/// \class Prec
/// \brief A single level preconditioner
/// \tparam ValueType value data type, e.g. \a double
/// \tparam IndexType index data type, e.g. \a int
/// \tparam OneBased if \a false (default), using C-based index system
/// \tparam SSSType default is LU with partial pivoting
/// \ingroup cpp
template <class ValueType, class IndexType, bool OneBased = false,
          SmallScaleType SSSType = SMALLSCALE_LUP>
struct Prec {
  typedef ValueType                             value_type;  ///< value type
  typedef IndexType                             index_type;  ///< index type
  typedef CRS<value_type, index_type, OneBased> crs_type;    ///< crs type
  typedef CCS<value_type, index_type, OneBased> ccs_type;    ///< ccs type
  typedef Array<index_type>                     perm_type;   ///< permutation
  typedef typename ccs_type::size_type          size_type;   ///< size
  typedef typename ccs_type::array_type         array_type;  ///< array

  static constexpr bool ONE_BASED  = OneBased;  ///< c index flag
  static constexpr char EMPTY_PREC = '\0';      ///< empty prec

 private:
  typedef SmallScaleSolverTrait<SSSType> _sss_trait;  ///< small scale trait
 public:
  typedef typename _sss_trait::template solver_type<value_type> sss_solver_type;
  ///< small scaled solver type

  /// \brief default constructor
  Prec() = default;

  /// \brief empty copy constructor
  Prec(const Prec &) {}

  /// \brief rvalue reference constructor
  /// \param[in] mm leading size
  /// \param[in] nn system size
  /// \param[in] L_b lower part
  /// \param[in] d_b diagonal
  /// \param[in] U_b upper part
  /// \param[in] e E part
  /// \param[in] f F part
  /// \param[in] S row scaling
  /// \param[in] T column scaling
  /// \param[in] P row permutation
  /// \param[in] Q_inv inverse column permutation
  /// \note This allows us to use emplace back in STL efficiently
  Prec(size_type mm, size_type nn, ccs_type &&L_b, array_type &&d_b,
       ccs_type &&U_b, ccs_type &&e, ccs_type &&f, array_type &&S,
       array_type &&T, perm_type &&P, perm_type &&Q_inv)
      : m(mm),
        n(nn),
        L_B(std::move(L_b)),
        d_B(std::move(d_b)),
        U_B(std::move(U_b)),
        E(std::move(e)),
        F(std::move(f)),
        s(std::move(S)),
        t(std::move(T)),
        p(std::move(P)),
        q_inv(std::move(Q_inv)) {}

  /// \brief enable explicitly calling move
  /// \param[in,out] L_b lower part
  /// \param[in,out] d_b diagonal
  /// \param[in,out] U_b upper part
  /// \param[in,out] e E part
  /// \param[in,out] f F part
  /// \param[in,out] S row scaling
  /// \param[in,out] T column scaling
  /// \param[in,out] P row permutation
  /// \param[in,out] Q_inv inverse column permutation
  /// \note Sizes are not included, assign them explicitly.
  ///
  /// we pass lvalue reference, but will explicitly destroy all input arguments
  /// on output. This allows one to avoid writing a ton of \a std::move while
  /// calling this routine.
  ///
  /// \warning Everything on output is destroyed, as the routine name says.
  inline void move_destroy(ccs_type &L_b, array_type &d_b, ccs_type &U_b,
                           ccs_type &e, ccs_type &f, array_type &S,
                           array_type &T, perm_type &P, perm_type &Q_inv) {
    L_B   = std::move(L_b);
    d_B   = std::move(d_b);
    U_B   = std::move(U_b);
    E     = std::move(e);
    F     = std::move(f);
    s     = std::move(S);
    t     = std::move(T);
    p     = std::move(P);
    q_inv = std::move(Q_inv);
  }

  /// \brief check if this a last level preconditioner
  ///
  /// The idea is first check if the \ref dense_solver is empty or not; be aware
  /// that there is another situation that we should treat the preconditioner
  /// is last level---if \ref m is equal to \ref n.
  ///
  /// \note Currently, we test m == n, which is fine for squared systems.
  inline bool is_last_level() const { return !dense_solver.empty() || m == n; }

  size_type       m;             ///< leading block size
  size_type       n;             ///< system size
  ccs_type        L_B;           ///< lower part of leading block
  array_type      d_B;           ///< diagonal block of leading block
  ccs_type        U_B;           ///< upper part of leading block
  ccs_type        E;             ///< scaled and permutated E part
  ccs_type        F;             ///< scaled and permutated F part
  array_type      s;             ///< row scaling vector
  array_type      t;             ///< column scaling vector
  perm_type       p;             ///< row permutation matrix
  perm_type       q_inv;         ///< column inverse permutation matrix
  sss_solver_type dense_solver;  ///< dense decomposition

 private:
  /// \brief allow casting to \a char, this is needed to add an empty node
  Prec(char) {}

  friend class std::list<Prec>;  // give list friendship
};

/// \typedef Precs
/// \brief multilevel preconditioners
/// \tparam ValueType value data type, e.g. \a double
/// \tparam IndexType index data type, e.g. \a int
/// \tparam OneBased if \a false (default), using C-based index system
/// \tparam SSSType default is LU with partial pivoting
/// \ingroup itr
///
/// We choose to use STL list because adding new node is constant time without
/// bothering copying. It's worth noting that querying the size of \a list has
/// become constant since C++11. Also, with the constructors in \ref Prec, the
/// following two ways can be used to construct list:
///
/// \code{.cpp}
/// using precs_t = Precs<double, int>;
/// using prec_t = precs_t::value_type;
/// precs_t precs;
/// precs.emplace_back(m, n, /* rvalue references */);
/// \endcode
///
/// and
///
/// \code{.cpp}
/// using precs_t = Precs<double, int>;
/// using prec_t = precs_t::value_type;
/// precs_t precs;
/// precs.push_back(prec_t::EMPTY_PREC);
/// auto &prec = precs.back();
/// prec.move_destroy(/* lvalue references */);
/// prec.m = m;
/// prec.n = n;
/// \endcode
///
/// These fit into the usage for Precs. Notice that for the second way, after
/// calling \ref Prec::move_destroy, all input arguments will be destroyed.
template <class ValueType, class IndexType, bool OneBased = false,
          SmallScaleType SSSType = SMALLSCALE_LUP>
using Precs = std::list<Prec<ValueType, IndexType, OneBased, SSSType>>;

}  // namespace psmilu

#endif  // _PSMILU_PREC_HPP
