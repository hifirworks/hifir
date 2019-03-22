//@HEADER
//----------------------------------------------------------------------------
//                Copyright (C) 2019 The PSMILU AUTHORS
//----------------------------------------------------------------------------
//@HEADER

/// \file psmilu_MT/PrecPart.hpp
/// \brief Partitions for \ref Prec
/// \authors Qiao,

#ifndef _PSMILU_MT_PRECPART_HPP
#define _PSMILU_MT_PRECPART_HPP

#include <list>

#include "LevelSets.hpp"
#include "utils.hpp"

namespace psmilu {

/*!
 * \addtogroup mt
 * @{
 */

/// \class PrecPart
/// \brief data partition across different threads for a \ref PRec
/// \tparam IndexType index type, e.g. \a int
template <class IndexType>
struct PrecPart {
  typedef IndexType             index_type;      ///< index type
  typedef LevelSets<index_type> levelsets_type;  ///< type of level set

  /// \brief default constructor
  PrecPart() = default;

  /// \brief create partition for a level of preconditioner
  /// \tparam PrecType preconditioner type, see \ref Prec
  /// \param[in] prec a level of preconditioner
  /// \param[in] threads number of threads
  template <class PrecType>
  PrecPart(const PrecType &prec, const int threads)
      : L_B_ls(levelsets_type::template create_from<true>(prec.L_B)),
        U_B_ls(levelsets_type::template create_from<false>(prec.U_B)),
        n_parts(make_uni_parts(prec.n, threads)),
        m_parts(make_uni_parts(prec.m, threads)),
        nm_parts(make_uni_parts(prec.n - prec.m, threads)) {}

  // default stuff
  PrecPart(PrecPart &&) = default;
  PrecPart &operator=(PrecPart &&) = default;

  /// \brief check number of threads
  inline int threads() const { return m_parts.size(); }

  /// \brief check empty
  inline bool empty() const { return L_B_ls.empty(); }

  levelsets_type L_B_ls;    ///< level sets for L_B
  levelsets_type U_B_ls;    ///< level sets for U_B
  UniformParts   n_parts;   ///< partition with overall system size
  UniformParts   m_parts;   ///< partition with size leading block
  UniformParts   nm_parts;  ///< partition with size offset, e.g. rows of E
};

/// \typedef PrecParts
/// \brief Like \ref Precs, we use \a std::list to group different levels of
///        partitions for \ref Prec
/// \tparam IndexType index type, e.g. \a int
template <class IndexType>
using PrecParts = std::list<PrecPart<IndexType>>;

/*!
 * @}
 */ // group mt

}  // namespace psmilu

#endif  // _PSMILU_MT_PRECPART_HPP
