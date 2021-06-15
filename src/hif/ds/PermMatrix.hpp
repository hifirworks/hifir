///////////////////////////////////////////////////////////////////////////////
//                  This file is part of the HIFIR library                   //
///////////////////////////////////////////////////////////////////////////////

/*!
 * \file hif/ds/PermMatrix.hpp
 * \brief Permutation matrix/mapping
 * \author Qiao Chen

\verbatim
Copyright (C) 2021 NumGeom Group at Stony Brook University

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
\endverbatim

 */

#ifndef _HIF_DS_PERMUTATIONMATRIX_HPP
#define _HIF_DS_PERMUTATIONMATRIX_HPP

#include <algorithm>

#include "hif/ds/Array.hpp"

namespace hif {

/*!
 * \addtogroup ds
 * @{
 */

/// \class PermMatrix
/// \tparam IndexType index type used, e.g. \a int
/// \brief permutation matrix defined as the P(i) entry in original matrix is
///        the i-th entry in the permutated one
template <class IndexType>
class PermMatrix {
 public:
  typedef IndexType                             index_type;   ///< index type
  typedef Array<index_type>                     iarray_type;  ///< index array
  typedef typename iarray_type::size_type       size_type;    ///< size
  typedef typename iarray_type::reference       reference;    ///< reference
  typedef typename iarray_type::const_reference const_reference;
  ///< constant reference

  /// \brief default constructor
  PermMatrix() = default;

  /// \brief constructor with own data
  /// \param[in] n size of permutation vector
  explicit PermMatrix(const size_type n) : _p(n) {}

  /// \brief clone of wrap external vector
  /// \param[in] p external permutation vector
  /// \param[in] clone if \a false (default), do shallow copy
  explicit PermMatrix(const iarray_type &p, bool clone = false)
      : _p(p, clone) {}

  /// \brief deep- or shallow copy constructor
  /// \param[in] other another permutation vector
  /// \param[in] clone if \a false (default), do shallow copy
  PermMatrix(const PermMatrix &other, bool clone = false)
      : _p(other._p, clone) {}

  // default stuffs
  PermMatrix(PermMatrix &&) = default;
  PermMatrix &operator=(const PermMatrix &) = default;
  PermMatrix &operator=(PermMatrix &&) = default;

  /// \brief resize the permutation vector
  /// \param[in] n size
  inline void resize(const size_type n) { _p.resize(n); }

  /// \brief get the size of permutation vector
  inline size_type size() const { return _p.size(); }

  /// \brief swap entries i and j
  /// \param[in] i entry i
  /// \param[in] j entry j
  inline void interchange(const size_type i, const size_type j) {
    std::swap(_p[i], _p[j]);
  }

  /// \brief make an identity permutation
  inline void make_eye() {
    const size_type n = _p.size();
    for (size_type i = 0u; i < n; ++i) _p[i] = i;
  }

  /// \brief check if identity mapping
  inline bool is_eye() const {
    return size() ? _p.front() == 0 && std::is_sorted(_p.cbegin(), _p.cend())
                  : true;
  }

  /// \brief get the permutation entry
  /// \param[in] i i-th permutation entry
  inline reference operator[](const size_type i) { return _p[i]; }

  /// \brief get the permutation entry, constant version
  /// \param[in] i i-th permutation entry
  inline const_reference operator[](const size_type i) const { return _p[i]; }

  /// \brief get the underlaying data
  inline iarray_type &operator()() { return _p; }

  /// \brief get the underlaying data, constant reference
  inline const iarray_type &operator()() const { return _p; }

 protected:
  iarray_type _p;  ///< underlying permutation array
};

/// \class BiPermMatrix
/// \tparam IndexType index type used, e.g. \a int
/// \brief This is the bi-directional version, i.e. the inverse mapping is avail
template <class IndexType>
class BiPermMatrix : public PermMatrix<IndexType> {
  using _base = PermMatrix<IndexType>;

 public:
  typedef typename _base::size_type       size_type;        ///< size
  typedef typename _base::reference       reference;        ///< reference
  typedef typename _base::const_reference const_reference;  ///< const ref
  typedef typename _base::iarray_type     iarray_type;      ///< index array

  /// \brief default constructor
  BiPermMatrix() = default;

  /// \brief constructor with own data
  /// \param[in] n vector size
  explicit BiPermMatrix(const size_type n) : _base(n), _p_inv(n) {}

  /// \brief constructor with external permutation vector
  /// \param[in] p permutation vector
  /// \param[in] clone if \a false (default), do shallow copy of \a p
  /// \note the inverse mapping is allocated explicitly
  explicit BiPermMatrix(const iarray_type &p, bool clone = false)
      : _base(p, clone), _p_inv(_p.size()) {}

  /// \brief constructor with external permutation matrix
  /// \param[in] p permutation matrix
  /// \param[in] clone if \a false (default), do shallow copy of \a p
  /// \note the inverse mapping is allocated explicitly
  explicit BiPermMatrix(const _base &p, bool clone = false)
      : _base(p, clone), _p_inv(_p.size()) {}

  /// \brief shallow of deep copy constructor
  /// \param[in] other another bi-permutation
  /// \param[in] clone if \a false (default), do shallow copy of \a p
  BiPermMatrix(const BiPermMatrix &other, bool clone = false)
      : _base(other, clone), _p_inv(other._p_inv, clone) {}

  // default stuffs
  BiPermMatrix(BiPermMatrix &&) = default;
  BiPermMatrix &operator=(const BiPermMatrix &) = default;
  BiPermMatrix &operator=(BiPermMatrix &&) = default;

  /// \brief get the inverse permutation mapping
  /// \param[in] i i-th inverse entry
  inline reference inv(const size_type i) { return _p_inv[i]; }

  /// \brief get the inverse permutation mapping, constant version
  /// \param[in] i i-th inverse entry
  inline const_reference inv(const size_type i) const { return _p_inv[i]; }
  using _base::          operator[];

  /// \brief resize the permutation vectors
  /// \param[in] n new size
  inline void resize(const size_type n) {
    _p.resize(n);
    _p_inv.resize(n);
  }

  /// \brief make identity permutation and its inverse mapping
  inline void make_eye() {
    hif_assert(_p.size() == _p_inv.size(), "nonmatching sizes");
    _base::make_eye();
    std::copy(_p.cbegin(), _p.cend(), _p_inv.begin());
  }

  /// \brief build the inverse mapping
  /// \note Assume that we have a permutation vector that has been setup
  /// \note Complexity is linear, i.e. \f$\mathcal{O}(n)\f$, to the size
  inline void build_inv() {
    hif_assert(_p.size() == _p_inv.size(), "nonmatching sizes");
    const size_type n = _p.size();
    for (size_type i = 0u; i < n; ++i) _p_inv[_p[i]] = i;
  }

  /// \brief swap two entries and their corresponding inverse mappings
  /// \param[in] i i-th entry
  /// \param[in] j j-th entry
  inline void interchange(const size_type i, const size_type j) {
    _base::interchange(i, j);
    std::swap(_p_inv[_p[i]], _p_inv[_p[j]]);
  }

  /// \brief swap two inverse entries
  /// \param[in] i i-th entry
  /// \param[in] j j-th entry
  inline void interchange_inv(const size_type i, const size_type j) {
    std::swap(_p_inv[i], _p_inv[j]);
    std::swap(_p[_p_inv[i]], _p[_p_inv[j]]);
  }

  /// \brief get the underlaying data
  inline iarray_type &inv() { return _p_inv; }

  /// \brief get the underlaying data, constant reference
  inline const iarray_type &inv() const { return _p_inv; }

 protected:
  using _base::_p;
  iarray_type _p_inv;  ///< inverse mapping array
};

/*!
 * @}
 */ // group ds

}  // namespace hif

#endif  // _HIF_DS_PERMUTATION_HPP
