///////////////////////////////////////////////////////////////////////////////
//                This file is part of HILUCSI project                       //
///////////////////////////////////////////////////////////////////////////////

/*!
 * \file hilucsi/pre/rcm.hpp
 * \brief Kernel for reversed Cuthill-Mckee ordering
 * \authors Qiao,

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

// This file is a reorganization of the F2C results of the Fortran 77 reverse
// Cuthill-Mckee ordering method from
// "Computer Solution of Large Sparse Positive Definite Systems (chpt 4.4)"
// Author(s):
//  Qiao,

// Changes:
// 1. template-ize the interface so that no longer depending on f2c library
// 2. rename "integer" to "_Integer" as a template argument
// 3. use vector<bool> for "mask", the length must be n+1!!!

#ifndef _HILUCSI_PRE_RCM_HPP
#define _HILUCSI_PRE_RCM_HPP

#include <cstdlib>
#include <vector>

namespace hilucsi {
namespace rcm {
namespace detail {

/*!
 * \addtogroup pre
 * @{
 */

template <typename _Integer>
inline int rootls_(const _Integer root, const _Integer *xadj,
                   const _Integer *adjncy, std::vector<bool> &mask,
                   _Integer *nlvl, _Integer *xls, _Integer *ls) {
  _Integer i__, j, nbr, node, jstrt, lbegin, ccsize, lvlend, lvsize;
  --ls;
  --xls;
  --adjncy;
  --xadj;
  mask[root] = false;
  ls[1]      = root;
  *nlvl      = 0;
  lvlend     = 0;
  ccsize     = 1;
  for (;;) {
    lbegin = lvlend + 1;
    lvlend = ccsize;
    ++(*nlvl);
    xls[*nlvl]             = lbegin;
    const _Integer i__stop = lvlend;
    for (i__ = lbegin; i__ <= i__stop; ++i__) {
      node                 = ls[i__];
      jstrt                = xadj[node];
      const _Integer jstop = xadj[node + 1] - 1;
      if (jstop < jstrt) continue;
      for (j = jstrt; j <= jstop; ++j) {
        nbr = adjncy[j];
        if (!mask[nbr]) continue;
        ++ccsize;
        ls[ccsize] = nbr;
        mask[nbr]  = false;
      }
    }
    lvsize = ccsize - lvlend;
    if (lvsize > 0) continue;
    break;
  }
  xls[*nlvl + 1]         = lvlend + 1;
  const _Integer i__stop = ccsize;
  for (i__ = 1; i__ <= i__stop; ++i__) {
    node       = ls[i__];
    mask[node] = true;
  }
  return 0;
}

template <typename _Integer>
inline int fnroot_(_Integer *root, const _Integer *xadj, const _Integer *adjncy,
                   std::vector<bool> &mask, _Integer *nlvl, _Integer *xls,
                   _Integer *ls) {
  _Integer j, k, ndeg, node, jstrt, kstrt, mindeg, ccsize, nunlvl;
  --ls;
  --xls;
  --adjncy;
  --xadj;
  rootls_(*root, &xadj[1], &adjncy[1], mask, nlvl, &xls[1], &ls[1]);
  ccsize = xls[*nlvl + 1] - 1;
  if (*nlvl == 1 || *nlvl == ccsize) return 0;
  for (;;) {
    jstrt  = xls[*nlvl];
    mindeg = ccsize;
    *root  = ls[jstrt];
    if (ccsize != jstrt) {
      const _Integer jstop = ccsize;
      for (j = jstrt; j <= jstop; ++j) {
        node                 = ls[j];
        ndeg                 = 0;
        kstrt                = xadj[node];
        const _Integer kstop = xadj[node + 1] - 1;
        for (k = kstrt; k <= kstop; ++k) {
          if (mask[adjncy[k]]) ++ndeg;
        }
        if (ndeg >= mindeg) continue;
        *root  = node;
        mindeg = ndeg;
      }
    }
    rootls_(*root, &xadj[1], &adjncy[1], mask, &nunlvl, &xls[1], &ls[1]);
    if (nunlvl <= *nlvl) return 0;
    *nlvl = nunlvl;
    if (nunlvl < ccsize) continue;
    break;
  }
  return 0;
}

template <typename _Integer>
inline int degree_(const _Integer root, _Integer *xadj, const _Integer *adjncy,
                   std::vector<bool> &mask, _Integer *deg, _Integer *ccsize,
                   _Integer *ls) {
  _Integer i__, j, nbr, ideg, node, jstrt, lbegin, lvlend, lvsize;

  --ls;
  --deg;
  --adjncy;
  --xadj;
  ls[1]      = root;
  xadj[root] = -xadj[root];
  lvlend     = 0;
  *ccsize    = 1;
  for (;;) {
    lbegin                 = lvlend + 1;
    lvlend                 = *ccsize;
    const _Integer i__stop = lvlend;
    for (i__ = lbegin; i__ <= i__stop; ++i__) {
      node                 = ls[i__];
      jstrt                = -xadj[node];
      const _Integer jstop = std::abs(xadj[node + 1]) - 1;
      ideg                 = 0;
      if (jstop < jstrt) {
        deg[node] = ideg;
        continue;
      }
      for (j = jstrt; j <= jstop; ++j) {
        nbr = adjncy[j];
        if (!mask[nbr]) continue;
        ++ideg;
        if (xadj[nbr] < 0) continue;
        xadj[nbr] = -xadj[nbr];
        ++(*ccsize);
        ls[*ccsize] = nbr;
      }
      deg[node] = ideg;
    }
    lvsize = *ccsize - lvlend;
    if (lvsize > 0) continue;
    break;
  }
  const _Integer i__stop2 = *ccsize;
  for (i__ = 1; i__ <= i__stop2; ++i__) {
    node       = ls[i__];
    xadj[node] = -xadj[node];
  }
  return 0;
}

template <typename _Integer>
inline int rcm_(const _Integer root, _Integer *xadj, const _Integer *adjncy,
                std::vector<bool> &mask, _Integer *perm, _Integer *ccsize,
                _Integer *deg) {
  _Integer i__, j, k, l, nbr, node, fnbr, lnbr, lperm, jstrt;
  _Integer lbegin, lvlend;
  --deg;
  --perm;
  --adjncy;
  --xadj;
  degree_(root, &xadj[1], &adjncy[1], mask, &deg[1], ccsize, &perm[1]);
  mask[root] = false;
  if (*ccsize <= 1) return 0;
  lvlend = 0;
  lnbr   = 1;
  for (;;) {
    lbegin                 = lvlend + 1;
    lvlend                 = lnbr;
    const _Integer i__stop = lvlend;
    for (i__ = lbegin; i__ <= i__stop; ++i__) {
      node                 = perm[i__];
      jstrt                = xadj[node];
      const _Integer jstop = xadj[node + 1] - 1;
      fnbr                 = lnbr + 1;
      for (j = jstrt; j <= jstop; ++j) {
        nbr = adjncy[j];
        if (!mask[nbr]) continue;
        ++lnbr;
        mask[nbr]  = false;
        perm[lnbr] = nbr;
      }
      if (fnbr >= lnbr) continue;
      k = fnbr;
      for (;;) {
        l = k;
        ++k;
        nbr = perm[k];
        for (;;) {
          if (l < fnbr) break;
          lperm = perm[l];
          if (deg[lperm] <= deg[nbr]) break;
          perm[l + 1] = lperm;
          --l;
        }
        perm[l + 1] = nbr;
        if (k < lnbr) continue;
        break;
      }
    }
    if (lnbr > lvlend) continue;
    break;
  }
  k                       = *ccsize / 2;
  l                       = *ccsize;
  const _Integer i__stop2 = k;
  for (i__ = 1; i__ <= i__stop2; ++i__) {
    lperm     = perm[l];
    perm[l]   = perm[i__];
    perm[i__] = lperm;
    --l;
  }
  return 0;
}

template <typename _Integer>
inline int genrcm_(const _Integer neqns, _Integer *xadj, const _Integer *adjncy,
                   _Integer *perm, std::vector<bool> &mask, _Integer *xls) {
  _Integer i__;
  _Integer num, nlvl, root, ccsize;
  --xls;
  --perm;
  --adjncy;
  --xadj;

  for (i__ = 1; i__ <= neqns; ++i__) mask[i__] = true;
  num = 1;
  for (i__ = 1; i__ <= neqns; ++i__) {
    if (!mask[i__]) continue;
    root = i__;
    fnroot_(&root, &xadj[1], &adjncy[1], mask, &nlvl, &xls[1], &perm[num]);
    rcm_(root, &xadj[1], &adjncy[1], mask, &perm[num], &ccsize, &xls[1]);
    num += ccsize;
    if (num > neqns) return 0;
  }
  return 0;
}

/*!
 * @}
 */ // group pre

}  // namespace detail

/// \class RCM
/// \tparam _Integer integer type
/// \brief reversed Cuthill-Mckee ordering for reducing bandwidth
/// \ingroup pre
template <typename _Integer>
class RCM {
 public:
  /// \brief apply RCM ordering
  /// \param[in] n number of nodes
  /// \param[in] xadj adjency pointer array
  /// \param[in] adjncy adjencant edges
  /// \param[out] perm permutation array
  inline void apply(const _Integer n, _Integer *xadj, _Integer *adjncy,
                    _Integer *perm) const {
    if (n) {
      _init(n);
      detail::genrcm_(n, xadj, adjncy, perm, _mask, &_xls[0]);
    }
  }

 protected:
  mutable std::vector<bool>     _mask;  ///< not visited mask
  mutable std::vector<_Integer> _xls;   ///< levelset ds

 protected:
  /// \brief initialize workspace
  /// \param[in] n number of nodes
  inline void _init(const _Integer n) const {
    _mask.resize(n + 1);
    _xls.resize(n + 1);
  }
};

}  // namespace rcm
}  // namespace hilucsi

#endif  // _HILUCSI_PRE_RCM_HPP
