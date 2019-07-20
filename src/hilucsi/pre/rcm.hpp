//@HEADER
//----------------------------------------------------------------------------
//                Copyright (C) 2019 The HILUCSI AUTHORS
//----------------------------------------------------------------------------
//@HEADER

/// \file hilucsi/pre/rcm.hpp
/// \brief Kernel for reversed Cuthill-Mckee ordering
/// \authors Qiao,

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
namespace original {

/*!
 * \addtogroup pre
 * @{
 */

template <typename integer>
inline int rootls_(integer *root, integer *xadj, integer *adjncy,
                   std::vector<bool> &mask, integer *nlvl, integer *xls,
                   integer *ls) {
  integer i__1, i__2;
  integer i__, j, nbr, node, jstop, jstrt, lbegin, ccsize, lvlend, lvsize;
  --ls;
  --xls;
  --adjncy;
  --xadj;
  mask[*root] = 0;
  ls[1]       = *root;
  *nlvl       = 0;
  lvlend      = 0;
  ccsize      = 1;
L200:
  lbegin = lvlend + 1;
  lvlend = ccsize;
  ++(*nlvl);
  xls[*nlvl] = lbegin;
  i__1       = lvlend;
  for (i__ = lbegin; i__ <= i__1; ++i__) {
    node  = ls[i__];
    jstrt = xadj[node];
    jstop = xadj[node + 1] - 1;
    if (jstop < jstrt) {
      goto L400;
    }
    i__2 = jstop;
    for (j = jstrt; j <= i__2; ++j) {
      nbr = adjncy[j];
      if (mask[nbr] == 0) {
        goto L300;
      }
      ++ccsize;
      ls[ccsize] = nbr;
      mask[nbr]  = 0;
    L300:;
    }
  L400:;
  }
  lvsize = ccsize - lvlend;
  if (lvsize > 0) {
    goto L200;
  }
  xls[*nlvl + 1] = lvlend + 1;
  i__1           = ccsize;
  for (i__ = 1; i__ <= i__1; ++i__) {
    node       = ls[i__];
    mask[node] = 1;
  }
  return 0;
}

template <typename integer>
inline int fnroot_(integer *root, integer *xadj, integer *adjncy,
                   std::vector<bool> &mask, integer *nlvl, integer *xls,
                   integer *ls) {
  integer i__1, i__2;
  integer j, k, ndeg, node, nabor, kstop, jstrt, kstrt, mindeg, ccsize, nunlvl;
  --ls;
  --xls;
  --adjncy;
  --xadj;
  rootls_(root, &xadj[1], &adjncy[1], mask, nlvl, &xls[1], &ls[1]);
  ccsize = xls[*nlvl + 1] - 1;
  if (*nlvl == 1 || *nlvl == ccsize) {
    return 0;
  }
L100:
  jstrt  = xls[*nlvl];
  mindeg = ccsize;
  *root  = ls[jstrt];
  if (ccsize == jstrt) {
    goto L400;
  }
  i__1 = ccsize;
  for (j = jstrt; j <= i__1; ++j) {
    node  = ls[j];
    ndeg  = 0;
    kstrt = xadj[node];
    kstop = xadj[node + 1] - 1;
    i__2  = kstop;
    for (k = kstrt; k <= i__2; ++k) {
      nabor = adjncy[k];
      if (mask[nabor] > 0) {
        ++ndeg;
      }
    }
    if (ndeg >= mindeg) {
      goto L300;
    }
    *root  = node;
    mindeg = ndeg;
  L300:;
  }
L400:
  rootls_(root, &xadj[1], &adjncy[1], mask, &nunlvl, &xls[1], &ls[1]);
  if (nunlvl <= *nlvl) {
    return 0;
  }
  *nlvl = nunlvl;
  if (*nlvl < ccsize) {
    goto L100;
  }
  return 0;
}

template <typename integer>
inline int degree_(integer *root, integer *xadj, integer *adjncy,
                   std::vector<bool> &mask, integer *deg, integer *ccsize,
                   integer *ls) {
  integer i__1, i__2;
  integer i__, j, nbr, ideg, node, jstop, jstrt, lbegin, lvlend, lvsize;

  --ls;
  --deg;
  --adjncy;
  --xadj;
  ls[1]       = *root;
  xadj[*root] = -xadj[*root];
  lvlend      = 0;
  *ccsize     = 1;
L100:
  lbegin = lvlend + 1;
  lvlend = *ccsize;
  i__1   = lvlend;
  for (i__ = lbegin; i__ <= i__1; ++i__) {
    node  = ls[i__];
    jstrt = -xadj[node];
    jstop = (i__2 = xadj[node + 1], std::abs(i__2)) - 1;
    ideg  = 0;
    if (jstop < jstrt) {
      goto L300;
    }
    i__2 = jstop;
    for (j = jstrt; j <= i__2; ++j) {
      nbr = adjncy[j];
      if (mask[nbr] == 0) {
        goto L200;
      }
      ++ideg;
      if (xadj[nbr] < 0) {
        goto L200;
      }
      xadj[nbr] = -xadj[nbr];
      ++(*ccsize);
      ls[*ccsize] = nbr;
    L200:;
    }
  L300:
    deg[node] = ideg;
  }
  lvsize = *ccsize - lvlend;
  if (lvsize > 0) {
    goto L100;
  }
  i__1 = *ccsize;
  for (i__ = 1; i__ <= i__1; ++i__) {
    node       = ls[i__];
    xadj[node] = -xadj[node];
  }
  return 0;
}

template <typename integer>
inline int rcm_(integer *root, integer *xadj, integer *adjncy,
                std::vector<bool> &mask, integer *perm, integer *ccsize,
                integer *deg) {
  integer i__1, i__2;
  integer i__, j, k, l, nbr, node, fnbr, lnbr, lperm, jstop, jstrt;
  integer lbegin, lvlend;
  --deg;
  --perm;
  --adjncy;
  --xadj;
  degree_(root, &xadj[1], &adjncy[1], mask, &deg[1], ccsize, &perm[1]);
  mask[*root] = 0;
  if (*ccsize <= 1) {
    return 0;
  }
  lvlend = 0;
  lnbr   = 1;
L100:
  lbegin = lvlend + 1;
  lvlend = lnbr;
  i__1   = lvlend;
  for (i__ = lbegin; i__ <= i__1; ++i__) {
    node  = perm[i__];
    jstrt = xadj[node];
    jstop = xadj[node + 1] - 1;
    fnbr  = lnbr + 1;
    i__2  = jstop;
    for (j = jstrt; j <= i__2; ++j) {
      nbr = adjncy[j];
      if (mask[nbr] == 0) {
        goto L200;
      }
      ++lnbr;
      mask[nbr]  = 0;
      perm[lnbr] = nbr;
    L200:;
    }
    if (fnbr >= lnbr) {
      goto L600;
    }
    k = fnbr;
  L300:
    l = k;
    ++k;
    nbr = perm[k];
  L400:
    if (l < fnbr) {
      goto L500;
    }
    lperm = perm[l];
    if (deg[lperm] <= deg[nbr]) {
      goto L500;
    }
    perm[l + 1] = lperm;
    --l;
    goto L400;
  L500:
    perm[l + 1] = nbr;
    if (k < lnbr) {
      goto L300;
    }
  L600:;
  }
  if (lnbr > lvlend) {
    goto L100;
  }
  k    = *ccsize / 2;
  l    = *ccsize;
  i__1 = k;
  for (i__ = 1; i__ <= i__1; ++i__) {
    lperm     = perm[l];
    perm[l]   = perm[i__];
    perm[i__] = lperm;
    --l;
  }
  return 0;
}

template <typename integer>
inline int genrcm_(integer *neqns, integer *xadj, integer *adjncy,
                   integer *perm, std::vector<bool> &mask, integer *xls) {
  integer i__1;
  integer i__;
  integer num, nlvl, root, ccsize;
  --xls;
  --perm;
  --adjncy;
  --xadj;

  i__1 = *neqns;
  for (i__ = 1; i__ <= i__1; ++i__) {
    mask[i__] = 1;
  }
  num  = 1;
  i__1 = *neqns;
  for (i__ = 1; i__ <= i__1; ++i__) {
    if (mask[i__] == 0) {
      goto L200;
    }
    root = i__;
    fnroot_(&root, &xadj[1], &adjncy[1], mask, &nlvl, &xls[1], &perm[num]);
    rcm_(&root, &xadj[1], &adjncy[1], mask, &perm[num], &ccsize, &xls[1]);
    num += ccsize;
    if (num > *neqns) {
      return 0;
    }
  L200:;
  }
  return 0;
}

/*!
 * @}
 */ // group pre

}  // namespace original

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
  inline void apply(_Integer n, _Integer *xadj, _Integer *adjncy,
                    _Integer *perm) const {
    if (n) {
      _init(n);
      original::genrcm_(&n, xadj, adjncy, perm, _mask, &_xls[0]);
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
