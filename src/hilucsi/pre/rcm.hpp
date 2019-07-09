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

/* rootls.F -- translated by f2c (version 20160102).
   You must link the resulting object file with libf2c:
        on Microsoft Windows system, link with libf2c.lib;
        on Linux or Unix systems, link with .../path/to/libf2c.a -lm
        or, if you install libf2c.a in a standard place, with -lf2c -lm
        -- in that order, at the end of the command line, as in
                cc *.o -lf2c -lm
        Source for libf2c is in /netlib/f2c/libf2c.zip, e.g.,

                http://www.netlib.org/f2c/libf2c.zip
*/

template <typename integer>

/* ----- SUBROUTINE ROOTLS */
/* *************************************************************** */
/* *************************************************************** */
/* ********     ROOTLS ..... ROOTED LEVEL STRUCTURE      ********* */
/* *************************************************************** */
/* *************************************************************** */

/*     PURPOSE - ROOTLS GENERATES THE LEVEL STRUCTURE ROOTED */
/*        AT THE INPUT NODE CALLED ROOT. ONLY THOSE NODES FOR */
/*        WHICH MASK IS NONZERO WILL BE CONSIDERED. */

/*     INPUT PARAMETERS - */
/*        ROOT - THE NODE AT WHICH THE LEVEL STRUCTURE IS TO */
/*               BE ROOTED. */
/*        (XADJ, ADJNCY) - ADJACENCY STRUCTURE PAIR FOR THE */
/*               GIVEN GRAPH. */
/*        MASK - IS USED TO SPECIFY A SECTION SUBGRAPH. NODES */
/*               WITH MASK(I)=0 ARE IGNORED. */

/*     OUTPUT PARAMETERS - */
/*        NLVL - IS THE NUMBER OF LEVELS IN THE LEVEL STRUCTURE. */
/*        (XLS, LS) - ARRAY PAIR FOR THE ROOTED LEVEL STRUCTURE. */

/* *************************************************************** */

inline int rootls_(integer *root, integer *xadj, integer *adjncy,
                   std::vector<bool> &mask, integer *nlvl, integer *xls,
                   integer *ls) {
  /* System generated locals */
  integer i__1, i__2;

  /* Local variables */
  integer i__, j, nbr, node, jstop, jstrt, lbegin, ccsize, lvlend, lvsize;

  /* *************************************************************** */

  /* *************************************************************** */

  /*        ------------------ */
  /*        INITIALIZATION ... */
  /*        ------------------ */
  /* Parameter adjustments */
  --ls;
  --xls;
  // --mask;
  --adjncy;
  --xadj;

  /* Function Body */
  mask[*root] = 0;
  ls[1]       = *root;
  *nlvl       = 0;
  lvlend      = 0;
  ccsize      = 1;
/*        ----------------------------------------------------- */
/*        LBEGIN IS THE POINTER TO THE BEGINNING OF THE CURRENT */
/*        LEVEL, AND LVLEND POINTS TO THE END OF THIS LEVEL. */
/*        ----------------------------------------------------- */
L200:
  lbegin = lvlend + 1;
  lvlend = ccsize;
  ++(*nlvl);
  xls[*nlvl] = lbegin;
  /*        ------------------------------------------------- */
  /*        GENERATE THE NEXT LEVEL BY FINDING ALL THE MASKED */
  /*        NEIGHBORS OF NODES IN THE CURRENT LEVEL. */
  /*        ------------------------------------------------- */
  i__1 = lvlend;
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
  /*        ------------------------------------------ */
  /*        COMPUTE THE CURRENT LEVEL WIDTH. */
  /*        IF IT IS NONZERO, GENERATE THE NEXT LEVEL. */
  /*        ------------------------------------------ */
  lvsize = ccsize - lvlend;
  if (lvsize > 0) {
    goto L200;
  }
  /*        ------------------------------------------------------- */
  /*        RESET MASK TO ONE FOR THE NODES IN THE LEVEL STRUCTURE. */
  /*        ------------------------------------------------------- */
  xls[*nlvl + 1] = lvlend + 1;
  i__1           = ccsize;
  for (i__ = 1; i__ <= i__1; ++i__) {
    node       = ls[i__];
    mask[node] = 1;
    /* L500: */
  }
  return 0;
} /* rootls_ */

/* fnroot.F -- translated by f2c (version 20160102).
   You must link the resulting object file with libf2c:
        on Microsoft Windows system, link with libf2c.lib;
        on Linux or Unix systems, link with .../path/to/libf2c.a -lm
        or, if you install libf2c.a in a standard place, with -lf2c -lm
        -- in that order, at the end of the command line, as in
                cc *.o -lf2c -lm
        Source for libf2c is in /netlib/f2c/libf2c.zip, e.g.,

                http://www.netlib.org/f2c/libf2c.zip
*/

template <typename integer>

/* ----- SUBROUTINE FNROOT */
/* *************************************************************** */
/* *************************************************************** */
/* *******     FNROOT ..... FIND PSEUDO-PERIPHERAL NODE    ******* */
/* *************************************************************** */
/* *************************************************************** */

/*    PURPOSE - FNROOT IMPLEMENTS A MODIFIED VERSION OF THE */
/*       SCHEME BY GIBBS, POOLE, AND STOCKMEYER TO FIND PSEUDO- */
/*       PERIPHERAL NODES.  IT DETERMINES SUCH A NODE FOR THE */
/*       SECTION SUBGRAPH SPECIFIED BY MASK AND ROOT. */

/*    INPUT PARAMETERS - */
/*       (XADJ, ADJNCY) - ADJACENCY STRUCTURE PAIR FOR THE GRAPH. */
/*       MASK - SPECIFIES A SECTION SUBGRAPH. NODES FOR WHICH */
/*              MASK IS ZERO ARE IGNORED BY FNROOT. */

/*    UPDATED PARAMETER - */
/*       ROOT - ON INPUT, IT (ALONG WITH MASK) DEFINES THE */
/*              COMPONENT FOR WHICH A PSEUDO-PERIPHERAL NODE IS */
/*              TO BE FOUND. ON OUTPUT, IT IS THE NODE OBTAINED. */

/*    OUTPUT PARAMETERS - */
/*       NLVL - IS THE NUMBER OF LEVELS IN THE LEVEL STRUCTURE */
/*              ROOTED AT THE NODE ROOT. */
/*       (XLS,LS) - THE LEVEL STRUCTURE ARRAY PAIR CONTAINING */
/*                  THE LEVEL STRUCTURE FOUND. */

/*    PROGRAM SUBROUTINES - */
/*       ROOTLS. */

/* *************************************************************** */

inline int fnroot_(integer *root, integer *xadj, integer *adjncy,
                   std::vector<bool> &mask, integer *nlvl, integer *xls,
                   integer *ls) {
  /* System generated locals */
  integer i__1, i__2;

  /* Local variables */
  integer j, k, ndeg, node, nabor, kstop, jstrt, kstrt, mindeg, ccsize, nunlvl;

  /* *************************************************************** */

  /* *************************************************************** */

  /*        --------------------------------------------- */
  /*        DETERMINE THE LEVEL STRUCTURE ROOTED AT ROOT. */
  /*        --------------------------------------------- */
  /* Parameter adjustments */
  --ls;
  --xls;
  // --mask;
  --adjncy;
  --xadj;

  /* Function Body */
  rootls_(root, &xadj[1], &adjncy[1], mask, nlvl, &xls[1], &ls[1]);
  ccsize = xls[*nlvl + 1] - 1;
  if (*nlvl == 1 || *nlvl == ccsize) {
    return 0;
  }
/*        ---------------------------------------------------- */
/*        PICK A NODE WITH MINIMUM DEGREE FROM THE LAST LEVEL. */
/*        ---------------------------------------------------- */
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
      /* L200: */
    }
    if (ndeg >= mindeg) {
      goto L300;
    }
    *root  = node;
    mindeg = ndeg;
  L300:;
  }
/*        ---------------------------------------- */
/*        AND GENERATE ITS ROOTED LEVEL STRUCTURE. */
/*        ---------------------------------------- */
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
} /* fnroot_ */

/* degree.F -- translated by f2c (version 20160102).
   You must link the resulting object file with libf2c:
        on Microsoft Windows system, link with libf2c.lib;
        on Linux or Unix systems, link with .../path/to/libf2c.a -lm
        or, if you install libf2c.a in a standard place, with -lf2c -lm
        -- in that order, at the end of the command line, as in
                cc *.o -lf2c -lm
        Source for libf2c is in /netlib/f2c/libf2c.zip, e.g.,

                http://www.netlib.org/f2c/libf2c.zip
*/

template <typename integer>

/* ----- SUBROUTINE DEGREE */
/* *************************************************************** */
/* *************************************************************** */
/* ********     DEGREE ..... DEGREE IN MASKED COMPONENT   ******** */
/* *************************************************************** */
/* *************************************************************** */

/*     PURPOSE - THIS ROUTINE COMPUTES THE DEGREES OF THE NODES */
/*        IN THE CONNECTED COMPONENT SPECIFIED BY MASK AND ROOT. */
/*        NODES FOR WHICH MASK IS ZERO ARE IGNORED. */

/*     INPUT PARAMETER - */
/*        ROOT - IS THE INPUT NODE THAT DEFINES THE COMPONENT. */
/*        (XADJ, ADJNCY) - ADJACENCY STRUCTURE PAIR. */
/*        MASK - SPECIFIES A SECTION SUBGRAPH. */

/*     OUTPUT PARAMETERS - */
/*        DEG - ARRAY CONTAINING THE DEGREES OF THE NODES IN */
/*              THE COMPONENT. */
/*        CCSIZE-SIZE OF THE COMPONENT SPECIFED BY MASK AND ROOT */

/*     WORKING PARAMETER - */
/*        LS - A TEMPORARY VECTOR USED TO STORE THE NODES OF THE */
/*               COMPONENT LEVEL BY LEVEL. */

/* *************************************************************** */

inline int degree_(integer *root, integer *xadj, integer *adjncy,
                   std::vector<bool> &mask, integer *deg, integer *ccsize,
                   integer *ls) {
  /* System generated locals */
  integer i__1, i__2;

  /* Local variables */
  integer i__, j, nbr, ideg, node, jstop, jstrt, lbegin, lvlend, lvsize;

  /* *************************************************************** */

  /* *************************************************************** */

  /*        ------------------------------------------------- */
  /*        INITIALIZATION ... */
  /*        THE ARRAY XADJ IS USED AS A TEMPORARY MARKER TO */
  /*        INDICATE WHICH NODES HAVE BEEN CONSIDERED SO FAR. */
  /*        ------------------------------------------------- */
  /* Parameter adjustments */
  --ls;
  --deg;
  // --mask;
  --adjncy;
  --xadj;

  /* Function Body */
  ls[1]       = *root;
  xadj[*root] = -xadj[*root];
  lvlend      = 0;
  *ccsize     = 1;
/*        ----------------------------------------------------- */
/*        LBEGIN IS THE POINTER TO THE BEGINNING OF THE CURRENT */
/*        LEVEL, AND LVLEND POINTS TO THE END OF THIS LEVEL. */
/*        ----------------------------------------------------- */
L100:
  lbegin = lvlend + 1;
  lvlend = *ccsize;
  /*        ----------------------------------------------- */
  /*        FIND THE DEGREES OF NODES IN THE CURRENT LEVEL, */
  /*        AND AT THE SAME TIME, GENERATE THE NEXT LEVEL. */
  /*        ----------------------------------------------- */
  i__1 = lvlend;
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
    /* L400: */
  }
  /*        ------------------------------------------ */
  /*        COMPUTE THE CURRENT LEVEL WIDTH. */
  /*        IF IT IS NONZERO , GENERATE ANOTHER LEVEL. */
  /*        ------------------------------------------ */
  lvsize = *ccsize - lvlend;
  if (lvsize > 0) {
    goto L100;
  }
  /*        ------------------------------------------ */
  /*        RESET XADJ TO ITS CORRECT SIGN AND RETURN. */
  /*        ------------------------------------------ */
  i__1 = *ccsize;
  for (i__ = 1; i__ <= i__1; ++i__) {
    node       = ls[i__];
    xadj[node] = -xadj[node];
    /* L500: */
  }
  return 0;
} /* degree_ */

/* rcm.F -- translated by f2c (version 20160102).
   You must link the resulting object file with libf2c:
        on Microsoft Windows system, link with libf2c.lib;
        on Linux or Unix systems, link with .../path/to/libf2c.a -lm
        or, if you install libf2c.a in a standard place, with -lf2c -lm
        -- in that order, at the end of the command line, as in
                cc *.o -lf2c -lm
        Source for libf2c is in /netlib/f2c/libf2c.zip, e.g.,

                http://www.netlib.org/f2c/libf2c.zip
*/

template <typename integer>

/* ----- SUBROUTINE RCM */
/* *************************************************************** */
/* *************************************************************** */
/* ********     RCM ..... REVERSE CUTHILL-MCKEE ORDERING   ******* */
/* *************************************************************** */
/* *************************************************************** */

/*     PURPOSE - RCM NUMBERS A CONNECTED COMPONENT SPECIFIED BY */
/*        MASK AND ROOT, USING THE RCM ALGORITHM. */
/*        THE NUMBERING IS TO BE STARTED AT THE NODE ROOT. */

/*     INPUT PARAMETERS - */
/*        ROOT - IS THE NODE THAT DEFINES THE CONNECTED */
/*               COMPONENT AND IT IS USED AS THE STARTING */
/*               NODE FOR THE RCM ORDERING. */
/*        (XADJ, ADJNCY) - ADJACENCY STRUCTURE PAIR FOR */
/*               THE GRAPH. */

/*     UPDATED PARAMETERS - */
/*        MASK - ONLY THOSE NODES WITH NONZERO INPUT MASK */
/*               VALUES ARE CONSIDERED BY THE ROUTINE.  THE */
/*               NODES NUMBERED BY RCM WILL HAVE THEIR */
/*               MASK VALUES SET TO ZERO. */

/*     OUTPUT PARAMETERS - */
/*        PERM - WILL CONTAIN THE RCM ORDERING. */
/*        CCSIZE - IS THE SIZE OF THE CONNECTED COMPONENT */
/*               THAT HAS BEEN NUMBERED BY RCM. */

/*     WORKING PARAMETER - */
/*        DEG - IS A TEMPORARY VECTOR USED TO HOLD THE DEGREE */
/*               OF THE NODES IN THE SECTION GRAPH SPECIFIED */
/*               BY MASK AND ROOT. */

/*     PROGRAM SUBROUTINES - */
/*        DEGREE. */

/* *************************************************************** */

inline int rcm_(integer *root, integer *xadj, integer *adjncy,
                std::vector<bool> &mask, integer *perm, integer *ccsize,
                integer *deg) {
  /* System generated locals */
  integer i__1, i__2;

  /* Local variables */
  integer i__, j, k, l, nbr, node, fnbr, lnbr, lperm, jstop, jstrt;
  integer lbegin, lvlend;

  /* *************************************************************** */

  /* *************************************************************** */

  /*        ------------------------------------- */
  /*        FIND THE DEGREES OF THE NODES IN THE */
  /*        COMPONENT SPECIFIED BY MASK AND ROOT. */
  /*        ------------------------------------- */
  /* Parameter adjustments */
  --deg;
  --perm;
  // --mask;
  --adjncy;
  --xadj;

  /* Function Body */
  degree_(root, &xadj[1], &adjncy[1], mask, &deg[1], ccsize, &perm[1]);
  mask[*root] = 0;
  if (*ccsize <= 1) {
    return 0;
  }
  lvlend = 0;
  lnbr   = 1;
/*        -------------------------------------------- */
/*        LBEGIN AND LVLEND POINT TO THE BEGINNING AND */
/*        THE END OF THE CURRENT LEVEL RESPECTIVELY. */
/*        -------------------------------------------- */
L100:
  lbegin = lvlend + 1;
  lvlend = lnbr;
  i__1   = lvlend;
  for (i__ = lbegin; i__ <= i__1; ++i__) {
    /*           ---------------------------------- */
    /*           FOR EACH NODE IN CURRENT LEVEL ... */
    /*           ---------------------------------- */
    node  = perm[i__];
    jstrt = xadj[node];
    jstop = xadj[node + 1] - 1;
    /*           ------------------------------------------------ */
    /*           FIND THE UNNUMBERED NEIGHBORS OF NODE. */
    /*           FNBR AND LNBR POINT TO THE FIRST AND LAST */
    /*           UNNUMBERED NEIGHBORS RESPECTIVELY OF THE CURRENT */
    /*           NODE IN PERM. */
    /*           ------------------------------------------------ */
    fnbr = lnbr + 1;
    i__2 = jstop;
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
    /*              ------------------------------------------ */
    /*              SORT THE NEIGHBORS OF NODE IN INCREASING */
    /*              ORDER BY DEGREE. LINEAR INSERTION IS USED. */
    /*              ------------------------------------------ */
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
  /*        --------------------------------------- */
  /*        WE NOW HAVE THE CUTHILL MCKEE ORDERING. */
  /*        REVERSE IT BELOW ... */
  /*        --------------------------------------- */
  k    = *ccsize / 2;
  l    = *ccsize;
  i__1 = k;
  for (i__ = 1; i__ <= i__1; ++i__) {
    lperm     = perm[l];
    perm[l]   = perm[i__];
    perm[i__] = lperm;
    --l;
    /* L700: */
  }
  return 0;
} /* rcm_ */

/* genrcm.F -- translated by f2c (version 20160102).
   You must link the resulting object file with libf2c:
        on Microsoft Windows system, link with libf2c.lib;
        on Linux or Unix systems, link with .../path/to/libf2c.a -lm
        or, if you install libf2c.a in a standard place, with -lf2c -lm
        -- in that order, at the end of the command line, as in
                cc *.o -lf2c -lm
        Source for libf2c is in /netlib/f2c/libf2c.zip, e.g.,

                http://www.netlib.org/f2c/libf2c.zip
*/

template <typename integer>

/* ----- SUBROUTINE GENRCM */
/* *************************************************************** */
/* *************************************************************** */
/* ********   GENRCM ..... GENERAL REVERSE CUTHILL MCKEE   ******* */
/* *************************************************************** */
/* *************************************************************** */

/*     PURPOSE - GENRCM FINDS THE REVERSE CUTHILL-MCKEE */
/*        ORDERING FOR A GENERAL GRAPH. FOR EACH CONNECTED */
/*        COMPONENT IN THE GRAPH, GENRCM OBTAINS THE ORDERING */
/*        BY CALLING THE SUBROUTINE RCM. */

/*     INPUT PARAMETERS - */
/*        NEQNS - NUMBER OF EQUATIONS */
/*        (XADJ, ADJNCY) - ARRAY PAIR CONTAINING THE ADJACENCY */
/*               STRUCTURE OF THE GRAPH OF THE MATRIX. */

/*     OUTPUT PARAMETER - */
/*        PERM - VECTOR THAT CONTAINS THE RCM ORDERING. */

/*     WORKING PARAMETERS - */
/*        MASK - IS USED TO MARK VARIABLES THAT HAVE BEEN */
/*               NUMBERED DURING THE ORDERING PROCESS. IT IS */
/*               INITIALIZED TO 1, AND SET TO ZERO AS EACH NODE */
/*               IS NUMBERED. */
/*        XLS - THE INDEX VECTOR FOR A LEVEL STRUCTURE.  THE */
/*               LEVEL STRUCTURE IS STORED IN THE CURRENTLY */
/*               UNUSED SPACES IN THE PERMUTATION VECTOR PERM. */

/*     PROGRAM SUBROUTINES - */
/*        FNROOT, RCM. */

/* *************************************************************** */

inline int genrcm_(integer *neqns, integer *xadj, integer *adjncy,
                   integer *perm, std::vector<bool> &mask, integer *xls) {
  /* System generated locals */
  integer i__1;

  /* Local variables */
  integer i__;
  integer num, nlvl, root, ccsize;

  /* *************************************************************** */

  /* *************************************************************** */

  /* Parameter adjustments */
  --xls;
  // --mask;
  --perm;
  --adjncy;
  --xadj;

  /* Function Body */
  i__1 = *neqns;
  for (i__ = 1; i__ <= i__1; ++i__) {
    mask[i__] = 1;
    /* L100: */
  }
  num  = 1;
  i__1 = *neqns;
  for (i__ = 1; i__ <= i__1; ++i__) {
    /*           --------------------------------------- */
    /*           FOR EACH MASKED CONNECTED COMPONENT ... */
    /*           --------------------------------------- */
    if (mask[i__] == 0) {
      goto L200;
    }
    root = i__;
    /*              ----------------------------------------- */
    /*              FIRST FIND A PSEUDO-PERIPHERAL NODE ROOT. */
    /*              NOTE THAT THE LEVEL STRUCTURE FOUND BY */
    /*              FNROOT IS STORED STARTING AT PERM(NUM). */
    /*              THEN RCM IS CALLED TO ORDER THE COMPONENT */
    /*              USING ROOT AS THE STARTING NODE. */
    /*              ----------------------------------------- */
    fnroot_(&root, &xadj[1], &adjncy[1], mask, &nlvl, &xls[1], &perm[num]);
    rcm_(&root, &xadj[1], &adjncy[1], mask, &perm[num], &ccsize, &xls[1]);
    num += ccsize;
    if (num > *neqns) {
      return 0;
    }
  L200:;
  }
  return 0;
} /* genrcm_ */

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
