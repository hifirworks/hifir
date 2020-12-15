///////////////////////////////////////////////////////////////////////////////
//                  This file is part of HIF project                         //
//                                                                           //
//            Copyright (C) 1996-2015, Timothy A. Davis,                     //
//                                     Patrick R. Amestoy, and               //
//                                     Iain S. Duff.                         //
///////////////////////////////////////////////////////////////////////////////

/*!
 * \file hif/pre/amd.hpp
 * \brief Kernel for approximated minimum degree ordering
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

#ifndef _HIF_PRE_AMD_HPP
#define _HIF_PRE_AMD_HPP

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <type_traits>

#include <cstdint>  // c++11

#ifndef DOXYGEN_SHOULD_SKIP_THIS

#  ifndef AMD_H
#    define AMD_CONTROL 5
#    define AMD_INFO 20

#    define AMD_DENSE 0
#    define AMD_AGGRESSIVE 1

#    define AMD_DEFAULT_DENSE 10.0
#    define AMD_DEFAULT_AGGRESSIVE 1

#    define AMD_STATUS 0
#    define AMD_N 1
#    define AMD_NZ 2
#    define AMD_SYMMETRY 3
#    define AMD_NZDIAG 4
#    define AMD_NZ_A_PLUS_AT 5
#    define AMD_NDENSE 6
#    define AMD_MEMORY 7
#    define AMD_NCMPA 8
#    define AMD_LNZ 9
#    define AMD_NDIV 10
#    define AMD_NMULTSUBS_LDL 11
#    define AMD_NMULTSUBS_LU 12
#    define AMD_DMAX 13

#    define AMD_OK 0
#    define AMD_OUT_OF_MEMORY -1
#    define AMD_INVALID -2
#    define AMD_OK_BUT_JUMBLED 1

#  endif  // AMD_H header checking

// HIF extensions
#  define HIF_AMD_CONTROL (AMD_CONTROL + 2)
#  define HIF_AMD_CHECKING AMD_CONTROL
#  define HIF_AMD_SYMM_FLAG (AMD_CONTROL + 1)

#  ifdef FLIP
#    undef FLIP
#  endif

#  ifdef EMPTY
#    undef EMPTY
#  endif

#  define FLIP(i) (-(i)-2)
#  define UNFLIP(i) ((i < EMPTY) ? FLIP(i) : (i))

#  define SIZE_T_MAX SIZE_MAX

#endif  // DOXYGEN_SHOULD_SKIP_THIS

namespace hif {
namespace amd {

/// \class AMD
/// \brief kernels for approxmiated minimum degree ordering
/// \ingroup pre
///
/// This class contains only \a static member functions, which are extracted
/// and template-ized from original \b SuiteSparse AMD implementation.
template <class Int>
class AMD {
 public:
  constexpr static Int EMPTY   = static_cast<Int>(-1);
  constexpr static Int Int_MAX = std::numeric_limits<Int>::max();

  inline static Int *AMD_malloc(size_t len, size_t n) {
    return (Int *)std::malloc(len * n);
  }

  inline static void AMD_free(void *ptr) {
    if (ptr) std::free(ptr);
  }

  inline static void defaults(double *Control) {
    int i;

    if (Control != nullptr) {
      for (i = 0; i < HIF_AMD_CONTROL; i++) {
        Control[i] = 0;
      }
      Control[AMD_DENSE]         = AMD_DEFAULT_DENSE;
      Control[AMD_AGGRESSIVE]    = AMD_DEFAULT_AGGRESSIVE;
      Control[HIF_AMD_CHECKING]  = 1;
      Control[HIF_AMD_SYMM_FLAG] = 0;
    }
  }

  template <class OStream>
  inline static OStream &control(OStream &s, double Control[]) {
    double alpha;
    Int    aggressive;

    if (Control != nullptr) {
      alpha      = Control[AMD_DENSE];
      aggressive = Control[AMD_AGGRESSIVE] != 0;
    } else {
      alpha      = AMD_DEFAULT_DENSE;
      aggressive = AMD_DEFAULT_AGGRESSIVE;
    }
    s << "\nAMD version 2.4.6, May 4, 2016: approximate minimum degree "
         "ordering\n"
      << "    dense row parameter: " << alpha << '\n';
    if (alpha < 0) {
      s << "    no rows treated as dense\n";
    } else {
      s << "    (rows with more than max (" << alpha
        << " * sqrt (n), 16) entries are\n"
           "    considered \"dense\", and placed last in output "
           "permutation)\n";
    }
    if (aggressive) {
      s << "    aggressive absorption:  yes\n";
    } else {
      s << "    aggressive absorption:  no\n";
    }
    s << "    size of AMD integer: " << sizeof(Int) << "\n\n";
    return s;
  }

  template <class OStream>
  inline static OStream &info(OStream &s, double *Info) {
    double n, ndiv, nmultsubs_ldl, nmultsubs_lu, lnz, lnzd;

    s << "\nAMD version 2.4.6, May 4, 2016, results:\n";
    if (!Info) {
      return s;
    }
    n             = Info[AMD_N];
    ndiv          = Info[AMD_NDIV];
    nmultsubs_ldl = Info[AMD_NMULTSUBS_LDL];
    nmultsubs_lu  = Info[AMD_NMULTSUBS_LU];
    lnz           = Info[AMD_LNZ];
    lnzd          = (n >= 0 && lnz >= 0) ? (n + lnz) : (-1);
    s << "    status: ";
    if (Info[AMD_STATUS] == AMD_OK) {
      s << "OK\n";
    } else if (Info[AMD_STATUS] == AMD_OUT_OF_MEMORY) {
      s << "out of memory\n";
    } else if (Info[AMD_STATUS] == AMD_INVALID) {
      s << "invalid matrix\n";
    } else if (Info[AMD_STATUS] == AMD_OK_BUT_JUMBLED) {
      s << "OK, but jumbled\n";
    } else {
      s << "unknown\n";
    }

#ifdef PRI
#  undef PRI
#endif
#define PRI(__f, __n) \
  if ((__n) >= 0) s << __f << (__n) << '\n'

    PRI("    n, dimension of A:                                  ", n);
    PRI("    nz, number of nonzeros in A:                        ",
        Info[AMD_NZ]);
    PRI("    symmetry of A:                                      ",
        Info[AMD_SYMMETRY]);
    PRI("    number of nonzeros on diagonal:                     ",
        Info[AMD_NZDIAG]);
    PRI("    nonzeros in pattern of A+A' (excl. diagonal):       ",
        Info[AMD_NZ_A_PLUS_AT]);
    PRI("    # dense rows/columns of A+A':                       ",
        Info[AMD_NDENSE]);

    PRI("    memory used, in bytes:                              ",
        Info[AMD_MEMORY]);
    PRI("    # of memory compactions:                            ",
        Info[AMD_NCMPA]);

    s << "\n"
         "    The following approximate statistics are for a subsequent\n"
         "    factorization of A(P,P) + A(P,P)'.  They are slight upper\n"
         "    bounds if there are no dense rows/columns in A+A', and become\n"
         "    looser if dense rows/columns exist.\n\n";

    PRI("    nonzeros in L (excluding diagonal):                 ", lnz);
    PRI("    nonzeros in L (including diagonal):                 ", lnzd);
    PRI("    # divide operations for LDL' or LU:                 ", ndiv);
    PRI("    # multiply-subtract operations for LDL':            ",
        nmultsubs_ldl);
    PRI("    # multiply-subtract operations for LU:              ",
        nmultsubs_lu);
    PRI("    max nz. in any column of L (incl. diagonal):        ",
        Info[AMD_DMAX]);

#undef PRI

    if (n >= 0 && ndiv >= 0 && nmultsubs_ldl >= 0 && nmultsubs_lu >= 0) {
      s << "\n"
           "    chol flop count for real A, sqrt counted as 1 flop: "
        << n + ndiv + 2 * nmultsubs_ldl << '\n'
        << "    LDL' flop count for real A:                         "
        << ndiv + 2 * nmultsubs_ldl << '\n'
        << "    LDL' flop count for complex A:                      "
        << 9 * ndiv + 8 * nmultsubs_ldl << '\n'
        << "    LU flop count for real A (with no pivoting):        "
        << ndiv + 2 * nmultsubs_lu << '\n'
        << "    LU flop count for complex A (with no pivoting):     "
        << 9 * ndiv + 8 * nmultsubs_lu << "\n\n";
    }
    return s;
  }

  inline static Int post_tree(Int root, Int k, Int *Child, const Int *Sibling,
                              Int *Order, Int *Stack
#ifndef NDEBUG
                              ,
                              Int nn
#endif
  ) {
    Int f, head, h, i;
#ifndef NDEBUG
    (void)nn;  // disable warning
#endif
    head     = 0;
    Stack[0] = root;
    while (head >= 0) {
      i = Stack[head];
      if (Child[i] != EMPTY) {
        for (f = Child[i]; f != EMPTY; f = Sibling[f]) {
          head++;
        }
        h = head;
        for (f = Child[i]; f != EMPTY; f = Sibling[f]) {
          Stack[h--] = f;
        }
        Child[i] = EMPTY;
      } else {
        head--;
        Order[i] = k++;
      }
    }
    return (k);
  }

  inline static void postorder(Int nn, Int *Parent, Int *Nv, Int *Fsize,
                               Int *Order, Int *Child, Int *Sibling,
                               Int *Stack) {
    Int i, j, k, parent, frsize, f, fprev, maxfrsize, bigfprev, bigf, fnext;
    for (j = 0; j < nn; j++) {
      Child[j]   = EMPTY;
      Sibling[j] = EMPTY;
    }
    for (j = nn - 1; j >= 0; j--) {
      if (Nv[j] > 0) {
        parent = Parent[j];
        if (parent != EMPTY) {
          Sibling[j]    = Child[parent];
          Child[parent] = j;
        }
      }
    }
    for (i = 0; i < nn; i++) {
      if (Nv[i] > 0 && Child[i] != EMPTY) {
        fprev     = EMPTY;
        maxfrsize = EMPTY;
        bigfprev  = EMPTY;
        bigf      = EMPTY;
        for (f = Child[i]; f != EMPTY; f = Sibling[f]) {
          frsize = Fsize[f];
          if (frsize >= maxfrsize) {
            maxfrsize = frsize;
            bigfprev  = fprev;
            bigf      = f;
          }
          fprev = f;
        }
        fnext = Sibling[bigf];
        if (fnext != EMPTY) {
          if (bigfprev == EMPTY) {
            Child[i] = fnext;
          } else {
            Sibling[bigfprev] = fnext;
          }
          Sibling[bigf]  = EMPTY;
          Sibling[fprev] = bigf;
        }
      }
    }
    for (i = 0; i < nn; i++) {
      Order[i] = EMPTY;
    }
    k = 0;
    for (i = 0; i < nn; i++) {
      if (Parent[i] == EMPTY && Nv[i] > 0) {
        k = post_tree(i, k, Child, Sibling, Order, Stack
#ifndef NDEBUG
                      ,
                      nn
#endif
        );
      }
    }
  }

  static void two(Int n, Int *Pe, Int *Iw, Int *Len, Int iwlen, Int pfree,
                  Int *Nv, Int *Next, Int *Last, Int *Head, Int *Elen,
                  Int *Degree, Int *W, double *Control, double *Info) {
    using std::sqrt;
    const auto clear_flag = [](Int wflg, Int wbig, Int *W, Int n) -> Int {
      Int x;
      if (wflg < 2 || wflg >= wbig) {
        for (x = 0; x < n; x++) {
          if (W[x] != 0) W[x] = 1;
        }
        wflg = 2;
      }
      return (wflg);
    };
    Int deg, degme, dext, lemax, e, elenme, eln, i, ilast, inext, j, jlast,
        jnext, k, knt1, knt2, knt3, lenj, ln, me, mindeg, nel, nleft, nvi, nvj,
        nvpiv, slenme, wbig, we, wflg, wnvi, ok, ndense, ncmpa, dense,
        aggressive;
    typename std::make_unsigned<Int>::type hash;
    double f, r, ndiv, s, nms_lu, nms_ldl, dmax, alpha, lnz, lnzme;
    Int    p, p1, p2, p3, p4, pdst, pend, pj, pme, pme1, pme2, pn, psrc;

    lnz     = 0;
    ndiv    = 0;
    nms_lu  = 0;
    nms_ldl = 0;
    dmax    = 1;
    me      = EMPTY;
    mindeg  = 0;
    ncmpa   = 0;
    nel     = 0;
    lemax   = 0;

    if (Control != nullptr) {
      alpha      = Control[AMD_DENSE];
      aggressive = (Control[AMD_AGGRESSIVE] != 0);
    } else {
      alpha      = AMD_DEFAULT_DENSE;
      aggressive = AMD_DEFAULT_AGGRESSIVE;
    }
    if (alpha < 0) {
      dense = n - 2;
    } else {
      dense = alpha * sqrt((double)n);
    }
    dense = std::max((Int)16, dense);
    dense = std::min(n, dense);
    for (i = 0; i < n; i++) {
      Last[i]   = EMPTY;
      Head[i]   = EMPTY;
      Next[i]   = EMPTY;
      Nv[i]     = 1;
      W[i]      = 1;
      Elen[i]   = 0;
      Degree[i] = Len[i];
    }
    wbig   = Int_MAX - n;
    wflg   = clear_flag(0, wbig, W, n);
    ndense = 0;
    for (i = 0; i < n; i++) {
      deg = Degree[i];
      if (deg == 0) {
        Elen[i] = FLIP(1);
        nel++;
        Pe[i] = EMPTY;
        W[i]  = 0;
      } else if (deg > dense) {
        ndense++;
        Nv[i]   = 0;
        Elen[i] = EMPTY;
        nel++;
        Pe[i] = EMPTY;
      } else {
        inext = Head[deg];
        if (inext != EMPTY) Last[inext] = i;
        Next[i]   = inext;
        Head[deg] = i;
      }
    }
    while (nel < n) {
      for (deg = mindeg; deg < n; deg++) {
        me = Head[deg];
        if (me != EMPTY) break;
      }
      mindeg = deg;
      inext  = Next[me];
      if (inext != EMPTY) Last[inext] = EMPTY;
      Head[deg] = inext;
      elenme    = Elen[me];
      nvpiv     = Nv[me];
      nel += nvpiv;
      Nv[me] = -nvpiv;
      degme  = 0;
      if (elenme == 0) {
        pme1 = Pe[me];
        pme2 = pme1 - 1;
        for (p = pme1; p <= pme1 + Len[me] - 1; p++) {
          i   = Iw[p];
          nvi = Nv[i];
          if (nvi > 0) {
            degme += nvi;
            Nv[i]      = -nvi;
            Iw[++pme2] = i;
            ilast      = Last[i];
            inext      = Next[i];
            if (inext != EMPTY) Last[inext] = ilast;
            if (ilast != EMPTY) {
              Next[ilast] = inext;
            } else {
              Head[Degree[i]] = inext;
            }
          }
        }
      } else {
        p      = Pe[me];
        pme1   = pfree;
        slenme = Len[me] - elenme;
        for (knt1 = 1; knt1 <= elenme + 1; knt1++) {
          if (knt1 > elenme) {
            e  = me;
            pj = p;
            ln = slenme;
          } else {
            e  = Iw[p++];
            pj = Pe[e];
            ln = Len[e];
          }
          for (knt2 = 1; knt2 <= ln; knt2++) {
            i   = Iw[pj++];
            nvi = Nv[i];
            if (nvi > 0) {
              if (pfree >= iwlen) {
                Pe[me] = p;
                Len[me] -= knt1;
                if (Len[me] == 0) Pe[me] = EMPTY;
                Pe[e]  = pj;
                Len[e] = ln - knt2;
                if (Len[e] == 0) Pe[e] = EMPTY;
                ncmpa++;
                for (j = 0; j < n; j++) {
                  pn = Pe[j];
                  if (pn >= 0) {
                    Pe[j]  = Iw[pn];
                    Iw[pn] = FLIP(j);
                  }
                }
                psrc = 0;
                pdst = 0;
                pend = pme1 - 1;
                while (psrc <= pend) {
                  j = FLIP(Iw[psrc++]);
                  if (j >= 0) {
                    Iw[pdst] = Pe[j];
                    Pe[j]    = pdst++;
                    lenj     = Len[j];
                    for (knt3 = 0; knt3 <= lenj - 2; knt3++) {
                      Iw[pdst++] = Iw[psrc++];
                    }
                  }
                }
                p1 = pdst;
                for (psrc = pme1; psrc <= pfree - 1; psrc++) {
                  Iw[pdst++] = Iw[psrc];
                }
                pme1  = p1;
                pfree = pdst;
                pj    = Pe[e];
                p     = Pe[me];
              }
              degme += nvi;
              Nv[i]       = -nvi;
              Iw[pfree++] = i;
              ilast       = Last[i];
              inext       = Next[i];
              if (inext != EMPTY) Last[inext] = ilast;
              if (ilast != EMPTY) {
                Next[ilast] = inext;
              } else {
                Head[Degree[i]] = inext;
              }
            }
          }
          if (e != me) {
            Pe[e] = FLIP(me);
            W[e]  = 0;
          }
        }
        pme2 = pfree - 1;
      }
      Degree[me] = degme;
      Pe[me]     = pme1;
      Len[me]    = pme2 - pme1 + 1;
      Elen[me]   = FLIP(nvpiv + degme);
      wflg       = clear_flag(wflg, wbig, W, n);
      for (pme = pme1; pme <= pme2; pme++) {
        i   = Iw[pme];
        eln = Elen[i];
        if (eln > 0) {
          nvi  = -Nv[i];
          wnvi = wflg - nvi;
          for (p = Pe[i]; p <= Pe[i] + eln - 1; p++) {
            e  = Iw[p];
            we = W[e];
            if (we >= wflg) {
              we -= nvi;
            } else if (we != 0) {
              we = Degree[e] + wnvi;
            }
            W[e] = we;
          }
        }
      }
      for (pme = pme1; pme <= pme2; pme++) {
        i    = Iw[pme];
        p1   = Pe[i];
        p2   = p1 + Elen[i] - 1;
        pn   = p1;
        hash = 0;
        deg  = 0;
        if (aggressive) {
          for (p = p1; p <= p2; p++) {
            e  = Iw[p];
            we = W[e];
            if (we != 0) {
              dext = we - wflg;
              if (dext > 0) {
                deg += dext;
                Iw[pn++] = e;
                hash += e;
              } else {
                Pe[e] = FLIP(me);
                W[e]  = 0;
              }
            }
          }
        } else {
          for (p = p1; p <= p2; p++) {
            e  = Iw[p];
            we = W[e];
            if (we != 0) {
              dext = we - wflg;
              deg += dext;
              Iw[pn++] = e;
              hash += e;
            }
          }
        }
        Elen[i] = pn - p1 + 1;
        p3      = pn;
        p4      = p1 + Len[i];
        for (p = p2 + 1; p < p4; p++) {
          j   = Iw[p];
          nvj = Nv[j];
          if (nvj > 0) {
            deg += nvj;
            Iw[pn++] = j;
            hash += j;
          }
        }
        if (Elen[i] == 1 && p3 == pn) {
          Pe[i] = FLIP(me);
          nvi   = -Nv[i];
          degme -= nvi;
          nvpiv += nvi;
          nel += nvi;
          Nv[i]   = 0;
          Elen[i] = EMPTY;
        } else {
          Degree[i] = std::min(Degree[i], deg);
          Iw[pn]    = Iw[p3];
          Iw[p3]    = Iw[p1];
          Iw[p1]    = me;
          Len[i]    = pn - p1 + 1;
          hash      = hash % n;
          j         = Head[hash];
          if (j <= EMPTY) {
            Next[i]    = FLIP(j);
            Head[hash] = FLIP(i);
          } else {
            Next[i] = Last[j];
            Last[j] = i;
          }
          Last[i] = hash;
        }
      }
      Degree[me] = degme;
      lemax      = std::max(lemax, degme);
      wflg += lemax;
      wflg = clear_flag(wflg, wbig, W, n);
      for (pme = pme1; pme <= pme2; pme++) {
        i = Iw[pme];
        if (Nv[i] < 0) {
          hash = Last[i];
          j    = Head[hash];
          if (j == EMPTY) {
            i = EMPTY;
          } else if (j < EMPTY) {
            i          = FLIP(j);
            Head[hash] = EMPTY;
          } else {
            i       = Last[j];
            Last[j] = EMPTY;
          }
          while (i != EMPTY && Next[i] != EMPTY) {
            ln  = Len[i];
            eln = Elen[i];
            for (p = Pe[i] + 1; p <= Pe[i] + ln - 1; p++) {
              W[Iw[p]] = wflg;
            }
            jlast = i;
            j     = Next[i];
            while (j != EMPTY) {
              ok = (Len[j] == ln) && (Elen[j] == eln);
              for (p = Pe[j] + 1; ok && p <= Pe[j] + ln - 1; p++) {
                if (W[Iw[p]] != wflg) ok = 0;
              }
              if (ok) {
                Pe[j] = FLIP(i);
                Nv[i] += Nv[j];
                Nv[j]       = 0;
                Elen[j]     = EMPTY;
                j           = Next[j];
                Next[jlast] = j;
              } else {
                jlast = j;
                j     = Next[j];
              }
            }
            wflg++;
            i = Next[i];
          }
        }
      }
      p     = pme1;
      nleft = n - nel;
      for (pme = pme1; pme <= pme2; pme++) {
        i   = Iw[pme];
        nvi = -Nv[i];
        if (nvi > 0) {
          Nv[i] = nvi;
          deg   = Degree[i] + degme - nvi;
          deg   = std::min(deg, nleft - nvi);
          inext = Head[deg];
          if (inext != EMPTY) Last[inext] = i;
          Next[i]   = inext;
          Last[i]   = EMPTY;
          Head[deg] = i;
          mindeg    = std::min(mindeg, deg);
          Degree[i] = deg;
          Iw[p++]   = i;
        }
      }
      Nv[me]  = nvpiv;
      Len[me] = p - pme1;
      if (Len[me] == 0) {
        Pe[me] = EMPTY;
        W[me]  = 0;
      }
      if (elenme != 0) {
        pfree = p;
      }
      if (Info != nullptr) {
        f     = nvpiv;
        r     = degme + ndense;
        dmax  = std::max(dmax, f + r);
        lnzme = f * r + (f - 1) * f / 2;
        lnz += lnzme;
        ndiv += lnzme;
        s = f * r * r + r * (f - 1) * f + (f - 1) * f * (2 * f - 1) / 6;
        nms_lu += s;
        nms_ldl += (s + lnzme) / 2;
      }
    }
    if (Info != nullptr) {
      f     = ndense;
      dmax  = std::max(dmax, (double)ndense);
      lnzme = (f - 1) * f / 2;
      lnz += lnzme;
      ndiv += lnzme;
      s = (f - 1) * f * (2 * f - 1) / 6;
      nms_lu += s;
      nms_ldl += (s + lnzme) / 2;
      Info[AMD_LNZ]           = lnz;
      Info[AMD_NDIV]          = ndiv;
      Info[AMD_NMULTSUBS_LDL] = nms_ldl;
      Info[AMD_NMULTSUBS_LU]  = nms_lu;
      Info[AMD_NDENSE]        = ndense;
      Info[AMD_DMAX]          = dmax;
      Info[AMD_NCMPA]         = ncmpa;
      Info[AMD_STATUS]        = AMD_OK;
    }

    for (i = 0; i < n; i++) {
      Pe[i] = FLIP(Pe[i]);
    }
    for (i = 0; i < n; i++) {
      Elen[i] = FLIP(Elen[i]);
    }
    for (i = 0; i < n; i++) {
      if (Nv[i] == 0) {
        j = Pe[i];
        if (j == EMPTY) {
          continue;
        }
        while (Nv[j] == 0) {
          j = Pe[j];
        }
        e = j;
        j = i;
        while (Nv[j] == 0) {
          jnext = Pe[j];
          Pe[j] = e;
          j     = jnext;
        }
      }
    }
    postorder(n, Pe, Nv, Elen, W, Head, Next, Last);
    for (k = 0; k < n; k++) {
      Head[k] = EMPTY;
      Next[k] = EMPTY;
    }
    for (e = 0; e < n; e++) {
      k = W[e];
      if (k != EMPTY) {
        Head[k] = e;
      }
    }
    nel = 0;
    for (k = 0; k < n; k++) {
      e = Head[k];
      if (e == EMPTY) break;
      Next[e] = nel;
      nel += Nv[e];
    }
    for (i = 0; i < n; i++) {
      if (Nv[i] == 0) {
        e = Pe[i];
        if (e != EMPTY) {
          Next[i] = Next[e];
          Next[e]++;
        } else {
          Next[i] = nel++;
        }
      }
    }
    for (i = 0; i < n; i++) {
      k       = Next[i];
      Last[k] = i;
    }
  }

  inline static void one(Int n, const Int *Ap, const Int *Ai, Int *P, Int *Pinv,
                         Int *Len, Int slen, Int *S, double *Control,
                         double *Info) {
    Int i, j, k, p, pfree, iwlen, pj, p1, p2, pj2, *Iw, *Pe, *Nv, *Head, *Elen,
        *Degree, *s, *W, *Sp, *Tp;
    iwlen = slen - 6 * n;
    s     = S;
    Pe    = s;
    s += n;
    Nv = s;
    s += n;
    Head = s;
    s += n;
    Elen = s;
    s += n;
    Degree = s;
    s += n;
    W = s;
    s += n;
    Iw = s;
    s += iwlen;
    Sp    = Nv;
    Tp    = W;
    pfree = 0;
    for (j = 0; j < n; j++) {
      Pe[j] = pfree;
      Sp[j] = pfree;
      pfree += Len[j];
    }
    for (k = 0; k < n; k++) {
      p1 = Ap[k];
      p2 = Ap[k + 1];
      for (p = p1; p < p2;) {
        j = Ai[p];
        if (j < k) {
          Iw[Sp[j]++] = k;
          Iw[Sp[k]++] = j;
          p++;
        } else if (j == k) {
          p++;
          break;
        } else {
          break;
        }
        pj2 = Ap[j + 1];
        for (pj = Tp[j]; pj < pj2;) {
          i = Ai[pj];
          if (i < k) {
            Iw[Sp[i]++] = j;
            Iw[Sp[j]++] = i;
            pj++;
          } else if (i == k) {
            pj++;
            break;
          } else {
            break;
          }
        }
        Tp[j] = pj;
      }
      Tp[k] = p;
    }
    for (j = 0; j < n; j++) {
      for (pj = Tp[j]; pj < Ap[j + 1]; pj++) {
        i           = Ai[pj];
        Iw[Sp[i]++] = j;
        Iw[Sp[j]++] = i;
      }
    }
    two(n, Pe, Iw, Len, iwlen, pfree, Nv, Pinv, P, Head, Elen, Degree, W,
        Control, Info);
  }

  inline static size_t aat(Int n, const Int *Ap, const Int *Ai, Int *Len,
                           Int *Tp, double *Info) {
    Int    p1, p2, p, i, j, pj, pj2, k, nzdiag, nzboth, nz;
    double sym;
    size_t nzaat;
    if (Info != nullptr) {
      for (i = 0; i < AMD_INFO; i++) {
        Info[i] = EMPTY;
      }
      Info[AMD_STATUS] = AMD_OK;
    }
    for (k = 0; k < n; k++) {
      Len[k] = 0;
    }
    nzdiag = 0;
    nzboth = 0;
    nz     = Ap[n];
    for (k = 0; k < n; k++) {
      p1 = Ap[k];
      p2 = Ap[k + 1];
      for (p = p1; p < p2;) {
        j = Ai[p];
        if (j < k) {
          Len[j]++;
          Len[k]++;
          p++;
        } else if (j == k) {
          p++;
          nzdiag++;
          break;
        } else {
          break;
        }
        pj2 = Ap[j + 1];
        for (pj = Tp[j]; pj < pj2;) {
          i = Ai[pj];
          if (i < k) {
            Len[i]++;
            Len[j]++;
            pj++;
          } else if (i == k) {
            pj++;
            nzboth++;
            break;
          } else {
            break;
          }
        }
        Tp[j] = pj;
      }
      Tp[k] = p;
    }
    for (j = 0; j < n; j++) {
      for (pj = Tp[j]; pj < Ap[j + 1]; pj++) {
        i = Ai[pj];
        Len[i]++;
        Len[j]++;
      }
    }
    if (nz == nzdiag) {
      sym = 1;
    } else {
      sym = (2 * (double)nzboth) / ((double)(nz - nzdiag));
    }

    nzaat = 0;
    for (k = 0; k < n; k++) {
      nzaat += Len[k];
    }
    if (Info != nullptr) {
      Info[AMD_STATUS]       = AMD_OK;
      Info[AMD_N]            = n;
      Info[AMD_NZ]           = nz;
      Info[AMD_SYMMETRY]     = sym;
      Info[AMD_NZDIAG]       = nzdiag;
      Info[AMD_NZ_A_PLUS_AT] = nzaat;
    }
    return (nzaat);
  }

  inline static Int valid(Int n_row, Int n_col, const Int *Ap, const Int *Ai) {
    Int nz, j, p1, p2, ilast, i, p, result = AMD_OK;

    if (n_row < 0 || n_col < 0 || Ap == nullptr || Ai == nullptr) {
      return (AMD_INVALID);
    }
    nz = Ap[n_col];
    if (Ap[0] != 0 || nz < 0) {
      return (AMD_INVALID);
    }
    for (j = 0; j < n_col; j++) {
      p1 = Ap[j];
      p2 = Ap[j + 1];
      if (p1 > p2) {
        return (AMD_INVALID);
      }
      ilast = EMPTY;
      for (p = p1; p < p2; p++) {
        i = Ai[p];
        if (i < 0 || i >= n_row) {
          return (AMD_INVALID);
        }
        if (i <= ilast) {
          result = AMD_OK_BUT_JUMBLED;
        }
        ilast = i;
      }
    }
    return (result);
  }

  inline static void preprocess(Int n, const Int *Ap, const Int *Ai, Int *Rp,
                                Int *Ri, Int *W, Int *Flag) {
    Int i, j, p, p2;

    for (i = 0; i < n; i++) {
      W[i]    = 0;
      Flag[i] = EMPTY;
    }
    for (j = 0; j < n; j++) {
      p2 = Ap[j + 1];
      for (p = Ap[j]; p < p2; p++) {
        i = Ai[p];
        if (Flag[i] != j) {
          W[i]++;
          Flag[i] = j;
        }
      }
    }
    Rp[0] = 0;
    for (i = 0; i < n; i++) {
      Rp[i + 1] = Rp[i] + W[i];
    }
    for (i = 0; i < n; i++) {
      W[i]    = Rp[i];
      Flag[i] = EMPTY;
    }
    for (j = 0; j < n; j++) {
      p2 = Ap[j + 1];
      for (p = Ap[j]; p < p2; p++) {
        i = Ai[p];
        if (Flag[i] != j) {
          Ri[W[i]++] = j;
          Flag[i]    = j;
        }
      }
    }
  }

  inline static Int order(Int n, const Int *Ap, const Int *Ai, Int *P,
                          double *Control, double *Info) {
    Int *Len, *S, nz, i, *Pinv, info, status, *Rp, *Ri, *Cp, *Ci, ok, do_check,
        symm_flag;
    size_t nzaat, slen;
    double mem = 0;
    info       = Info != nullptr;
    if (info) {
      for (i = 0; i < AMD_INFO; i++) {
        Info[i] = EMPTY;
      }
      Info[AMD_N]      = n;
      Info[AMD_STATUS] = AMD_OK;
    }
    if (Ai == nullptr || Ap == nullptr || P == nullptr || n < 0) {
      if (info) Info[AMD_STATUS] = AMD_INVALID;
      return (AMD_INVALID);
    }
    if (n == 0) {
      return (AMD_OK);
    }
    nz = Ap[n];
    if (info) {
      Info[AMD_NZ] = nz;
    }
    if (nz < 0) {
      if (info) Info[AMD_STATUS] = AMD_INVALID;
      return (AMD_INVALID);
    }
    if (((size_t)n) >= SIZE_T_MAX / sizeof(Int) ||
        ((size_t)nz) >= SIZE_T_MAX / sizeof(Int)) {
      if (info) Info[AMD_STATUS] = AMD_OUT_OF_MEMORY;
      return (AMD_OUT_OF_MEMORY);
    }

    if ((int)Control[HIF_AMD_CHECKING] == 1) {
      status = valid(n, n, Ap, Ai);
    } else {
      if ((int)Control[HIF_AMD_SYMM_FLAG] == 0)
        status = AMD_OK;
      else
        status = AMD_OK_BUT_JUMBLED;
    }
    if (status == AMD_INVALID) {
      if (info) Info[AMD_STATUS] = AMD_INVALID;
      return (AMD_INVALID);
    }
    Len  = AMD_malloc(n, sizeof(Int));
    Pinv = AMD_malloc(n, sizeof(Int));
    mem += n;
    mem += n;
    if (!Len || !Pinv) {
      AMD_free(Len);
      AMD_free(Pinv);
      if (info) Info[AMD_STATUS] = AMD_OUT_OF_MEMORY;
      return (AMD_OUT_OF_MEMORY);
    }

    if (status == AMD_OK_BUT_JUMBLED && (int)Control[HIF_AMD_SYMM_FLAG] != 0) {
      Rp = AMD_malloc(n + 1, sizeof(Int));
      Ri = AMD_malloc(nz, sizeof(Int));
      mem += (n + 1);
      mem += std::max(nz, (Int)1);
      if (!Rp || !Ri) {
        AMD_free(Rp);
        AMD_free(Ri);
        AMD_free(Len);
        AMD_free(Pinv);
        if (info) Info[AMD_STATUS] = AMD_OUT_OF_MEMORY;
        return (AMD_OUT_OF_MEMORY);
      }
      preprocess(n, Ap, Ai, Rp, Ri, Len, Pinv);
      Cp = Rp;
      Ci = Ri;
    } else {
      Rp = nullptr;
      Ri = nullptr;
      Cp = (Int *)Ap;
      Ci = (Int *)Ai;
    }
    nzaat = aat(n, Cp, Ci, Len, P, Info);
    S     = nullptr;
    slen  = nzaat;
    ok    = ((slen + nzaat / 5) >= slen);
    slen += nzaat / 5;
    for (i = 0; ok && i < 7; i++) {
      ok = ((slen + n) > slen);
      slen += n;
    }
    mem += slen;
    ok = ok && (slen < SIZE_T_MAX / sizeof(Int));
    ok = ok && (slen < Int_MAX);
    if (ok) {
      S = AMD_malloc(slen, sizeof(Int));
    }
    if (!S) {
      AMD_free(Rp);
      AMD_free(Ri);
      AMD_free(Len);
      AMD_free(Pinv);
      if (info) Info[AMD_STATUS] = AMD_OUT_OF_MEMORY;
      return (AMD_OUT_OF_MEMORY);
    }
    if (info) {
      Info[AMD_MEMORY] = mem * sizeof(Int);
    }
    one(n, Cp, Ci, P, Pinv, Len, slen, S, Control, Info);
    AMD_free(Rp);
    AMD_free(Ri);
    AMD_free(Len);
    AMD_free(Pinv);
    AMD_free(S);
    if (info) Info[AMD_STATUS] = status;
    return (status);
  }
};

}  // namespace amd
}  // namespace hif

#endif  // _HIF_PRE_AMD_HPP
