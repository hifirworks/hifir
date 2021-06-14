///////////////////////////////////////////////////////////////////////////////
//                  This file is part of the HIFIR library                   //
///////////////////////////////////////////////////////////////////////////////

/*!
 * \file hif/pre/equilibrate.hpp
 * \brief balancing systems to achieve better conditioning
 * \author Yujie Xiao
 * \author Qiao Chen

\verbatim
Copyright (C) 2021 NumGeom Group at Stony Brook University

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

#ifndef _HIF_PRE_EQUILIBRATE_HPP
#define _HIF_PRE_EQUILIBRATE_HPP

#include <cmath>
#include <vector>

#define _PAR_NUM 10

namespace hif {
namespace eql {
namespace detail {

/*!
 * \addtogroup pre
 * @{
 */

template <typename _Integer>
inline void set_default_pars(_Integer *pars) {
  pars[0] = 6;
  pars[1] = 6;
  pars[2] = -1;
  for (int i = 3; i < _PAR_NUM; ++i) pars[i] = 0;
}

template <typename _Integer, typename _Real>
inline void kernel(_Integer, const _Integer *, const _Integer *, const _Real *,
                   _Integer *, _Integer *, _Integer *, _Integer *, _Integer *,
                   _Integer *, _Integer *, _Real *, _Real *);

template <typename _Integer, typename _ValReal, typename _Real>
inline void compute(const _Integer n, const _Integer ne, const _Integer *indptr,
                    const _Integer *indices, const _ValReal *vals, _Integer *no,
                    _Integer *cperm, _Integer liwork, _Integer *iwork,
                    _Integer lwork, _Real *work, const _Integer *pars,
                    _Integer *info) {
  _Integer j, k;
  _Real    fact, rinf;
  --cperm;
  --indptr;
  --vals;
  --indices;
  --iwork;
  --work;
  --pars;
  --info;
  rinf = sizeof(_Real) == 8u ? HUGE_VAL : HUGE_VALF;
  if (n < 1) {
    info[1] = -2;
    info[2] = n;
    return;
  }
  if (ne < 1) {
    info[1] = -3;
    info[2] = ne;
    return;
  }
  if (liwork < n * 5) {
    info[1] = -4;
    info[2] = n * 5;
    return;
  }
  if (lwork < n * 3 + ne) {
    info[1] = -5;
    info[2] = n * 3 + ne;
    return;
  }
  if (pars[4] == 0) {
    for (_Integer i = 1; i <= n; ++i) iwork[i] = 0;
    for (j = 1; j <= n; ++j) {
      const _Integer kstop = indptr[j + 1] - 1;
      for (k = indptr[j]; k <= kstop; ++k) {
        const _Integer node = indices[k];
        if (node < 1 || node > n) {
          info[1] = -6;
          info[2] = j;
          return;
        }
        if (iwork[node] == j) {
          info[1] = -7;
          info[2] = j;
          return;
        } else
          iwork[node] = j;
      }
    }
  }
  for (int i = 1; i <= 10; ++i) info[i] = 0;
  for (j = 1; j <= n; ++j) {
    fact                 = 0;
    const _Integer kstop = indptr[j + 1] - 1;
    for (k = indptr[j]; k <= kstop; ++k) {
      work[n * 3 + k] = std::abs(vals[k]);
      if (work[n * 3 + k] > fact) fact = work[n * 3 + k];
    }
    work[(n << 1) + j] = fact;
    fact               = fact != 0 ? std::log(fact) : rinf / n;
    for (k = indptr[j]; k <= kstop; ++k) {
      if (work[n * 3 + k] != 0)
        work[n * 3 + k] = fact - std::log(work[n * 3 + k]);
      else
        work[n * 3 + k] = rinf / n;
    }
  }
  kernel(n, &indptr[1], &indices[1], &work[n * 3 + 1], &cperm[1], no, &iwork[1],
         &iwork[n + 1], &iwork[(n << 1) + 1], &iwork[n * 3 + 1],
         &iwork[(n << 2) + 1], &work[1], &work[n + 1]);
  if (*no == n) {
    for (j = 1; j <= n; ++j) {
      if (work[(n << 1) + j] != 0)
        work[n + j] -= std::log(work[(n << 1) + j]);
      else
        work[n + j] = 0;
    }
  }
  fact = std::log(rinf) * .5;
  for (j = 1; j <= n; ++j) {
    if (work[j] < fact && work[n + j] < fact) continue;
    info[1] = 2;
    break;
  }
  if (info[1] == 0 && *no < n) info[1] = 1;
}

// if flag equals 1, node I is pushed upwards
// if flag equals 0, node I is pushed downwards
template <typename _Integer, typename _Real>
inline void kernel_(const _Integer j, const _Integer n, _Integer *q, _Real *d__,
                    _Integer *l, const bool flag) {
  _Integer qk, pos, pos_h;
  --l;
  --d__;
  --q;
  pos = l[j];
  // pos is index of current position of node I in the tree
  if (pos > 1) {
    const _Real dj = d__[j];
    if (flag) {
      for (_Integer __i__ = 1; __i__ <= n; ++__i__) {
        pos_h = pos / 2;
        qk    = q[pos_h];
        if (dj <= d__[qk]) break;
        q[pos] = qk;
        l[qk]  = pos;
        pos    = pos_h;
        if (pos <= 1) break;
      }
    } else {
      for (_Integer __i__ = 1; __i__ <= n; ++__i__) {
        pos_h = pos / 2;
        qk    = q[pos_h];
        if (dj >= d__[qk]) break;
        q[pos] = qk;
        l[qk]  = pos;
        pos    = pos_h;
        if (pos <= 1) break;
      }
    }
  }
  q[pos] = j;
  l[j]   = pos;
}

// delete root node from the binary heap
template <typename _Integer, typename _Real>
inline void kernel__(_Integer *ql, const _Integer n, _Integer *q, _Real *d__,
                     _Integer *l, const bool flag) {
  _Real    dk, dr;
  _Integer pos, pos2;
  // const _Integer n = *n;
  --l;
  --d__;
  --q;
  const _Integer j  = q[*ql];
  const _Real    dj = d__[j];
  --(*ql);
  pos = 1;
  if (flag) {
    for (_Integer __i__ = 1; __i__ <= n; ++__i__) {
      pos2 = pos << 1;
      if (pos2 > *ql) break;
      dk = d__[q[pos2]];
      if (pos2 < *ql) {
        dr = d__[q[pos2 + 1]];
        if (dk < dr) {
          ++pos2;
          dk = dr;
        }
      }
      if (dj >= dk) break;
      // Exchange old last element with smaller child
      q[pos]    = q[pos2];
      l[q[pos]] = pos;
      pos       = pos2;
    }
  }
  // the point is never reached
  else {
    for (_Integer __i__ = 1; __i__ <= n; ++__i__) {
      pos2 = pos << 1;
      if (pos2 > *ql) break;
      dk = d__[q[pos2]];
      if (pos2 < *ql) {
        dr = d__[q[pos2 + 1]];
        if (dk > dr) {
          ++pos2;
          dk = dr;
        }
      }
      if (dj <= dk) break;
      q[pos]    = q[pos2];
      l[q[pos]] = pos;
      pos       = pos2;
    }
  }
  q[pos] = j;
  l[j]   = pos;
}

// move last element in the heap
template <typename _Integer, typename _Real>
inline void kernel___(_Integer *pos_, _Integer *qlen, const _Integer n,
                      _Integer *q, _Real *d__, _Integer *l, const bool flag) {
  _Real    dk, dr;
  _Integer qk, pos, posk;
  --l;
  --d__;
  --q;
  const _Integer pos_in = *pos_;
  if (*qlen == pos_in) {
    --(*qlen);
    return;
  }
  const _Integer j  = q[*qlen];
  const _Real    dj = d__[j];
  --(*qlen);
  pos = pos_in;
  if (flag) {
    if (pos > 1)
      for (_Integer __i__ = 1; __i__ <= n; ++__i__) {
        posk = pos / 2;
        qk   = q[posk];
        if (dj <= d__[qk]) break;
        q[pos] = qk;
        l[qk]  = pos;
        pos    = posk;
        if (pos <= 1) break;
      }
    q[pos] = j;
    l[j]   = pos;
    if (pos != pos_in) return;
    for (_Integer __i__ = 1; __i__ <= n; ++__i__) {
      posk = pos << 1;
      if (posk > *qlen) break;
      dk = d__[q[posk]];
      if (posk < *qlen) {
        dr = d__[q[posk + 1]];
        if (dk < dr) {
          ++posk;
          dk = dr;
        }
      }
      if (dj >= dk) break;
      qk     = q[posk];
      q[pos] = qk;
      l[qk]  = pos;
      pos    = posk;
    }
  } else {
    if (pos > 1) {
      for (_Integer __i__ = 1; __i__ <= n; ++__i__) {
        posk = pos / 2;
        qk   = q[posk];
        if (dj >= d__[qk]) break;
        q[pos] = qk;
        l[qk]  = pos;
        pos    = posk;
        if (pos <= 1) break;
      }
    }
    q[pos] = j;
    l[j]   = pos;
    if (pos != pos_in) return;
    for (_Integer __i__ = 1; __i__ <= n; ++__i__) {
      posk = pos << 1;
      if (posk > *qlen) break;
      dk = d__[q[posk]];
      if (posk < *qlen) {
        dr = d__[q[posk + 1]];
        if (dk > dr) {
          ++posk;
          dk = dr;
        }
      }
      if (dj <= dk) break;
      qk     = q[posk];
      q[pos] = qk;
      l[qk]  = pos;
      pos    = posk;
    }
  }
  q[pos] = j;
  l[j]   = pos;
}

// clean up function to be used in the kernel
template <typename _Integer, typename _Real>
void cleanup(const _Integer n, _Integer *num, _Integer *iperm, _Integer *jperm,
             const _Integer *irn, _Integer *out, const _Real *a, _Real *d__,
             _Real *u) {
  _Integer i__, j, k;

  // store dual column variable u in d(1:n)
  for (j = 1; j <= n; ++j) {
    k = jperm[j];
    if (k != 0)
      d__[j] = a[k] - u[irn[k]];
    else
      d__[j] = 0;
    if (iperm[j] == 0) u[j] = 0;
  }
  if (*num == n) return;
  // else, the matrix is structurally singular, complete iperm
  // jperm, out are work arrays
  for (j = 1; j <= n; ++j) jperm[j] = 0;
  k = 0;
  for (i__ = 1; i__ <= n; ++i__) {
    if (iperm[i__] == 0) {
      ++k;
      out[k] = i__;
    } else {
      j        = iperm[i__];
      jperm[j] = i__;
    }
  }
  k = 0;
  for (j = 1; j <= n; ++j) {
    if (jperm[j] != 0) continue;
    ++k;
    iperm[out[k]] = -j;
  }
}

// main loop
template <typename _Integer, typename _Real>
inline void kernel(const _Integer n, const _Integer *ip, const _Integer *irn,
                   const _Real *a, _Integer *iperm, _Integer *num,
                   _Integer *jperm, _Integer *out, _Integer *pr, _Integer *q,
                   _Integer *l, _Real *u, _Real *d__) {
  _Integer i__, j, k, i0, k0, k1;
  _Real    di;
  _Integer ii, jj, kk;
  _Real    vj;
  _Integer up;
  _Real    dq0;
  _Integer kk1;
  _Real    csp;
  _Integer isp, jsp, low;
  _Real    dmin_, dnew;
  _Integer jord, qlen;
  _Real    rinf;
  _Integer lpos;
  --d__;
  --u;
  --l;
  --q;
  --pr;
  --out;
  --jperm;
  --iperm;
  --ip;
  --a;
  --irn;
  // initialize
  rinf = sizeof(_Real) == 8u ? HUGE_VAL : HUGE_VALF;
  *num = 0;
  for (k = 1; k <= n; ++k) {
    u[k]     = rinf;
    d__[k]   = 0.f;
    iperm[k] = 0;
    jperm[k] = 0;
    pr[k]    = ip[k];
    l[k]     = 0;
  }
  // initialize u
  for (j = 1; j <= n; ++j) {
    const _Integer kstop = ip[j + 1] - 1;
    for (k = ip[j]; k <= kstop; ++k) {
      i__ = irn[k];
      if (a[k] > u[i__]) continue;
      u[i__]     = a[k];
      iperm[i__] = j;
      l[i__]     = k;
    }
  }
  for (i__ = 1; i__ <= n; ++i__) {
    j = iperm[i__];
    if (j == 0) continue;
    // row i is not empty
    iperm[i__] = 0;
    if (jperm[j] != 0) continue;
    // don't choose cheap assignment from dense columns
    if (ip[j + 1] - ip[j] > n / 10 && n > 50) continue;
    // assignment of column j to row i
    ++(*num);
    iperm[i__] = j;
    jperm[j]   = l[i__];
  }
  if (*num == n) {
    cleanup(n, num, iperm, jperm, irn, out, a, d__, u);
    return;
  }
  // scan unassigned columns
  for (j = 1; j <= n; ++j) {
    if (jperm[j] != 0) continue;  // only when j is already assigned
    k1                   = ip[j];
    const _Integer kstop = ip[j + 1] - 1;
    // continue only if j is not empty
    if (k1 > kstop) continue;
    // allow NaNs
    i0 = irn[k1];
    vj = a[k1] - u[i0];
    k0 = k1;
    for (k = k1 + 1; k <= kstop; ++k) {
      i__ = irn[k];
      di  = a[k] - u[i__];
      if (di > vj) continue;
      if (di < vj || di == rinf) {
        vj = di;
        i0 = i__;
        k0 = k;
        continue;
      }
      if (iperm[i__] != 0 || iperm[i0] == 0) continue;
      vj = di;
      i0 = i__;
      k0 = k;
    }
    d__[j] = vj;
    k      = k0;
    i__    = i0;
    if (iperm[i__] == 0) {
      ++(*num);
      jperm[j]   = k;
      iperm[i__] = j;
      pr[j]      = k + 1;
      continue;
    }
    for (k = k0; k <= kstop; ++k) {
      i__ = irn[k];
      if (a[k] - u[i__] > vj) continue;
      jj = iperm[i__];
      // scan remaining part of assigned column jj
      kk1                   = pr[jj];
      const _Integer kkstop = ip[jj + 1] - 1;
      if (kk1 > kkstop) continue;
      bool bb = false;
      for (kk = kk1; kk <= kkstop; ++kk) {
        ii = irn[kk];
        if (iperm[ii] > 0) continue;
        if (a[kk] - u[ii] <= d__[jj]) {
          jperm[jj] = kk;
          iperm[ii] = jj;
          pr[jj]    = kk + 1;
          ++(*num);
          jperm[j]   = k;
          iperm[i__] = j;
          pr[j]      = k + 1;
          bb         = true;
          break;
        }
      }
      if (bb) break;
      pr[jj] = kkstop + 1;
    }
  }
  if (*num == n) {
    cleanup(n, num, iperm, jperm, irn, out, a, d__, u);
    return;
  }
  // prepare for the main loop
  for (i__ = 1; i__ <= n; ++i__) {
    d__[i__] = rinf;
    l[i__]   = 0;
  }

  // main loop: similar to Dijkstra's algorithm
  for (jord = 1; jord <= n; ++jord) {
    if (jperm[jord] != 0) continue;
    dmin_ = rinf;
    qlen  = 0;
    low   = n + 1;
    up    = n + 1;
    csp   = rinf;
    j     = jord;
    pr[j] = -1;
    // scan column j
    const _Integer kstop = ip[j + 1] - 1;
    for (k = ip[j]; k <= kstop; ++k) {
      i__  = irn[k];
      dnew = a[k] - u[i__];
      if (dnew >= csp) continue;
      if (iperm[i__] == 0) {
        csp = dnew;
        isp = k;
        jsp = j;
      } else {
        if (dnew < dmin_) dmin_ = dnew;
        d__[i__] = dnew;
        ++qlen;
        q[qlen] = k;
      }
    }
    // initialize heap q and q2 with rows held in q(1:qlen)
    const _Integer kkstop = qlen;
    qlen                  = 0;
    for (kk = 1; kk <= kkstop; ++kk) {
      k   = q[kk];
      i__ = irn[k];
      if (csp <= d__[i__]) {
        d__[i__] = rinf;
        continue;
      }
      if (d__[i__] <= dmin_) {
        --low;
        q[low] = i__;
        l[i__] = low;
      } else {
        ++qlen;
        l[i__] = qlen;
        kernel_(i__, n, &q[1], &d__[1], &l[1], false);
      }
      // update tree
      jj      = iperm[i__];
      out[jj] = k;
      pr[jj]  = j;
    }
    const _Integer jstop = *num;
    // if q2 is empty, extract rows from q
    for (_Integer __j_ = 1; __j_ <= jstop; ++__j_) {
      if (low == up) {
        if (qlen == 0) break;
        i__ = q[1];
        if (d__[i__] >= csp) break;
        dmin_ = d__[i__];
        for (;;) {
          kernel__(&qlen, n, &q[1], &d__[1], &l[1], false);
          --low;
          q[low] = i__;
          l[i__] = low;
          if (qlen == 0) break;
          i__ = q[1];
          if (d__[i__] > dmin_) break;
        }
      }
      // q0 is the row whose cost d(q0) is the smallest
      const _Integer q0 = q[up - 1];
      dq0               = d__[q0];
      // exit loop if path to q0 is no longer the shortest
      if (dq0 >= csp) break;
      --up;
      // scan column matching q0
      j                    = iperm[q0];
      vj                   = dq0 - a[jperm[j]] + u[q0];
      const _Integer kstop = ip[j + 1] - 1;
      for (k = ip[j]; k <= kstop; ++k) {
        i__ = irn[k];
        if (l[i__] >= up) continue;
        dnew = vj + a[k] - u[i__];  // dnew is the new cost
        if (dnew >= csp)
          continue;  // if dnew >= cost of shortest path, do not update
        // row i is unmatched, update shortest path
        if (iperm[i__] == 0) {
          csp = dnew;
          isp = k;
          jsp = j;
        } else {
          // row i is matched, do not update d(i) if dnew is larger
          di = d__[i__];
          if (di <= dnew) continue;
          if (l[i__] >= low) continue;
          d__[i__] = dnew;
          if (dnew <= dmin_) {
            lpos = l[i__];
            if (lpos != 0) {
              kernel___(&lpos, &qlen, n, &q[1], &d__[1], &l[1], false);
            }
            --low;
            q[low] = i__;
            l[i__] = low;
          } else {
            if (l[i__] == 0) {
              ++qlen;
              l[i__] = qlen;
            }
            kernel_(i__, n, &q[1], &d__[1], &l[1], false);
          }
          // update tree
          jj      = iperm[i__];
          out[jj] = k;
          pr[jj]  = j;
        }
      }
    }
    // if csp = rinf, no augmenting path is found
    if (csp != rinf) {
      // find augmenting path by tracing back in pr;
      // update iperm, jperm
      ++(*num);
      i__                   = irn[isp];
      iperm[i__]            = jsp;
      jperm[jsp]            = isp;
      j                     = jsp;
      const _Integer jstop_ = *num;
      for (_Integer __j_ = 1; __j_ <= jstop_; ++__j_) {
        jj = pr[j];
        if (jj == -1) break;
        k          = out[j];
        i__        = irn[k];
        iperm[i__] = jj;
        jperm[jj]  = k;
        j          = jj;
      }
      // update u for rows in q(up:n)
      for (kk = up; kk <= n; ++kk) {
        i__    = q[kk];
        u[i__] = u[i__] + d__[i__] - csp;
      }
    }
    for (kk = low; kk <= n; ++kk) {
      i__      = q[kk];
      d__[i__] = rinf;
      l[i__]   = 0;
    }
    for (kk = 1; kk <= qlen; ++kk) {
      i__      = q[kk];
      d__[i__] = rinf;
      l[i__]   = 0;
    }
  }

  // store u in d(1:n), handle singular case
  cleanup(n, num, iperm, jperm, irn, out, a, d__, u);
}

/*!
 * @}
 */

}  // namespace detail

namespace internal {
template <class V>
using std_vector = std::vector<V>;
}

/// \class Equilibrator
/// \tparam _Integer integer type
/// \tparam _Value value type
/// \tparam _Real real value type
/// \tparam _Container container metatype, default is \a std::vector
/// \brief object for equilibrating a CRS/CCS matrix
/// \ingroup pre
///
/// As to the container, besides it must be array, the following interfaces are
/// needed:
/// - \a data
///   -# normal version returns normal memory address
///   -# constant version returns memory address to constant variable
/// - \a resize, (re)allocate memory
/// - \a size, query the length of the container
/// - \a swap, used for \ref destroy memory space
/// - \a operator[], only constant version is needed for accessing entries
template <typename _Integer, typename _Value, typename _Real = _Value,
          template <class> class _Container = internal::std_vector>
class Equilibrator {
 public:
  using index_type  = _Integer;                ///< index type
  using value_type  = _Value;                  ///< scalar type
  using real_type   = _Real;                   ///< real type
  using iarray_type = _Container<index_type>;  ///< index array
  using array_type  = _Container<real_type>;   ///< real array
  using varray_type = _Container<value_type>;  ///< value array

  /// \brief equilibrate a CRS/CCS matrix
  /// \param[in] n number of nodes
  /// \param[in] nnz number of nonzeros
  /// \param[in] indptr index pointer array
  /// \param[in] indices index array
  /// \param[in] vals value array
  /// \param[out] perm permutation result
  /// \return information
  inline index_type compute(index_type n, index_type nnz,
                            const index_type *indptr, const index_type *indices,
                            const value_type *vals, index_type *perm) const {
    index_type pars[_PAR_NUM];
    detail::set_default_pars(pars);
#ifdef NDEBUG
    pars[3] = 1;
#endif
    _init(n, nnz);
    index_type info[_PAR_NUM], num, liwork = _iwork.size(),
                                    lwork = _work.size();
    detail::compute(n, nnz, (index_type *)indptr, (index_type *)indices,
                    (value_type *)vals, &num, perm, liwork,
                    (index_type *)_iwork.data(), lwork,
                    (real_type *)_work.data(), pars, info);
    return info[0];
  }

  /// \brief equilibrate a CRS/CCS matrix with container inputs
  /// \param[in] indptr index pointer array
  /// \param[in] indices index array
  /// \param[in] vals value array
  /// \param[out] perm permutation result
  /// \return information
  inline index_type compute(const iarray_type &indptr,
                            const iarray_type &indices, const varray_type &vals,
                            iarray_type &perm) const {
    if (indptr.size() > 1u) {
      if (indices.size() != vals.size()) return -100;
      if (perm.size() + 1 < indptr.size()) perm.resize(indptr.size() - 1);
      return compute(indptr.size() - 1, indices.size(), indptr.data(),
                     indices.data(), vals.data(), perm.data());
    }
    return 0;
  }

  /// \brief get the S scaling factor
  inline real_type s(const index_type i) const { return std::exp(_s[i]); }

  /// \brief get the T scaling factor
  inline real_type t(const index_type i) const { return std::exp(_t[i]); }

  /// \brief destroy internal workspace
  inline void destroy() {
    iarray_type().swap(_iwork);
    array_type().swap(_work);
    _s = _t = nullptr;
  }

 protected:
  mutable iarray_type      _iwork;  ///< integer work space
  mutable array_type       _work;   ///< real work space
  mutable const real_type *_s;      ///< left scaling
  mutable const real_type *_t;      ///< right scaling

 protected:
  /// \brief initialize work spaces
  /// \param[in] n number of nodes
  /// \param[in] nnz number of nonzeros
  inline void _init(const index_type n, const index_type nnz) const {
    _iwork.resize(n * 5);
    _work.resize(n * 3 + nnz);
    _s = _work.data();
    _t = _s + n;
  }
};
}  // namespace eql
}  // namespace hif

#endif  // _HIF_PRE_EQUILIBRATE_HPP
