///////////////////////////////////////////////////////////////////////////////
//                This file is part of HILUCSI project                       //
///////////////////////////////////////////////////////////////////////////////

/*!
 * \file hilucsi/pre/MC64.hpp
 * \brief old (Fortran 77) MC64 Wrapper
 * \authors Qiao,
 * \warning This file will be removed in the future with our own implementation

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

#ifndef _HILUCSI_PRE_MC64_HPP
#define _HILUCSI_PRE_MC64_HPP

#include "hilucsi/utils/fc_mangling.hpp"

#ifndef DOXYGEN_SHOULD_SKIP_THIS

extern "C" {
// routines to set default paramters
void HILUCSI_FC(mc64i, MC64I)(int *);
void HILUCSI_FC(mc64id, MC64ID)(int *);

// routine for computing the matching and scaling
void HILUCSI_FC(mc64a, MC64A)(int *, int *, int *, int *, int *, float *, int *,
                              int *, int *, int *, int *, float *, int *,
                              int *);
void HILUCSI_FC(mc64ad, MC64AD)(int *, int *, int *, int *, int *, double *,
                                int *, int *, int *, int *, int *, double *,
                                int *, int *);
}

#endif  // DOXYGEN_SHOULD_SKIP_THIS

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <new>
#include <type_traits>

#include "hilucsi/Options.h"
#include "hilucsi/ds/Array.hpp"
#include "hilucsi/ds/CompressedStorage.hpp"
#include "hilucsi/pre/a_priori_scaling.hpp"
#include "hilucsi/utils/common.hpp"
#include "hilucsi/utils/log.hpp"

namespace hilucsi {

#ifndef DOXYGEN_SHOULD_SKIP_THIS

template <bool IsDouble>
inline void set_default_controls(int *controls) {
  IsDouble ? HILUCSI_FC(mc64id, MC64ID)(controls)
           : HILUCSI_FC(mc64i, MC64I)(controls);
}

inline void mc64(int job, int n, int ne, const int *ip, const int *irn,
                 const float *a, int &num, int *cperm, int liw, int *iw,
                 int ldw, float *dw, int *icntl, int *info) {
  HILUCSI_FC(mc64a, MC64A)
  (&job, &n, &ne, const_cast<int *>(ip), const_cast<int *>(irn),
   const_cast<float *>(a), &num, cperm, &liw, iw, &ldw, dw, icntl, info);
}

inline void mc64(int job, int n, int ne, const int *ip, const int *irn,
                 const double *a, int &num, int *cperm, int liw, int *iw,
                 int ldw, double *dw, int *icntl, int *info) {
  HILUCSI_FC(mc64ad, MC64AD)
  (&job, &n, &ne, const_cast<int *>(ip), const_cast<int *>(irn),
   const_cast<double *>(a), &num, cperm, &liw, iw, &ldw, dw, icntl, info);
}

#endif  // DOXYGEN_SHOULD_SKIP_THIS

/*!
 * \addtogroup pre
 * @{
 */

template <class ValueType, class IndexType>
class MC64 {
  static_assert(std::is_same<ValueType, double>::value ||
                    std::is_same<ValueType, float>::value,
                "MC64 only supports double or single");
  constexpr static bool _IS_DOUBLE = std::is_same<ValueType, double>::value;
  ///< internal flag to check if the input data type is \a double

 public:
  using value_type = ValueType;                      ///< value type
  using index_type = IndexType;                      ///< index type
  using par_type   = std::array<int, 10>;            ///< parameter type
  using ccs_type   = CCS<value_type, index_type>;    ///< ccs type
  using crs_type   = typename ccs_type::other_type;  ///< crs type
  using size_type  = typename ccs_type::size_type;

  inline static void set_default_controls(par_type &p, const int verbose) {
    hilucsi::template set_default_controls<_IS_DOUBLE>(p.data());
    if (verbose == VERBOSE_NONE)
      p[0] = p[1] = -1;
    else if (hilucsi_verbose2(PRE, verbose))
      p[2] = 1;
#ifdef NDEBUG
    p[3] = 1;
#endif
  }

  template <bool IsSymm>
  inline static void do_matching(const int verbose, crs_type &B,
                                 Array<index_type> &p, Array<index_type> &q,
                                 Array<value_type> &s, Array<value_type> &t,
                                 const int pre_scale = 0) {
    constexpr static bool consist_int = sizeof(index_type) == sizeof(int);
    const size_type       n = B.nrows(), nnz = B.nnz();
    hilucsi_error_if(B.nrows() != n, "must be squared systems");
    hilucsi_assert(p.size() >= n, "invalid P permutation size");
    hilucsi_assert(q.size() >= n, "invalid Q permutation size");
    hilucsi_assert(s.size() >= n, "invalid S row scaling size");
    hilucsi_assert(t.size() >= n, "invalid T column scaling size");

    const static int job = 5;

    do {
      Array<int> iw(5 * n);
      hilucsi_error_if(iw.status() == DATA_UNDEF, "memory allocation failed");
      Array<value_type> dw(3 * n + nnz);
      hilucsi_error_if(dw.status() == DATA_UNDEF, "memory allocation failed");
      par_type info, icntl;
      set_default_controls(icntl, verbose);
      switch (pre_scale) {
        case 0:
          scale_eye(B, s, t);
          break;
        case 1:
          scale_extreme_values<IsSymm>(B, s, t);
          break;
        default:
          iterative_scale<IsSymm>(B, s, t, 1e-10, 5);
          break;
      }
      int *indptr(nullptr), *indices(nullptr), *P(nullptr);
      indptr  = ensure_type_consistency<int>(B.row_start());
      indices = ensure_type_consistency<int>(B.col_ind());
      P       = ensure_type_consistency<int>(p, false);

      // call mc64 on transpose
      int num;
      mc64(job, n, nnz, indptr, indices, B.vals().data(), num, P, iw.size(),
           iw.data(), dw.size(), dw.data(), icntl.data(), info.data());

      if (info[0] < 0) {
        if (!consist_int) {
          if (indptr) delete[] indptr;
          if (indices) delete[] indices;
          if (P) delete[] P;
        }
        hilucsi_error(
            "MC64 matching returned negative %d flag!\nIn addition, the 2nd "
            "info entry is of value %d.\nPlease refer MC64 documentation "
            "section 2.1.2 for error info interpretation.",
            info[0], info[1]);
      } else if (info[0] == 1) {
        hilucsi_warning("MC64 the input matrix is structurally singular!");
      } else if (info[0] == 2) {
        hilucsi_warning("MC64 the result scaling may cause overflow issue!");
      }

      if (!consist_int) {
        std::copy_n(P, n, p.begin());
        if (indptr) delete[] indptr;
        if (indices) delete[] indices;
        if (P) delete[] P;
      }

      // update scaling if necessary
      if (job == 5) {
        for (size_type i(0); i < n; ++i) {
          s[i] *= std::exp(dw[i + n]);
          t[i] *= std::exp(dw[i]);
        }
      }
    } while (false);  // bufs freed here

    // post processing
    // NOTE mc64 NOT return inverse permutation
    for (size_type i(0); i < n; ++i) p[i] = std::abs(p[i]) - 1;

    // handle symmetric
    if (IsSymm) {
      std::copy_n(p.cbegin(), n, q.begin());
      for (size_type i(0); i < n; ++i) s[i] = std::sqrt(s[i] * t[i]);
      std::copy_n(s.cbegin(), n, t.begin());
    } else
      for (size_type i(0); i < n; ++i) q[i] = i;
  }
};

/*!
 * @}
 */

}  // namespace hilucsi

#endif  // _HILUCSI_PRE_MC64_HPP