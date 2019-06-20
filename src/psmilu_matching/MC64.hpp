//@HEADER
//----------------------------------------------------------------------------
//                Copyright (C) 2019 The PSMILU AUTHORS
//----------------------------------------------------------------------------
//@HEADER

/// \file psmilu_matching/MC64.hpp
/// \brief old (Fortran 77) MC64 Wrapper
/// \authors Qiao,

#ifndef _PSMILU_MATCHING_MC64_HPP
#define _PSMILU_MATCHING_MC64_HPP

#include "psmilu_fc_mangling.h"

#ifndef DOXYGEN_SHOULD_SKIP_THIS

extern "C" {
// routines to set default paramters
void PSMILU_FC(mc64i, MC64I)(int *);
void PSMILU_FC(mc64id, MC64ID)(int *);

// routine for computing the matching and scaling
void PSMILU_FC(mc64a, MC64A)(int *, int *, int *, int *, int *, float *, int *,
                             int *, int *, int *, int *, float *, int *, int *);
void PSMILU_FC(mc64ad, MC64AD)(int *, int *, int *, int *, int *, double *,
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

#include "psmilu_Array.hpp"
#include "psmilu_CompressedStorage.hpp"
#include "psmilu_Options.h"
#include "psmilu_log.hpp"
#include "psmilu_matching/common.hpp"
#include "psmilu_utils.hpp"

namespace psmilu {

#ifndef DOXYGEN_SHOULD_SKIP_THIS

template <bool IsDouble>
inline void set_default_controls(int *controls) {
  IsDouble ? PSMILU_FC(mc64id, MC64ID)(controls)
           : PSMILU_FC(mc64i, MC64I)(controls);
}

inline void mc64(int job, int n, int ne, const int *ip, const int *irn,
                 const float *a, int &num, int *cperm, int liw, int *iw,
                 int ldw, float *dw, int *icntl, int *info) {
  PSMILU_FC(mc64a, MC64A)
  (&job, &n, &ne, const_cast<int *>(ip), const_cast<int *>(irn),
   const_cast<float *>(a), &num, cperm, &liw, iw, &ldw, dw, icntl, info);
}

inline void mc64(int job, int n, int ne, const int *ip, const int *irn,
                 const double *a, int &num, int *cperm, int liw, int *iw,
                 int ldw, double *dw, int *icntl, int *info) {
  PSMILU_FC(mc64ad, MC64AD)
  (&job, &n, &ne, const_cast<int *>(ip), const_cast<int *>(irn),
   const_cast<double *>(a), &num, cperm, &liw, iw, &ldw, dw, icntl, info);
}

#endif  // DOXYGEN_SHOULD_SKIP_THIS

/*!
 * \addtogroup pre
 * @{
 */

template <class ValueType, class IndexType, bool OneBased>
class MC64 {
  static_assert(std::is_same<ValueType, double>::value ||
                    std::is_same<ValueType, float>::value,
                "MC64 only supports double or single");
  constexpr static bool _IS_DOUBLE = std::is_same<ValueType, double>::value;

 public:
  using value_type                = ValueType;            ///< value type
  using index_type                = IndexType;            ///< index type
  using par_type                  = std::array<int, 10>;  ///< parameter type
  constexpr static bool ONE_BASED = OneBased;
  using ccs_type  = CCS<value_type, index_type, ONE_BASED>;  ///< ccs type
  using crs_type  = typename ccs_type::other_type;           ///< crs type
  using size_type = typename ccs_type::size_type;

  inline static void set_default_controls(par_type &p, const int verbose) {
    psmilu::template set_default_controls<_IS_DOUBLE>(p.data());
    if (verbose == VERBOSE_NONE)
      p[0] = p[1] = -1;
    else if (psmilu_verbose2(PRE, verbose))
      p[2] = 1;
#ifdef NDEBUG
    p[3] = 1;
#endif
  }

  template <bool IsSymm>
  inline static void do_matching(const int verbose, crs_type &B,
                                 Array<index_type> &p, Array<index_type> &q,
                                 Array<value_type> &s, Array<value_type> &t,
                                 const bool do_iter = false) {
    constexpr static bool consist_int = sizeof(index_type) == sizeof(int);
    const size_type       n = B.nrows(), nnz = B.nnz();
    psmilu_error_if(B.nrows() != n, "must be squared systems");
    psmilu_assert(p.size() >= n, "invalid P permutation size");
    psmilu_assert(q.size() >= n, "invalid Q permutation size");
    psmilu_assert(s.size() >= n, "invalid S row scaling size");
    psmilu_assert(t.size() >= n, "invalid T column scaling size");

    const static int job = 5;

    do {
      Array<int> iw(5 * n);
      psmilu_error_if(iw.status() == DATA_UNDEF, "memory allocation failed");
      Array<value_type> dw(3 * n + nnz);
      psmilu_error_if(dw.status() == DATA_UNDEF, "memory allocation failed");
      par_type info, icntl;
      set_default_controls(icntl, verbose);
      constexpr static bool must_be_fortran_index = true;
      do_iter
          ? iterative_scale<IsSymm>(B, s, t, 1e-10, 5, must_be_fortran_index)
          : scale_extreme_values<IsSymm>(B, s, t, must_be_fortran_index);
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
        psmilu_error(
            "MC64 matching returned negative %d flag!\nIn addition, the 2nd "
            "info entry is of value %d.\nPlease refer MC64 documentation "
            "section 2.1.2 for error info interpretation.",
            info[0], info[1]);
      } else if (info[0] == 1) {
        psmilu_warning("MC64 the input matrix is structurally singular!");
      } else if (info[0] == 2) {
        psmilu_warning("MC64 the result scaling may cause overflow issue!");
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

}  // namespace psmilu

#endif  // _PSMILU_MATCHING_MC64_HPP