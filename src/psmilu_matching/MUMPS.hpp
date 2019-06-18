//@HEADER
//----------------------------------------------------------------------------
//                Copyright (C) 2019 The PSMILU AUTHORS
//----------------------------------------------------------------------------
//@HEADER

/// \file psmilu_matching/MUMPS.hpp
/// \brief mumps matching interface written in Fortran77
/// \authors Qiao,

#ifndef _PSMILU_MATCHING_MUMPS_HPP
#define _PSMILU_MATCHING_MUMPS_HPP

#include "psmilu_fc_mangling.h"

#ifndef DOXYGEN_SHOULD_SKIP_THIS

extern "C" {
void PSMILU_FC(dmumps_match, DMUMPS_MATCH)(int *, int *, int *, int *, int *,
                                           double *, double *, double *);
void PSMILU_FC(smumps_match, SMUMPS_MATCH)(int *, int *, int *, int *, int *,
                                           float *, float *, float *);
}

namespace psmilu {

inline void mumps_match(int n, int nnz, int *p, const int *indptr,
                        const int *indices, const double *vals, double *s,
                        double *t) {
  PSMILU_FC(dmumps_match, DMUMPS_MATCH)
  (&n, &nnz, p, const_cast<int *>(indptr), const_cast<int *>(indices),
   const_cast<double *>(vals), s, t);
}

inline void mumps_match(int n, int nnz, int *p, const int *indptr,
                        const int *indices, const float *vals, float *s,
                        float *t) {
  PSMILU_FC(smumps_match, SMUMPS_MATCH)
  (&n, &nnz, p, const_cast<int *>(indptr), const_cast<int *>(indices),
   const_cast<float *>(vals), s, t);
}

}  // namespace psmilu

#endif  // DOXYGEN_SHOULD_SKIP_THIS

#include <algorithm>
#include <cmath>
#include <new>
#include <type_traits>

#include "psmilu_Array.hpp"
#include "psmilu_CompressedStorage.hpp"
#include "psmilu_log.hpp"

namespace psmilu {

/*!
 * \addtogroup pre
 * @{
 */
template <class ValueType, class IndexType, bool OneBased>
class MUMPS {
 public:
  using value_type                = ValueType;               ///< value type
  using index_type                = IndexType;               ///< index type
  constexpr static bool ONE_BASED = OneBased;                ///< index base
  using ccs_type  = CCS<value_type, index_type, ONE_BASED>;  ///< ccs input
  using size_type = typename ccs_type::size_type;            ///< size type

  template <bool IsSymm>
  inline static void do_matching(
      const CCS<value_type, index_type, ONE_BASED> &A, const int /* verbose */,
      Array<index_type> &p, Array<index_type> &q, Array<value_type> &s,
      Array<value_type> &t) {
    constexpr static bool consist_int = sizeof(index_type) == sizeof(int);
    constexpr static bool alloc_mem   = !consist_int || !ONE_BASED;

    psmilu_error_if(A.nrows() != A.ncols(),
                    "only squared systems are supported");

    const size_type n = A.nrows(), nnz = A.nnz();
    psmilu_assert(p.size() >= n, "invalid P permutation size");
    psmilu_assert(q.size() >= n, "invalid Q permutation size");
    psmilu_assert(s.size() >= n, "invalid S row scaling size");
    psmilu_assert(t.size() >= n, "invalid T column scaling size");

    int *indptr(nullptr), *indices(nullptr);

    // TODO check integer overflows
    if (!alloc_mem) {
      indptr  = (int *)A.col_start().data();
      indices = (int *)A.row_ind().data();
    } else {
      const size_type np1 = n + 1;
      indptr              = new (std::nothrow) int[np1];
      psmilu_error_if(!indptr, "memory allocation failed");
      indices = new (std::nothrow) int[nnz];
      psmilu_error_if(!indices, "memory allocation failed");
      for (size_type i(0); i < np1; ++i)
        indptr[i] = A.col_start()[i] + !ONE_BASED;
      for (size_type i(0); i < nnz; ++i)
        indices[i] = A.row_ind()[i] + !ONE_BASED;
    }

    // call matching
    mumps_match(n, nnz, q.data(), indptr, indices, A.vals().data(), s.data(),
                t.data());

    if (alloc_mem) {
      if (indptr) delete[] indptr;
      if (indices) delete[] indices;
    }

    // NOTE that it's 1-based index!
    for (size_type i(0); i < n; ++i) --q[i];

    // handle symmetric
    if (IsSymm) {
      std::copy_n(q.cbegin(), n, p.begin());
      // blend the scaling into sqrt(s*t)
      for (size_type i(0); i < n; ++i) s[i] = std::sqrt(std::abs(s[i] * t[i]));
      std::copy_n(s.cbegin(), n, t.begin());
    } else
      for (size_type i(0); i < n; ++i) p[i] = i;
  }
};

/*!
 * @}
 */

}  // namespace psmilu

#endif  // _PSMILU_MATCHING_MUMPS_HPP