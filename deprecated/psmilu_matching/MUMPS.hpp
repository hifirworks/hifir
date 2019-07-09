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
#include "psmilu_matching/common.hpp"

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
  using crs_type  = typename ccs_type::other_type;           ///< crs type
  using size_type = typename ccs_type::size_type;            ///< size type

  template <bool IsSymm>
  inline static void do_matching(const int /* verbose */, crs_type &B,
                                 Array<index_type> &p, Array<index_type> &q,
                                 Array<value_type> &s, Array<value_type> &t,
                                 const int pre_scale = 0) {
    constexpr static bool consist_int = sizeof(index_type) == sizeof(int);

    psmilu_error_if(B.nrows() != B.ncols(),
                    "only squared systems are supported");

    const size_type n = B.nrows(), nnz = B.nnz();
    psmilu_assert(p.size() >= n, "invalid P permutation size");
    psmilu_assert(q.size() >= n, "invalid Q permutation size");
    psmilu_assert(s.size() >= n, "invalid S row scaling size");
    psmilu_assert(t.size() >= n, "invalid T column scaling size");

    do {
      constexpr static bool must_be_fortran_index = true;
      Array<value_type>     buf_s(n), buf_t(n);
      psmilu_error_if(
          buf_s.status() == DATA_UNDEF || buf_t.status() == DATA_UNDEF,
          "memory allocation failed");
      int *indptr(nullptr), *indices(nullptr), *P(nullptr);
      switch (pre_scale) {
        case 0:
          scale_eye(B, s, t, must_be_fortran_index);
          break;
        case 1:
          scale_extreme_values<IsSymm>(B, s, t, must_be_fortran_index);
          break;
        default:
          iterative_scale<IsSymm>(B, s, t, 1e-10, 5, must_be_fortran_index);
          break;
      }
      // TODO check integer overflows
      indptr  = ensure_type_consistency<int>(B.row_start());
      indices = ensure_type_consistency<int>(B.col_ind());
      P       = ensure_type_consistency<int>(p, false);
      // call matching on transpose
      mumps_match(n, nnz, P, indptr, indices, B.vals().data(), buf_t.data(),
                  buf_s.data());
      if (!consist_int) {
        std::copy_n(P, n, p.begin());
        if (indptr) delete[] indptr;
        if (indices) delete[] indices;
        if (P) delete[] P;
      }
      for (size_type i(0); i < n; ++i) {
        s[i] *= buf_s[i];
        t[i] *= buf_t[i];
      }
    } while (false);  // bufs freed here

    // NOTE that it's 1-based index!
    for (size_type i(0); i < n; ++i) --p[i];

    // handle symmetric
    if (IsSymm) {
      std::copy_n(p.cbegin(), n, q.begin());
      for (size_type i(0); i < n; ++i) s[i] = std::sqrt(std::abs(s[i] * t[i]));
      std::copy_n(s.cbegin(), n, t.begin());
    } else
      for (size_type i(0); i < n; ++i) q[i] = i;
  }
};

/*!
 * @}
 */

}  // namespace psmilu

#endif  // _PSMILU_MATCHING_MUMPS_HPP