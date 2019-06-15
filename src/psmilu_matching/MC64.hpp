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
  using mc64_mat_type             = CCS<value_type, int, true>;
  using size_type                 = typename mc64_mat_type::size_type;

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

  inline static mc64_mat_type ensure_input(
      const CCS<value_type, index_type, ONE_BASED> &A) {
    if (sizeof(index_type) == sizeof(int))
      if (ONE_BASED)
        return mc64_mat_type(A.nrows(), A.ncols(),
                             const_cast<int *>(reinterpret_cast<const int *>(
                                 A.col_start().data())),
                             const_cast<int *>(reinterpret_cast<const int *>(
                                 A.row_ind().data())),
                             const_cast<value_type *>(A.vals().data()), true);
    mc64_mat_type   B;
    const size_type n = A.ncols();
    B.resize(A.nrows(), n);
    psmilu_error_if(B.col_start().status() == DATA_UNDEF,
                    "memory allocation failed");
    const size_type nnz = A.nnz();
    B.row_ind().resize(nnz);
    psmilu_error_if(B.row_ind().status() == DATA_UNDEF,
                    "memory allocation failed");
    // IMPORTANT, ignore values
    B.vals() = A.vals();
    for (size_type i(0); i < n; ++i)
      B.col_start()[i + 1] = A.col_start()[i + 1] + !ONE_BASED;
    B.col_start().front() = 1;
    for (size_type i(0); i < nnz; ++i)
      B.row_ind()[i] = A.row_ind()[i] + !ONE_BASED;
    return B;
  }

  template <class PermType>
  inline static void do_matching(
      const CCS<value_type, index_type, ONE_BASED> &A, const int verbose,
      Array<index_type> &q, Array<value_type> &s, Array<value_type> &t) {
    constexpr static bool consist_int = sizeof(index_type) == sizeof(int);
    const size_type       m = A.nrows(), n = A.ncols(), nnz = A.nnz();
    psmilu_error_if(m != n, "must be squared systems");
    psmilu_assert(q.size() >= n, "invalid Q permutation size");
    psmilu_assert(s.size() >= m, "invalid S row scaling size");
    psmilu_assert(t.size() >= n, "invalid T column scaling size");

    const static int job = 5;

    int *         Q  = nullptr;
    mc64_mat_type A1 = ensure_input(A);
    if (consist_int)
      Q = reinterpret_cast<int *>(q.data());
    else {
      Q = new (std::nothrow) int[q.size()];
      psmilu_error_if(!Q, "memory allocation failed");
    }
    Array<int> iw(5 * n);
    psmilu_error_if(iw.status() == DATA_UNDEF, "memory allocation failed");
    Array<value_type> dw(3 * n + A.nnz());
    psmilu_error_if(dw.status() == DATA_UNDEF, "memory allocation failed");

    par_type icntl, info;
    set_default_controls(icntl, verbose);

    // call mc64
    int num;
    mc64(job, n, A1.nnz(), A1.col_start().data(), A1.row_ind().data(),
         A1.vals().data(), num, Q, 5 * n, iw.data(), 3 * n + A1.nnz(),
         dw.data(), icntl.data(), info.data());

    if (info[0] < 0) {
      if (!consist_int)
        if (Q) delete[] Q;
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

    // post processing
    // NOTE mc64 NOT return inverse permutation
    for (size_type i(0); i < n; ++i) q[i] = std::abs(Q[i]) - 1;
    if (job == 5) {
      for (size_type i(0); i < n; ++i) {
        s[i] = std::exp(dw[i]);
        t[i] = std::exp(dw[i + n]);
      }
    } else
      for (size_type i(0); i < n; ++i) s[i] = t[i] = 1;

    if (!consist_int)
      if (Q) delete[] Q;
  }
};

/*!
 * @}
 */

}  // namespace psmilu

#endif  // _PSMILU_MATCHING_MC64_HPP