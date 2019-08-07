//@HEADER
//----------------------------------------------------------------------------
//                Copyright (C) 2019 The HILUCSI AUTHORS
//----------------------------------------------------------------------------
//@HEADER

/// \file hilucsi/pre/EqlDriver.hpp
/// \brief Equilibrator driver interface
/// \authors Qiao,

#ifndef _HILUCSI_PRE_EQL_DRIVER_HPP
#define _HILUCSI_PRE_EQL_DRIVER_HPP

#include <algorithm>
#include <cmath>
#include <limits>
#include <new>
#include <type_traits>

#include "hilucsi/ds/Array.hpp"
#include "hilucsi/ds/CompressedStorage.hpp"
#include "hilucsi/pre/a_priori_scaling.hpp"
#include "hilucsi/pre/equilibrate.hpp"
#include "hilucsi/utils/common.hpp"
#include "hilucsi/utils/log.hpp"

namespace hilucsi {

/// \class EqlDriver
/// \brief driver class wrapped around \ref eql::Equilibrator
/// \tparam ValueType value type
/// \tparam IndexType index type, e.g. \a int
/// \ingroup pre
template <class ValueType, class IndexType>
class EqlDriver {
 public:
  using value_type  = ValueType;                      ///< value type
  using index_type  = IndexType;                      ///< index type
  using ccs_type    = CCS<value_type, index_type>;    ///< ccs type
  using crs_type    = typename ccs_type::other_type;  ///< crs type
  using size_type   = typename ccs_type::size_type;   ///< size type
  using kernel_type = eql::Equilibrator<index_type, value_type, Array>;
  ///< kernel type

  template <bool IsSymm>
  inline static void do_matching(const int /* verbose */, crs_type &B,
                                 Array<index_type> &p, Array<index_type> &q,
                                 Array<value_type> &s, Array<value_type> &t,
                                 const int pre_scale = 0) {
    const size_type n = B.nrows(), nnz = B.nnz();
    hilucsi_error_if(B.nrows() != n, "must be squared systems");
    hilucsi_assert(p.size() >= n, "invalid P permutation size");
    hilucsi_assert(q.size() >= n, "invalid Q permutation size");
    hilucsi_assert(s.size() >= n, "invalid S row scaling size");
    hilucsi_assert(t.size() >= n, "invalid T column scaling size");

    do {
      // check if pre-scaling is requested
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
      // deal with scaling and matching
      kernel_type eq;
      const int   info =
          eq.compute(n, nnz, B.row_start().data(), B.col_ind().data(),
                     B.vals().data(), p.data());

      if (info < 0) {
        hilucsi_error(
            "MC64 matching returned negative %d flag!\n"
            "Please refer MC64 documentation section 2.1.2 for error info "
            "interpretation.",
            info);
      } else if (info == 1) {
        hilucsi_warning("MC64 the input matrix is structurally singular!");
      } else if (info == 2) {
        hilucsi_warning("MC64 the result scaling may cause overflow issue!");
      }

      // update scaling if necessary, NOTE we pass in CRS, thus it matching
      // rescaling on the transpose of CCS
      for (size_type i(0); i < n; ++i) {
        s[i] *= eq.t(i);
        t[i] *= eq.s(i);
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

}  // namespace hilucsi

#endif  // _HILUCSI_PRE_EQL_DRIVER_HPP