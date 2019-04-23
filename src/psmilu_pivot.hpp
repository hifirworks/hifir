//@HEADER
//----------------------------------------------------------------------------
//                Copyright (C) 2019 The PSMILU AUTHORS
//----------------------------------------------------------------------------
//@HEADER

/// \file psmilu_pivot.hpp
/// \brief Finding pivoting candidates
/// \authors Qiao,

#ifndef _PSMILU_PIVOT_HPP
#define _PSMILU_PIVOT_HPP

#include <algorithm>

#include "psmilu_log.hpp"
#include "psmilu_utils.hpp"

namespace psmilu {

template <class CrsType, class CcsType, class PermType, class AugCrsType,
          class AugCcsType, class PosArray, class SparseArray>
inline void find_pvt_candidates(const typename PosArray::size_type step,
                                const typename PosArray::size_type m,
                                const CrsType &A_crs, const CcsType &A_ccs,
                                const PermType &p, const PermType &q,
                                const AugCrsType &U, const PosArray &U_start,
                                const AugCcsType &L, const PosArray &L_start,
                                SparseArray &                      ls,
                                const typename PosArray::size_type ns = 10) {
  using size_type  = typename PosArray::size_type;
  using index_type = typename CrsType::index_type;

  const size_type start = step + 1;
  const size_type last  = std::min(start + ns, m);

  // reset counter
  ls.reset_counter();

  if (start < last) {
    const size_type n = p.size();
    // register all candidates first
    for (size_type i(start); i < last; ++i) ls.push_back(i);

    // kernel for get the size of an entry
    const auto get_size = [&, step, n](const index_type i) -> size_type {
      // get size from A
      const size_type A_s = std::min(
          n - step, std::max(A_crs.nnz_in_row(p[i]), A_ccs.nnz_in_col(q[i])));
      size_type local_size(0);

      if (step) {
        index_type aug_id = L.start_row_id(i);
        while (!L.is_nil(aug_id)) {
          const size_type idx = L.col_idx(aug_id);
          if (idx < step - 1)
            local_size = std::max(
                local_size, size_type(U.row_start()[idx + 1] - U_start[idx]));
          else
            local_size = std::max(local_size, U.nnz_in_row(idx));
          aug_id = L.next_row_id(aug_id);
        }

        aug_id = U.start_col_id(i);
        while (!U.is_nil(aug_id)) {
          const size_type idx = U.row_idx(aug_id);
          if (idx < step - 1)
            local_size = std::max(
                local_size, size_type(L.col_start()[idx + 1] - L_start[idx]));
          else
            local_size = std::max(local_size, L.nnz_in_col(idx));
          aug_id = U.next_col_id(aug_id);
        }
      }
      return std::max(A_s, local_size);
    };
    const auto sort_kernel = [&](const index_type l,
                                 const index_type r) -> bool {
      return get_size(l) < get_size(r);
    };
    std::sort(ls.inds().begin(), ls.inds().begin() + ls.size(), sort_kernel);
  }
}
}  // namespace psmilu

#endif  // _PSMILU_PIVOT_HPP