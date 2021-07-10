///////////////////////////////////////////////////////////////////////////////
//                  This file is part of the HIFIR library                   //
///////////////////////////////////////////////////////////////////////////////

/*!
 * \file hif/utils/coo2compress.hpp
 * \brief Helper function to convert coordinate to compressed format
 * \author Qiao Chen

\verbatim
Copyright (C) 2021 NumGeom Group at Stony Brook University

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
\endverbatim

 */

#ifndef _HIF_UTILS_COO2COMPRESS_HPP
#define _HIF_UTILS_COO2COMPRESS_HPP

#include <algorithm>

#include "hif/ds/Array.hpp"

namespace hif {

/// \brief Convert a coordinate sparse format to compressed storage
/// \tparam IndPtrType Index pointer data type
/// \tparam IndexType Index data type
/// \tparam ValueType Value data type
/// \param[in] primary_size Size in the compressed axis, e.g., nrows for CRS
/// \param[in] secondary_size Size in the uncompressed axis, e.g., ncols for CRS
/// \param[in] pidx Primary index list, e.g., rows for CRS
/// \param[in] sids Secondary index list, e.g., cols for CRS
/// \param[in] coovals Coo format values
/// \param[out] ind_start Index pointer
/// \param[out] indices Index list
/// \param[out] vals Values corresponding to \a indices
///
/// For CRS, call this function as convert_coo2crs(nrows, ncols, rows, cols,
/// coovals, rowptr, colind, vals). For CCS, call this function as
/// convert_coo2crs(ncols, nrows, cols, rows, coovals, colptr, rowind, vals).
template <class IndPtrType, class IndexType, class ValueType>
inline void convert_coo2cs(const std::size_t       primary_size,
                           const std::size_t       secondary_size,
                           const Array<IndexType> &pidx,
                           const Array<IndexType> &sids,
                           const Array<ValueType> &coovals,
                           Array<IndPtrType> &     ind_start,
                           Array<IndexType> &indices, Array<ValueType> &vals) {
  using size_type = typename Array<IndexType>::size_type;

  ind_start.resize(primary_size + 1);
  hif_error_if(ind_start.status() == DATA_UNDEF, "memory allocation failed");
  std::fill(ind_start.begin(), ind_start.end(), IndPtrType(0));
  const size_type nnz = pidx.size();
  hif_assert(nnz == sids.size(), "unmatched sizes");
  hif_assert(nnz == coovals.size(), "unmatched sizes");

  // determine structure first
  for (size_type i(0); i < nnz; ++i) ++ind_start[pidx[i] + 1];
  // accumulate
  for (size_type i(0); i < primary_size; ++i) ind_start[i + 1] += ind_start[i];
  hif_assert(nnz == (size_type)ind_start[primary_size], "fatal bug");

  // assemble to compressed storage
  indices.resize(nnz);
  hif_error_if(indices.status() == DATA_UNDEF, "memory allocation failed");
  vals.resize(nnz);
  hif_error_if(vals.status() == DATA_UNDEF, "memory allocation failed");
  for (size_type i(0); i < nnz; ++i) {
    const auto pos = ind_start[pidx[i]];
    indices[pos]   = sids[i];
    vals[pos]      = coovals[i];
    ++ind_start[pidx[i]];
  }

  // reset ind_start
  for (size_type i(primary_size); i != 0u; --i) ind_start[i] = ind_start[i - 1];
  ind_start[0] = 0u;

  // sort locally
  Array<ValueType> vbuf;
  for (size_type i(0); i < primary_size; ++i)
    if (!std::is_sorted(indices.cbegin() + ind_start[i],
                        indices.cbegin() + ind_start[i + 1])) {
      // not sorted
      vbuf.resize(secondary_size);
      hif_error_if(vbuf.status() == DATA_UNDEF, "memory allocation failed");
      // backup values
      for (size_type k = ind_start[i]; k < (size_type)ind_start[i + 1]; ++k)
        vbuf[indices[k]] = vals[k];
      // sort
      std::sort(indices.begin() + ind_start[i],
                indices.begin() + ind_start[i + 1]);
      // copy back
      for (size_type k = ind_start[i]; k < (size_type)ind_start[i + 1]; ++k)
        vals[k] = vbuf[indices[k]];
    }
}

}  // namespace hif

#endif  // _HIF_UTILS_COO2COMPRESS_HPP
