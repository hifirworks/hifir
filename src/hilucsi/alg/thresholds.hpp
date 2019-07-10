//@HEADER
//----------------------------------------------------------------------------
//                Copyright (C) 2019 The HILUCSI AUTHORS
//----------------------------------------------------------------------------
//@HEADER

/// \file hilucsi/alg/thresholds.hpp
/// \brief Implementation of scalability-oriented and inverse-based dual
///        thresholding for dropping
/// \authors Qiao,

#ifndef _HILUCSI_ALG_THRESHOLDS_HPP
#define _HILUCSI_ALG_THRESHOLDS_HPP

#include "hilucsi/utils/common.hpp"
#include "hilucsi/utils/log.hpp"

namespace hilucsi {

/// \brief apply numerical dropping based on dropping tolerance
/// \tparam KappaType data type for inverse norm
/// \tparam SpVecType sparse vector type, see \ref SparseVector
/// \param[in] tau dropping threshold parameter
/// \param[in] kappa inverse condition number of either L or U
/// \param[in,out] v sparse vector
/// \sa apply_space_dropping
/// \ingroup inv
template <class KappaType, class SpVecType>
inline void apply_num_dropping(const double tau, const KappaType kappa,
                               SpVecType &v) {
  using size_type = typename SpVecType::size_type;

  const KappaType coeff = tau / kappa;

  const size_type n = v.size();
  for (size_type i = 0u; i < n; ++i)
    if (std::abs(v.val(i)) <= coeff) v.mark_delete(i);
  v.compress_indices();  // NOTE sparse flags are reset here
}

/// \brief apply space limitation dropping
/// \tparam SpVecType sparse vector type, see \ref SparseVector
/// \param[in] nnz reference (local) number of nonzeros, i.e. input matrix
/// \param[in] alpha filling limiter
/// \param[in,out] v sparse vector
/// \param[in] start_size starting size, default is 0
/// \sa apply_num_dropping
/// \ingroup scl
template <class SpVecType>
inline void apply_space_dropping(
    const typename SpVecType::size_type nnz, const int alpha, SpVecType &v,
    const typename SpVecType::size_type start_size = 0u) {
  using size_type  = typename SpVecType::size_type;
  using index_type = typename SpVecType::index_type;
  using extractor  = internal::SpVInternalExtractor<SpVecType>;

  if (v.size()) {
    size_type N1 = alpha * nnz;
    if (start_size >= N1) N1 = start_size + 1;
    const size_type N = N1 - start_size;
    psmilu_assert(N != 0u, "zero number of limitation!");
    const size_type n = v.size();
    if (n > N) {
      // we need to extract the N values with largest mag of values, in other
      // words, the rest any entry in the rest n2-N is smaller the extracted
      // entries in terms of mag

      auto &      inds  = v.inds();  // std::vector
      const auto &vals  = v.vals();  // std::vector
      auto        first = inds.begin(), last = first + n;

      if (N == 1u)
        // special case, not sure if nth_element can have nth == first
        std::iter_swap(first,
                       std::max_element(
                           first, last,
                           [&](const index_type i, const index_type j) -> bool {
                             return std::abs(vals[i]) < std::abs(vals[j]);
                           }));
      else
        // using c++ built-in selection/partition to find the n largest
        // values. Note that it's most likely that the internal implementation
        // is using introselect, where the average complexity is O(n2), worst
        // cast O(n2 log n2)
        //
        // explain of the routine (basically the lambda CMP):
        // after this routine returns, the following relation is satisfied:
        // for each j in [first, first+N) and i in [first+N,last)
        //  the lambda CMP(i,j)==false,
        // in other words, any entry in the first part is an upper bound for the
        // second part with the following impl
        std::nth_element(first, first + N - 1, last,
                         [&](const index_type i, const index_type j) -> bool {
                           return std::abs(vals[i]) > std::abs(vals[j]);
                         });

      // directly modify the internal counter O(1)
      static_cast<extractor &>(v).counts() = N;
    }
  }
}

}  // namespace hilucsi

#endif  // _HILUCSI_ALG_THRESHOLDS_HPP