//@HEADER
//----------------------------------------------------------------------------
//                Copyright (C) 2019 The PSMILU AUTHORS
//----------------------------------------------------------------------------
//@HEADER

/// \file psmilu_inv_thres.hpp
/// \brief Implementation of inverse-based thresholding for dropping
/// \authors Qiao,

#ifndef _PSMILU_INVTHRES_HPP
#define _PSMILU_INVTHRES_HPP

#include <algorithm>

#include "psmilu_log.hpp"
#include "psmilu_utils.hpp"

namespace psmilu {

/// \brief apply dropping and sort the resulting index list
/// \tparam TauType data type for dropper parameter \f$\tau\f$, e.g. \a double
/// \tparam KappaType data type for condition number \f$\kappa\f$
/// \tparam SpVecType sparse vector type, see \ref SparseVector
/// \param[in] tau dropping threshold parameter
/// \param[in] kappa condition number of either L or U
/// \param[in] nnz reference (local) number of nonzeros, i.e. input matrix
/// \param[in] alpha filling limiter
/// \param[in,out] v sparse vector
/// \param[in] start_size starting size
/// \return the intermediate vector size before applying final limitation
/// \note The returned parameter is mainly for unit-testing/debugging
/// \ingroup inv
///
/// This subroutine is to apply the dropping and, then, sort the remaining
/// index list. Mathematically, there are three steps:
///
/// \b Step1, apply numerical dropping to ensure each entry is no smaller than
/// certain threshold, which is carried out by \f$\frac{\tau}{\kappa}\f$. By
/// "no smaller", we mean the magnitude.
///
/// \f[
///   \left|v_i\right|>\frac{\tau}{\kappa}
/// \f]
///
/// Note that the complexity for first step is \f$\mathcal{O}(n_1)\f$, where
/// \f$n_1\f$ is the number of nonzeros in v as original input.
///
/// \b Step2, once we have done the previous step, we compress the list, thus
/// the new problem size is \f$n_2\f$. Now, compute our size limiter to ensure
/// that the list size is well-bounded!
///
/// \f[
///   n_3=\alpha\times\textrm{nnz}
/// \f]
///
/// if \f$n_2>n_3\f$, then we need to perform further dropping to ensure the
/// size is well-bounded for optimal time complexity. The strategy is
/// straightforward, i.e. we choose first \f$n_3\f$ entries with largest values.
/// However, as regards of implementation, this requires a selection/partition
/// algorithm, which must be taken special care. Currently, we utilize standard
/// c++ implementation \a std::nth_element to perform the partition, which,
/// usually, is implemented with \a introselect algorithm thus has linear
/// average performance. Be aware that \a introselect has worst case performance
/// of \f$\mathcal{O}(n_2\log n_2)\f$.
///
/// For this step, the complexity is either empty,
/// \f$\mathcal{O}(n_2\log n_2)\f$, or \f$\mathcal{O}(n_2)\f$ (most cases).
///
/// \b Step3, after previous step, we should have a list with size of
/// \f$\min(n_2,n_3)\f$, say \f$m\f$. We just need to sort the tiny list, which
/// takes \f$\mathcal{O}(m\log m)\f$; this bound has become strict by calling
/// \a std::sort \b since C++11 (prior to C++11, this is average performance).
///
/// Therefore, overall, dropping and sorting, on average, cost:
///
/// \f[
///   \mathcal{O}(n_1+n_2+n_3\log n_3)\sim\mathcal{O}(n_1+n_3\log n_3)
/// \f]
///
/// which aligns with our analysis in the paper. In the worst situation, it may
/// become:
///
/// \f[
///   \mathcal{O}(n_1+n_2\log n_2+n_3\log n_3)
/// \f]
///
/// But this is less a concern in practice.
template <class TauType, class KappaType, class SpVecType>
inline typename SpVecType::size_type apply_dropping_and_sort(
    const TauType tau, const KappaType kappa,
    const typename SpVecType::size_type nnz, const int alpha, SpVecType &v,
    const typename SpVecType::size_type start_size = 0u) {
  using size_type                 = typename SpVecType::size_type;
  using index_type                = typename SpVecType::index_type;
  using extractor                 = internal::SpVInternalExtractor<SpVecType>;
  constexpr static bool ONE_BASED = SpVecType::ONE_BASED;

  psmilu_assert(tau != TauType(), "zero threshold tau!");
  const size_type N1 = alpha * nnz;
  if (start_size >= N1) {
    psmilu_warning(
        "inv-thres start size %zd exceeds bound (alpha*nnz) %zd, drop all "
        "entries!",
        start_size, N1);
    static_cast<extractor &>(v).counts() = 0u;
    return 0u;
  }
  const size_type N = N1 - start_size;
  psmilu_assert(N != 0u, "zero number of limitation!");
  // filter coeff based on inverse-based thresholding
  // TODO need to test potential float overflow?
  const KappaType coeff = tau / kappa;
  const auto      n1    = v.size();
  // O(n1)
  for (size_type i = 0u; i < n1; ++i)
    if (std::abs(v.val(i)) <= coeff) v.mark_delete(i);
  // O(n1)
  v.compress_indices();  // NOTE sparse flags are reset here
  // NOTE, v, now, has a new size
  const auto n2 = v.size();
  if (n2 > N) {
    // we need to extract the N values with largest mag of values, in other
    // words, the rest any entry in the rest n2-N is smaller the extracted
    // entries in terms of mag

    auto &      inds  = v.inds();  // std::vector
    const auto &vals  = v.vals();  // std::vector
    auto        first = inds.begin(), last = first + n2;
    const auto  to_c = [](const index_type i) -> index_type {
      return to_c_idx<index_type, ONE_BASED>(i);
    };

    if (N == 1u)
      // special case, not sure if nth_element can have nth == first
      std::iter_swap(
          first,
          std::max_element(
              first, last, [&](const index_type i, const index_type j) -> bool {
                return std::abs(vals[to_c(i)]) > std::abs(vals[to_c(j)]);
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
      //
      std::nth_element(first, first + N - 1, last,
                       [&](const index_type i, const index_type j) -> bool {
                         return std::abs(vals[to_c(i)]) >
                                std::abs(vals[to_c(j)]);
                       });

    // directly modify the internal counter O(1)
    static_cast<extractor &>(v).counts() = N;
  }
  // sort indices, O(min(n2,N)log(min(n2,N))), note that we assume min(n2,N)
  // is constant for PDE problems
  v.sort_indices();
  return n2;  // mainly for unit testing purpose
}

}  // namespace psmilu

#endif  // _PSMILU_INVTHRES_HPP
