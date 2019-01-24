//@HEADER
//----------------------------------------------------------------------------
//                Copyright (C) 2019 The PSMILU AUTHORS
//----------------------------------------------------------------------------
//@HEADER

/// \file psmilu_lapack/lapack.hpp
/// \brief PSMILU LAPACK interface
/// \authors Qiao,

#ifndef _PSMILU_LAPACK_LAPACK_HPP
#define _PSMILU_LAPACK_LAPACK_HPP

#include <algorithm>
#include <type_traits>
#include <vector>

#include "psmilu_DenseMatrix.hpp"
#include "psmilu_lapack/lup.hpp"
#include "psmilu_log.hpp"

namespace psmilu {

/// \class Lapack
/// \tparam ValueType value type, e.g. \a double
/// \tparam IntType integer type, default is \ref psmilu_lapack_int
/// \note If the \a IntType and psmilu_lapack_int have different sizes, then
///       intermidiate buffers might be created.
template <class ValueType, class IntType = psmilu_lapack_int>
class Lapack {
 public:
  typedef IntType     int_type;    ///< integer type
  typedef ValueType   value_type;  ///< value type
  typedef value_type *pointer;     ///< pointer type

 private:
  constexpr static bool _INT_TYPE_CONSIS =
      sizeof(int_type) == sizeof(psmilu_lapack_int);
  static_assert(std::is_integral<int_type>::value, "must be integer type!");

 public:
  template <typename T = int>
  inline typename std::enable_if<_INT_TYPE_CONSIS, T>::type getrf(
      const int_type n, pointer a, const int_type lda, int_type *ipiv) const {
    return internal::getrf(n, n, a, lda, (psmilu_lapack_int *)ipiv);
  }

  template <typename T = int>
  inline typename std::enable_if<!_INT_TYPE_CONSIS, T>::type getrf(
      const int_type n, pointer a, const int_type lda, int_type *ipiv) const {
    std::vector<psmilu_lapack_int> buf(n);
    const int info = internal::getrf(n, n, a, lda, buf.data());
    std::copy(buf.cbegin(), buf.cend(), ipiv);
    return info;
  }

  inline void getrf(DenseMatrix<value_type> &a, int_type *ipiv) const {
    const int info = getrf(a.nrows(), a.data(), a.nrows(), ipiv);
    (void)info;
  }
};

}  // namespace psmilu

#endif  // _PSMILU_LAPACK_LAPACK_HPP
