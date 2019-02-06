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

#include "psmilu_Array.hpp"
#include "psmilu_DenseMatrix.hpp"
#include "psmilu_log.hpp"

#include "lup.hpp"
#include "qrcp.hpp"
#include "trsv.hpp"

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
  /// \name LU
  ///@{

  template <typename T = psmilu_lapack_int>
  inline static typename std::enable_if<_INT_TYPE_CONSIS, T>::type getrf(
      const int_type n, pointer a, const int_type lda, int_type *ipiv) {
    return internal::getrf(n, n, a, lda, (psmilu_lapack_int *)ipiv);
  }

  template <typename T = psmilu_lapack_int>
  inline static typename std::enable_if<!_INT_TYPE_CONSIS, T>::type getrf(
      const int_type n, pointer a, const int_type lda, int_type *ipiv) {
    std::vector<psmilu_lapack_int> buf(n);
    const psmilu_lapack_int info = internal::getrf(n, n, a, lda, buf.data());
    std::copy(buf.cbegin(), buf.cend(), ipiv);
    return info;
  }

  inline static int_type getrf(DenseMatrix<value_type> &a,
                               Array<int_type> &        ipiv) {
    psmilu_assert(a.nrows() == ipiv.size(),
                  "row size should match permutation vector length");
    return getrf(a.nrows(), a.data(), a.nrows(), ipiv.data());
  }

  template <typename T = psmilu_lapack_int>
  inline static typename std::enable_if<_INT_TYPE_CONSIS, T>::type getrs(
      const char tran, const int_type n, const int_type nrhs,
      const value_type *a, const int_type lda, const int_type *ipiv, pointer b,
      const int_type ldb) {
    return internal::getrs(tran, n, nrhs, a, lda, ipiv, b, ldb);
  }

  template <typename T = psmilu_lapack_int>
  inline static typename std::enable_if<!_INT_TYPE_CONSIS, T>::type getrs(
      const char tran, const int_type n, const int_type nrhs,
      const value_type *a, const int_type lda, const int_type *ipiv, pointer b,
      const int_type ldb) {
    std::vector<psmilu_lapack_int> buf(ipiv, ipiv + n);
    return internal::getrs(tran, n, nrhs, a, lda, buf.data(), b, ldb);
  }

  inline static int_type getrs(const DenseMatrix<value_type> &a,
                               const Array<int_type> &        ipiv,
                               Array<value_type> &b, char tran = 'N') {
    psmilu_assert(a.is_squared(), "matrix must be squared");
    psmilu_assert(a.nrows() == ipiv.size(),
                  "row size should match permutation vector length");
    psmilu_assert(a.nrows() == b.size(), "inconsistent matrix and rhs size");
    return getrs(tran, a.nrows(), 1, a.data(), a.nrows(), ipiv.data(), b.data(),
                 b.size());
  }

  inline static int_type getrs(const DenseMatrix<value_type> &a,
                               const Array<int_type> &        ipiv,
                               DenseMatrix<value_type> &b, char tran = 'N') {
    psmilu_assert(a.is_squared(), "matrix must be squared");
    psmilu_assert(a.nrows() == ipiv.size(),
                  "row size should match permutation vector length");
    psmilu_assert(a.nrows() == b.nrows(), "inconsistent matrix and rhs size");
    return getrs(tran, a.nrows(), b.ncols(), a.data(), a.nrows(), ipiv.data(),
                 b.data(), b.nrows());
  }

  ///@}

  /// \name QRCP
  ///@{

  template <typename T = psmilu_lapack_int>
  inline static typename std::enable_if<_INT_TYPE_CONSIS, T>::type geqp3(
      const int_type m, const int_type n, pointer a, const int_type lda,
      int_type *jpvt, pointer tau, pointer work, const int_type lwork) {
    return internal::geqp3(m, n, a, lda, jpvt, tau, work, lwork);
  }

  template <typename T = psmilu_lapack_int>
  inline static typename std::enable_if<!_INT_TYPE_CONSIS, T>::type geqp3(
      const int_type m, const int_type n, pointer a, const int_type lda,
      int_type *jpvt, pointer tau, pointer work, const int_type lwork) {
    if (lwork == (int_type)-1)
      return internal::geqp3(m, n, a, lda, nullptr, tau, work, -1);
    // NOTE jpvt can be input as well
    std::vector<psmilu_lapack_int> buf(jpvt, jpvt + n);
    const auto                     info =
        internal::geqp3(m, n, a, lda, buf.data(), tau, work, lwork);
    std::copy(buf.cbegin(), buf.cend(), jpvt);
    return info;
  }

  inline static int_type geqp3(DenseMatrix<value_type> &a,
                               Array<int_type> &jpvt, Array<value_type> &tau) {
    // query opt size
    psmilu_assert(jpvt.size() == a.ncols(),
                  "column pivoting should have size of column");
    psmilu_assert(tau.size() == std::min(a.nrows(), a.ncols()),
                  "tau should have size of min(m,n)");
    value_type lwork;
    geqp3(a.nrows(), a.ncols(), a.data(), a.nrows(), jpvt.data(), tau.data(),
          &lwork, -1);
    std::vector<value_type> work((int_type)lwork);
    return geqp3(a.nrows(), a.ncols(), a.data(), a.nrows(), jpvt.data(),
                 tau.data(), work.data(), (int_type)lwork);
  }

  inline static int_type ormqr(const char side, const char trans,
                               const int_type m, const int_type n,
                               const int_type k, const value_type *a,
                               const int_type lda, const value_type *tau,
                               pointer c, const int_type ldc, pointer work,
                               const int_type lwork) {
    return internal::ormqr(side, trans, m, n, k, a, lda, tau, c, ldc, work,
                           lwork);
  }

  inline static int_type trcon(const char norm, const char uplo,
                               const char diag, const psmilu_lapack_int n,
                               const value_type *a, const psmilu_lapack_int lda,
                               value_type &rcond, pointer work,
                               psmilu_lapack_int *iwork) {
    return internal::trcon(norm, uplo, diag, n, a, lda, rcond, work, iwork);
  }

  inline static int_type trcon(const char norm, const char uplo,
                               const char                     diag,
                               const DenseMatrix<value_type> &a,
                               value_type &                   rcond) {
    psmilu_assert(
        a.nrows() >= a.ncols(),
        "a must have row size that is no smaller than its column size");
    std::vector<value_type>        work(3 * a.ncols());
    std::vector<psmilu_lapack_int> iwork(a.ncols());
    return trcon(norm, uplo, diag, a.ncols(), a.data(), a.nrows(), rcond,
                 work.data(), iwork.data());
  }

  inline static int_type trcon(const char norm, const char uplo,
                               const char                     diag,
                               const DenseMatrix<value_type> &a,
                               const int_type rank, value_type &rcond) {
    psmilu_assert((int_type)a.nrows() >= rank,
                  "a must have row size that is no smaller than its rank size");
    std::vector<value_type>        work(3 * a.ncols());
    std::vector<psmilu_lapack_int> iwork(a.ncols());
    return trcon(norm, uplo, diag, rank, a.data(), a.nrows(), rcond,
                 work.data(), iwork.data());
  }

  ///@}

  /// \name common
  ///@{

  inline static void trsv(const char uplo, const char trans, const char diag,
                          const psmilu_lapack_int n, const value_type *a,
                          const psmilu_lapack_int lda, pointer x,
                          const psmilu_lapack_int incx) {
    internal::trsv(uplo, trans, diag, n, a, lda, x, incx);
  }

  inline static void trsv(const char uplo, const char trans, const char diag,
                          const DenseMatrix<value_type> &a,
                          Array<value_type> &            x) {
    psmilu_assert(a.is_squared(), "input must be squared matrix");
    trsv(uplo, trans, diag, a.nrows(), a.data(), a.nrows(), x.data(), 1);
  }

  ///@}
};

}  // namespace psmilu

#endif  // _PSMILU_LAPACK_LAPACK_HPP
