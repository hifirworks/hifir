///////////////////////////////////////////////////////////////////////////////
//                  This file is part of HIF project                         //
///////////////////////////////////////////////////////////////////////////////

/*!
 * \file hif/small_scale/lapack.hpp
 * \brief HIF LAPACK interface
 * \author Qiao Chen

\verbatim
Copyright (C) 2019 NumGeom Group at Stony Brook University

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
\endverbatim

 */

#ifndef _HIF_SMALLSCALE_LAPACK_LAPACK_HPP
#define _HIF_SMALLSCALE_LAPACK_LAPACK_HPP

#include <algorithm>
#include <type_traits>
#include <vector>

#include "hif/ds/Array.hpp"
#include "hif/ds/DenseMatrix.hpp"
#include "hif/utils/log.hpp"
#include "hif/utils/math.hpp"

#include "hif/small_scale/lup_lapack.hpp"
#include "hif/small_scale/qrcp_lapack.hpp"
#include "hif/small_scale/syeig_lapack.hpp"
#include "hif/small_scale/trsv_lapack.hpp"

namespace hif {

/// \class Lapack
/// \tparam ValueType value type, e.g. \a double
/// \tparam IntType integer type, default is \ref hif_lapack_int
/// \note If the \a IntType and hif_lapack_int have different sizes, then
///       intermidiate buffers might be created.
/// \ingroup sss
template <class ValueType, class IntType = hif_lapack_int>
class Lapack {
 public:
  typedef IntType     int_type;    ///< integer type
  typedef ValueType   value_type;  ///< value type
  typedef value_type *pointer;     ///< pointer type
  typedef typename ValueTypeTrait<value_type>::value_type scalar_type;
  ///< scalar type
  constexpr static bool IS_REAL = std::is_same<scalar_type, value_type>::value;
  ///< flag indicating whether or not complex arithmetic is used

 private:
  constexpr static bool _INT_TYPE_CONSIS =
      sizeof(int_type) == sizeof(hif_lapack_int);
  static_assert(std::is_integral<int_type>::value, "must be integer type!");

 public:
  /// \name LU
  ///@{

  template <typename T = hif_lapack_int>
  inline static typename std::enable_if<_INT_TYPE_CONSIS, T>::type getrf(
      const int_type n, pointer a, const int_type lda, int_type *ipiv) {
    return internal::getrf(n, n, a, lda, (hif_lapack_int *)ipiv);
  }

  template <typename T = hif_lapack_int>
  inline static typename std::enable_if<!_INT_TYPE_CONSIS, T>::type getrf(
      const int_type n, pointer a, const int_type lda, int_type *ipiv) {
    std::vector<hif_lapack_int> buf(n);
    const hif_lapack_int info = internal::getrf(n, n, a, lda, buf.data());
    std::copy(buf.cbegin(), buf.cend(), ipiv);
    return info;
  }

  inline static int_type getrf(DenseMatrix<value_type> &a,
                               Array<int_type> &        ipiv) {
    hif_assert(a.nrows() == ipiv.size(),
               "row size should match permutation vector length");
    return getrf(a.nrows(), a.data(), a.nrows(), ipiv.data());
  }

  template <typename T = hif_lapack_int>
  inline static typename std::enable_if<_INT_TYPE_CONSIS, T>::type getrs(
      const char tran, const int_type n, const int_type nrhs,
      const value_type *a, const int_type lda, const int_type *ipiv, pointer b,
      const int_type ldb) {
    return internal::getrs(tran, n, nrhs, a, lda, ipiv, b, ldb);
  }

  template <typename T = hif_lapack_int>
  inline static typename std::enable_if<!_INT_TYPE_CONSIS, T>::type getrs(
      const char tran, const int_type n, const int_type nrhs,
      const value_type *a, const int_type lda, const int_type *ipiv, pointer b,
      const int_type ldb) {
    std::vector<hif_lapack_int> buf(ipiv, ipiv + n);
    return internal::getrs(tran, n, nrhs, a, lda, buf.data(), b, ldb);
  }

  inline static int_type getrs(const DenseMatrix<value_type> &a,
                               const Array<int_type> &        ipiv,
                               Array<value_type> &b, char tran = 'N') {
    hif_assert(a.is_squared(), "matrix must be squared");
    hif_assert(a.nrows() == ipiv.size(),
               "row size should match permutation vector length");
    hif_assert(a.nrows() == b.size(), "inconsistent matrix and rhs size");
    return getrs(tran, a.nrows(), 1, a.data(), a.nrows(), ipiv.data(), b.data(),
                 b.size());
  }

  inline static int_type getrs(const DenseMatrix<value_type> &a,
                               const Array<int_type> &        ipiv,
                               DenseMatrix<value_type> &b, char tran = 'N') {
    hif_assert(a.is_squared(), "matrix must be squared");
    hif_assert(a.nrows() == ipiv.size(),
               "row size should match permutation vector length");
    hif_assert(a.nrows() == b.nrows(), "inconsistent matrix and rhs size");
    return getrs(tran, a.nrows(), b.ncols(), a.data(), a.nrows(), ipiv.data(),
                 b.data(), b.nrows());
  }

  ///@}

  /// \name QRCP
  ///@{

  template <typename T = hif_lapack_int>
  inline static typename std::enable_if<_INT_TYPE_CONSIS, T>::type geqp3(
      const int_type m, const int_type n, pointer a, const int_type lda,
      int_type *jpvt, pointer tau, pointer work, const int_type lwork,
      scalar_type *rwork) {
    return internal::geqp3(m, n, a, lda, jpvt, tau, work, lwork, rwork);
  }

  template <typename T = hif_lapack_int>
  inline static typename std::enable_if<!_INT_TYPE_CONSIS, T>::type geqp3(
      const int_type m, const int_type n, pointer a, const int_type lda,
      int_type *jpvt, pointer tau, pointer work, const int_type lwork,
      scalar_type *rwork) {
    if (lwork == (int_type)-1)
      return internal::geqp3(m, n, a, lda, nullptr, tau, work, -1, rwork);
    // NOTE jpvt can be input as well
    std::vector<hif_lapack_int> buf(jpvt, jpvt + n);
    const auto                  info =
        internal::geqp3(m, n, a, lda, buf.data(), tau, work, lwork, rwork);
    std::copy(buf.cbegin(), buf.cend(), jpvt);
    return info;
  }

  inline static int_type geqp3(DenseMatrix<value_type> &a,
                               Array<int_type> &jpvt, Array<value_type> &tau) {
    // query opt size
    hif_assert(jpvt.size() == a.ncols(),
               "column pivoting should have size of column");
    hif_assert(tau.size() == std::min(a.nrows(), a.ncols()),
               "tau should have size of min(m,n)");
    value_type lwork;
    geqp3(a.nrows(), a.ncols(), a.data(), a.nrows(), jpvt.data(), tau.data(),
          &lwork, -1, nullptr);
    std::vector<value_type> work((int_type)abs(lwork));
    Array<scalar_type>      rwork;
    if (!IS_REAL) rwork.resize(2 * a.ncols());
    return geqp3(a.nrows(), a.ncols(), a.data(), a.nrows(), jpvt.data(),
                 tau.data(), work.data(), (int_type)abs(lwork), rwork.data());
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
                               const char diag, const hif_lapack_int n,
                               const value_type *a, const hif_lapack_int lda,
                               value_type &rcond, pointer work,
                               hif_lapack_int *iwork) {
    return internal::trcon(norm, uplo, diag, n, a, lda, rcond, work, iwork);
  }

  inline static int_type trcon(const char norm, const char uplo,
                               const char                     diag,
                               const DenseMatrix<value_type> &a,
                               value_type &                   rcond) {
    hif_assert(a.nrows() >= a.ncols(),
               "a must have row size that is no smaller than its column size");
    std::vector<value_type>     work(3 * a.ncols());
    std::vector<hif_lapack_int> iwork(a.ncols());
    return trcon(norm, uplo, diag, a.ncols(), a.data(), a.nrows(), rcond,
                 work.data(), iwork.data());
  }

  inline static int_type trcon(const char norm, const char uplo,
                               const char                     diag,
                               const DenseMatrix<value_type> &a,
                               const int_type rank, value_type &rcond) {
    hif_assert((int_type)a.nrows() >= rank,
               "a must have row size that is no smaller than its rank size");
    std::vector<value_type>     work(3 * a.ncols());
    std::vector<hif_lapack_int> iwork(a.ncols());
    return trcon(norm, uplo, diag, rank, a.data(), a.nrows(), rcond,
                 work.data(), iwork.data());
  }

  inline static int_type laic1(const int_type job, const int_type j,
                               const value_type *x, const scalar_type sest,
                               const value_type *w, const value_type gamma,
                               scalar_type &sestpr, value_type &s,
                               value_type &c) {
    return internal::laic1(job, j, x, sest, w, gamma, sestpr, s, c);
  }

  ///@}

  /// \name SYEIG
  ///@{

  template <class T = int_type>
  inline static
      typename std::enable_if<std::is_floating_point<value_type>::value,
                              T>::type
      syev(const char uplo, const hif_lapack_int n, value_type *a,
           const hif_lapack_int lda, value_type *w, value_type *work,
           const hif_lapack_int lwork) {
    return internal::syev(uplo, n, a, lda, w, work, lwork);
  }

  template <class T = int_type>
  inline static
      typename std::enable_if<!std::is_floating_point<value_type>::value,
                              T>::type
      syev(const char, const hif_lapack_int, value_type *, const hif_lapack_int,
           value_type *, value_type *, const hif_lapack_int) {
    hif_error("?syev only works for *real* symmetric systems!");
    return 1;
  }

  inline static int_type syev(const char uplo, DenseMatrix<value_type> &a,
                              Array<value_type> &w, value_type *work,
                              const hif_lapack_int lwork) {
    return syev(uplo, a.nrows(), a.data(), a.nrows(), w.data(), work, lwork);
  }

  ///@}

  /// \name common
  ///@{

  inline static void trsv(const char uplo, const char trans, const char diag,
                          const hif_lapack_int n, const value_type *a,
                          const hif_lapack_int lda, pointer x,
                          const hif_lapack_int incx) {
    internal::trsv(uplo, trans, diag, n, a, lda, x, incx);
  }

  inline static void trsv(const char uplo, const char trans, const char diag,
                          const DenseMatrix<value_type> &a,
                          Array<value_type> &            x) {
    hif_assert(a.is_squared(), "input must be squared matrix");
    trsv(uplo, trans, diag, a.nrows(), a.data(), a.nrows(), x.data(), 1);
  }

  inline static void gemv(const char trans, const hif_lapack_int m,
                          const hif_lapack_int n, const value_type alpha,
                          const value_type *a, const hif_lapack_int lda,
                          const double *x, const double beta, double *y) {
    internal::gemv(trans, m, n, alpha, a, lda, x, beta, y);
  }

  inline static void gemv(const char trans, const double alpha,
                          const DenseMatrix<value_type> &a,
                          const Array<value_type> &x, const value_type beta,
                          Array<value_type> &y) {
    if (trans == 'N') {
      hif_assert(a.ncols() == x.size(), "unmatched sizes");
      hif_assert(a.nrows() == y.size(), "unmatched sizes");
    } else {
      hif_assert(a.ncols() == y.size(), "unmatched sizes");
      hif_assert(a.nrows() == x.size(), "unmatched sizes");
    }
    gemv(trans, a.nrows(), a.ncols(), alpha, a.data(), a.nrows(), x.data(),
         beta, y.data());
  }

  ///@}
};

}  // namespace hif

#endif  // _HIF_SMALLSCALE_LAPACK_LAPACK_HPP
