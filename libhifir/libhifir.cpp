///////////////////////////////////////////////////////////////////////////////
//                  This file is part of the HIFIR library                   //
///////////////////////////////////////////////////////////////////////////////

/*!
 * \file libhifir.cpp
 * \brief C++ implementation of libhifir
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

#include <cstdint>
#include <cstdlib>
#include <string>

// always throw instead of calling abort
#ifndef HIF_THROW
#  define HIF_THROW
#endif

#include "hifir.hpp"

#include "libhifir.h"

#define LIBHIFIR_MIXED 2
#define LIBHIFIR_COMPLEX 4
#define LIBHIFIR_INT64 8
#define __TOKEN_TAG__ -92

namespace libhifir {

// handle internal C++ HIFIR errors
static std::string error_msgs[2];

static inline bool is_mixed(int flag) { return flag & LIBHIFIR_MIXED; }
static inline bool is_complex(int flag) { return flag & LIBHIFIR_COMPLEX; }
static inline bool is_int64(int flag) { return flag & LIBHIFIR_INT64; }

// encoder/decoder helper
union PrecCoder {
  void *         ptr;  // decode data pointer
  std::uintptr_t tag;  // encode integer tag
};

template <class PrecType>
static inline void get_factor_stats(const PrecType &M, long long stats[]) {
  if (M.empty())
    for (int i = 0; i < 11; ++i) stats[i] = 0;
  else {
    stats[0]  = M.nnz();
    stats[1]  = M.nnz_ef();
    stats[2]  = M.nnz_ldu();
    stats[3]  = M.stats(0);
    stats[4]  = M.stats(1);
    stats[5]  = M.stats(4);
    stats[6]  = M.stats(5);
    stats[7]  = M.levels();
    stats[8]  = M.rank();
    stats[9]  = M.schur_rank();
    stats[10] = M.schur_size();
  }
}

static inline hif::Params create_params(const double params[]) {
  hif::Params p   = hif::get_default_params();
  p.tau_L         = params[LIBHIFIR_DROPTOL_L];
  p.tau_U         = params[LIBHIFIR_DROPTOL_U];
  p.kappa_d       = params[LIBHIFIR_COND_D];
  p.kappa         = params[LIBHIFIR_COND];
  p.alpha_L       = params[LIBHIFIR_ALPHA_L];
  p.alpha_U       = params[LIBHIFIR_ALPHA_U];
  p.verbose       = params[LIBHIFIR_VERBOSE];
  p.reorder       = params[LIBHIFIR_REORDER];
  p.symm_pre_lvls = params[LIBHIFIR_SYMMPRELVLS];
  p.threads       = params[LIBHIFIR_THREADS];
  p.rrqr_cond     = params[LIBHIFIR_RRQR_COND];
  p.pivot         = params[LIBHIFIR_PIVOT];
  p.beta          = params[LIBHIFIR_BETA];
  p.is_symm       = params[LIBHIFIR_ISSYMM];
  p.no_pre        = params[LIBHIFIR_NOPRE];
  return p;
}

template <bool UseCrs, class ValueType, class PrecType>
static inline void factorize(PrecType &M, const std::size_t n,
                             const void *ind_start, const void *indices,
                             const void *vals, const double params[]) {
  using index_t  = typename PrecType::index_type;
  using crs_t    = hif::CRS<ValueType, index_t>;
  using ccs_t    = hif::CCS<ValueType, index_t>;
  using matrix_t = typename std::conditional<UseCrs, crs_t, ccs_t>::type;
  using value_t  = typename matrix_t::value_type;

  // create input matrix
  matrix_t A(n, n, (index_t *)ind_start, (index_t *)indices, (value_t *)vals,
             true);
  M.factorize(A, create_params(params));
}

template <bool DoSolve, class ValueType, class PrecType>
static inline void apply(const PrecType &M, const std::size_t n, const void *b,
                         const bool trans, const int rank, void *x) {
  using array_t = hif::Array<ValueType>;
  array_t bb(n, (ValueType *)b, true);
  array_t xx(n, (ValueType *)x, true);
  if (DoSolve)
    M.solve(bb, xx, trans, rank);
  else
    M.mmultiply(bb, xx, trans, rank);
}

template <class ValueType, class PrecType>
static inline void hifir(const PrecType &M, const std::size_t n,
                         const void *ind_ptr, const void *indices,
                         const void *vals, const bool is_crs, const void *b,
                         const int nirs, const double *betas, const bool trans,
                         const int rank, void *x, int *iters, int *ir_status) {
  using array_t = hif::Array<ValueType>;
  using index_t = typename PrecType::index_type;
  array_t bb(n, (ValueType *)b, true);
  array_t xx(n, (ValueType *)x, true);
  if (!betas) {
    if (is_crs) {
      using matrix_t = hif::CRS<ValueType, index_t>;
      matrix_t A(n, n, (index_t *)ind_ptr, (index_t *)indices,
                 (ValueType *)vals, true);
      M.hifir(A, bb, nirs, xx, trans, rank);
    } else {
      using matrix_t = hif::CCS<ValueType, index_t>;
      matrix_t A(n, n, (index_t *)ind_ptr, (index_t *)indices,
                 (ValueType *)vals, true);
      M.hifir(A, bb, nirs, xx, trans, rank);
    }
  } else {
    if (is_crs) {
      using matrix_t = hif::CRS<ValueType, index_t>;
      matrix_t   A(n, n, (index_t *)ind_ptr, (index_t *)indices,
                 (ValueType *)vals, true);
      const auto info = M.hifir(A, bb, nirs, betas, xx, trans, rank);
      *iters          = info.first;
      *ir_status      = info.second;
    } else {
      using matrix_t = hif::CCS<ValueType, index_t>;
      matrix_t   A(n, n, (index_t *)ind_ptr, (index_t *)indices,
                 (ValueType *)vals, true);
      const auto info = M.hifir(A, bb, nirs, betas, xx, trans, rank);
      *iters          = info.first;
      *ir_status      = info.second;
    }
  }
}

}  // namespace libhifir

extern "C" {

// get versions
void libhifir_version(int vs[]) {
  vs[0] = HIF_GLOBAL_VERSION;
  vs[1] = HIF_MAJOR_VERSION;
  vs[2] = HIF_MINOR_VERSION;
}

// enabling warning
void libhifir_enable_warning() { hif::warn_flag(1); }

// disable warning
void libhifir_disable_warning() { hif::warn_flag(0); }

// error message
const char *libhifir_error_msg() {
  libhifir::error_msgs[0] = libhifir::error_msgs[1];
  libhifir::error_msgs[1].clear();
  if (libhifir::error_msgs[0].empty()) return NULL;
  return libhifir::error_msgs[0].c_str();
}

// setup parameters
void libhifir_setup_params(double params[]) {
  for (int i = 0; i < LIBHIFIR_NUMBER_PARAMS; ++i) params[i] = -100.0;
  params[LIBHIFIR_MATRIX_TYPE] = LIBHIFIR_MATRIX_CRS;
  params[LIBHIFIR_DROPTOL_L]   = hif::DEFAULT_PARAMS.tau_L;
  params[LIBHIFIR_DROPTOL_U]   = hif::DEFAULT_PARAMS.tau_U;
  params[LIBHIFIR_COND_D]      = hif::DEFAULT_PARAMS.kappa_d;
  params[LIBHIFIR_COND]        = hif::DEFAULT_PARAMS.kappa;
  params[LIBHIFIR_ALPHA_L]     = hif::DEFAULT_PARAMS.alpha_L;
  params[LIBHIFIR_ALPHA_U]     = hif::DEFAULT_PARAMS.alpha_U;
  params[LIBHIFIR_VERBOSE]     = hif::DEFAULT_PARAMS.verbose;
  params[LIBHIFIR_REORDER]     = hif::DEFAULT_PARAMS.reorder;
  params[LIBHIFIR_SYMMPRELVLS] = hif::DEFAULT_PARAMS.symm_pre_lvls;
  params[LIBHIFIR_THREADS]     = hif::DEFAULT_PARAMS.threads;
  params[LIBHIFIR_RRQR_COND]   = hif::DEFAULT_PARAMS.rrqr_cond;
  params[LIBHIFIR_PIVOT]       = hif::DEFAULT_PARAMS.pivot;
  params[LIBHIFIR_BETA]        = hif::DEFAULT_PARAMS.beta;
  params[LIBHIFIR_ISSYMM]      = hif::DEFAULT_PARAMS.is_symm;
  params[LIBHIFIR_NOPRE]       = hif::DEFAULT_PARAMS.no_pre;
}

// create
int libhifir_create(const int *is_mixed, const int *is_complex,
                    const int *is_int64, char M[]) {
  if ((int)M[0] == __TOKEN_TAG__) return LIBHIFIR_PREC_EXIST;
  int prec_type(1);
  if (is_mixed && *is_mixed) prec_type |= LIBHIFIR_MIXED;
  if (is_complex && *is_complex) prec_type |= LIBHIFIR_COMPLEX;
  if (is_int64 && *is_int64) prec_type |= LIBHIFIR_INT64;

  // create M
  void *prec = nullptr;
  if (!libhifir::is_mixed(prec_type)) {
    using value_t = double;
    if (!libhifir::is_complex(prec_type)) {
      if (!libhifir::is_int64(prec_type))
        prec = (void *)new (std::nothrow) hif::HIF<value_t, int>();
      else
        prec = (void *)new (std::nothrow) hif::HIF<value_t, std::int64_t>();
    } else {
      if (!libhifir::is_int64(prec_type))
        prec =
            (void *)new (std::nothrow) hif::HIF<std::complex<value_t>, int>();
      else
        prec = (void *)new (std::nothrow)
            hif::HIF<std::complex<value_t>, std::int64_t>();
    }
  } else {
    using value_t = float;
    if (!libhifir::is_complex(prec_type)) {
      if (!libhifir::is_int64(prec_type))
        prec = (void *)new (std::nothrow) hif::HIF<value_t, int>();
      else
        prec = (void *)new (std::nothrow) hif::HIF<value_t, std::int64_t>();
    } else {
      if (!libhifir::is_int64(prec_type))
        prec =
            (void *)new (std::nothrow) hif::HIF<std::complex<value_t>, int>();
      else
        prec = (void *)new (std::nothrow)
            hif::HIF<std::complex<value_t>, std::int64_t>();
    }
  }
  if (!prec) return LIBHIFIR_BAD_PREC;
  M[0] = __TOKEN_TAG__;  // mark as created by this function
  M[1] = prec_type;      // mark type
  libhifir::PrecCoder encoder;
  encoder.ptr              = prec;
  *(std::uintptr_t *)&M[2] = encoder.tag;
  return LIBHIFIR_SUCCESS;
}

// check
int libhifir_check(const char M[], int *is_mixed, int *is_complex,
                   int *is_int64) {
  if ((int)M[0] != __TOKEN_TAG__) return LIBHIFIR_BAD_PREC;
  hif_assert(is_mixed, "is_mixed cannot be NULL");
  *is_mixed = libhifir::is_mixed((int)M[1]);
  hif_assert(is_complex, "is_complex cannot be NULL");
  *is_complex = libhifir::is_complex((int)M[1]);
  hif_assert(is_int64, "is_int64 cannot be NULL");
  *is_int64 = libhifir::is_int64((int)M[1]);
  return LIBHIFIR_SUCCESS;
}

// destroy
int libhifir_destroy(char M[]) {
  if ((int)M[0] != __TOKEN_TAG__) return LIBHIFIR_BAD_PREC;
  void *               prec      = nullptr;
  const int            prec_type = M[1];
  const std::uintptr_t prec_tag  = *(std::uintptr_t *)&M[2];
  libhifir::PrecCoder  encoder;
  encoder.tag = prec_tag;
  prec        = encoder.ptr;
  // delete prec; // cannot do this because it won't call destructor
  if (!libhifir::is_mixed(prec_type)) {
    using value_t = double;
    if (!libhifir::is_complex(prec_type)) {
      if (!libhifir::is_int64(prec_type))
        delete (hif::HIF<value_t, int> *)prec;
      else
        delete (hif::HIF<value_t, std::int64_t> *)prec;
    } else {
      if (!libhifir::is_int64(prec_type))
        delete (hif::HIF<std::complex<value_t>, int> *)prec;
      else
        delete (hif::HIF<std::complex<value_t>, std::int64_t> *)prec;
    }
  } else {
    using value_t = float;
    if (!libhifir::is_complex(prec_type)) {
      if (!libhifir::is_int64(prec_type))
        delete (hif::HIF<value_t, int> *)prec;
      else
        delete (hif::HIF<value_t, std::int64_t> *)prec;
    } else {
      if (!libhifir::is_int64(prec_type))
        delete (hif::HIF<std::complex<value_t>, int> *)prec;
      else
        delete (hif::HIF<std::complex<value_t>, std::int64_t> *)prec;
    }
  }
  M[0] = M[1] = 0;  // reset token
  return LIBHIFIR_SUCCESS;
}

// check empty
int libhifir_empty(const char M[]) {
  if ((int)M[0] != __TOKEN_TAG__) return 1;
  void *               prec      = nullptr;
  const int            prec_type = M[1];
  const std::uintptr_t prec_tag  = *(std::uintptr_t *)&M[2];
  libhifir::PrecCoder  encoder;
  encoder.tag = prec_tag;
  prec        = encoder.ptr;
  if (!libhifir::is_mixed(prec_type)) {
    using value_t = double;
    if (!libhifir::is_complex(prec_type)) {
      if (!libhifir::is_int64(prec_type))
        return ((hif::HIF<value_t, int> *)prec)->empty();
      return ((hif::HIF<value_t, std::int64_t> *)prec)->empty();
    } else {
      if (!libhifir::is_int64(prec_type))
        return ((hif::HIF<std::complex<value_t>, int> *)prec)->empty();
      return ((hif::HIF<std::complex<value_t>, std::int64_t> *)prec)->empty();
    }
  } else {
    using value_t = float;
    if (!libhifir::is_complex(prec_type)) {
      if (!libhifir::is_int64(prec_type))
        return ((hif::HIF<value_t, int> *)prec)->empty();
      return ((hif::HIF<value_t, std::int64_t> *)prec)->empty();
    } else {
      if (!libhifir::is_int64(prec_type))
        return ((hif::HIF<std::complex<value_t>, int> *)prec)->empty();
      return ((hif::HIF<std::complex<value_t>, std::int64_t> *)prec)->empty();
    }
  }
}

// query stats
void libhifir_query_stats(const char M[], long long stats[]) {
  if ((int)M[0] != __TOKEN_TAG__) {
    for (int i = 0; i < 11; ++i) stats[i] = 0;
    return;
  }
  void *               prec      = nullptr;
  const int            prec_type = M[1];
  const std::uintptr_t prec_tag  = *(std::uintptr_t *)&M[2];
  libhifir::PrecCoder  encoder;
  encoder.tag = prec_tag;
  prec        = encoder.ptr;
  if (!libhifir::is_mixed(prec_type)) {
    using value_t = double;
    if (!libhifir::is_complex(prec_type)) {
      if (!libhifir::is_int64(prec_type))
        libhifir::get_factor_stats(*(hif::HIF<value_t, int> *)prec, stats);
      else
        libhifir::get_factor_stats(*(hif::HIF<value_t, std::int64_t> *)prec,
                                   stats);
    } else {
      if (!libhifir::is_int64(prec_type))
        libhifir::get_factor_stats(
            *(hif::HIF<std::complex<value_t>, int> *)prec, stats);
      else
        libhifir::get_factor_stats(
            *(hif::HIF<std::complex<value_t>, std::int64_t> *)prec, stats);
    }
  } else {
    using value_t = float;
    if (!libhifir::is_complex(prec_type)) {
      if (!libhifir::is_int64(prec_type))
        libhifir::get_factor_stats(*(hif::HIF<value_t, int> *)prec, stats);
      else
        libhifir::get_factor_stats(*(hif::HIF<value_t, std::int64_t> *)prec,
                                   stats);
    } else {
      if (!libhifir::is_int64(prec_type))
        libhifir::get_factor_stats(
            *(hif::HIF<std::complex<value_t>, int> *)prec, stats);
      else
        libhifir::get_factor_stats(
            *(hif::HIF<std::complex<value_t>, std::int64_t> *)prec, stats);
    }
  }
}

// factorization
int libhifir_factorize(const char M[], const long long *n,
                       const void *ind_start, const void *indices,
                       const void *vals, const double params[]) {
  if ((int)M[0] != __TOKEN_TAG__) return LIBHIFIR_BAD_PREC;
  const bool           use_crs   = (int)params[0] == LIBHIFIR_MATRIX_CRS;
  void *               prec      = nullptr;
  const int            prec_type = M[1];
  const std::uintptr_t prec_tag  = *(std::uintptr_t *)&M[2];
  libhifir::PrecCoder  encoder;
  encoder.tag = prec_tag;
  prec        = encoder.ptr;
  try {
    if (!libhifir::is_mixed(prec_type)) {
      using value_t = double;
      if (!libhifir::is_complex(prec_type)) {
        if (!libhifir::is_int64(prec_type))
          use_crs ? libhifir::factorize<true, value_t>(
                        *(hif::HIF<value_t, int> *)prec, *n, ind_start, indices,
                        vals, params)
                  : libhifir::factorize<false, value_t>(
                        *(hif::HIF<value_t, int> *)prec, *n, ind_start, indices,
                        vals, params);
        else
          use_crs ? libhifir::factorize<true, value_t>(
                        *(hif::HIF<value_t, std::int64_t> *)prec, *n, ind_start,
                        indices, vals, params)
                  : libhifir::factorize<false, value_t>(
                        *(hif::HIF<value_t, std::int64_t> *)prec, *n, ind_start,
                        indices, vals, params);
      } else {
        if (!libhifir::is_int64(prec_type))
          use_crs ? libhifir::factorize<true, std::complex<value_t>>(
                        *(hif::HIF<std::complex<value_t>, int> *)prec, *n,
                        ind_start, indices, vals, params)
                  : libhifir::factorize<false, std::complex<value_t>>(
                        *(hif::HIF<std::complex<value_t>, int> *)prec, *n,
                        ind_start, indices, vals, params);
        else
          use_crs ? libhifir::factorize<true, std::complex<value_t>>(
                        *(hif::HIF<std::complex<value_t>, std::int64_t> *)prec,
                        *n, ind_start, indices, vals, params)
                  : libhifir::factorize<false, std::complex<value_t>>(
                        *(hif::HIF<std::complex<value_t>, std::int64_t> *)prec,
                        *n, ind_start, indices, vals, params);
      }
    } else {
      // NOTE: The input is assumed to be double-precision
      using value_t = float;
      if (!libhifir::is_complex(prec_type)) {
        if (!libhifir::is_int64(prec_type))
          use_crs ? libhifir::factorize<true, double>(
                        *(hif::HIF<value_t, int> *)prec, *n, ind_start, indices,
                        vals, params)
                  : libhifir::factorize<false, double>(
                        *(hif::HIF<value_t, int> *)prec, *n, ind_start, indices,
                        vals, params);
        else
          use_crs ? libhifir::factorize<true, double>(
                        *(hif::HIF<value_t, std::int64_t> *)prec, *n, ind_start,
                        indices, vals, params)
                  : libhifir::factorize<false, double>(
                        *(hif::HIF<value_t, std::int64_t> *)prec, *n, ind_start,
                        indices, vals, params);
      } else {
        if (!libhifir::is_int64(prec_type))
          use_crs ? libhifir::factorize<true, std::complex<double>>(
                        *(hif::HIF<std::complex<value_t>, int> *)prec, *n,
                        ind_start, indices, vals, params)
                  : libhifir::factorize<false, std::complex<double>>(
                        *(hif::HIF<std::complex<value_t>, int> *)prec, *n,
                        ind_start, indices, vals, params);
        else
          use_crs ? libhifir::factorize<true, std::complex<double>>(
                        *(hif::HIF<std::complex<value_t>, std::int64_t> *)prec,
                        *n, ind_start, indices, vals, params)
                  : libhifir::factorize<false, std::complex<double>>(
                        *(hif::HIF<std::complex<value_t>, std::int64_t> *)prec,
                        *n, ind_start, indices, vals, params);
      }
    }
  } catch (const std::exception &e) {
    // internal error
    libhifir::error_msgs[1] = e.what();
    return LIBHIFIR_HIFIR_ERROR;
  }
  return LIBHIFIR_SUCCESS;
}

// solve
int libhifir_solve(const char M[], const long long *n, const void *b,
                   const int *trans, const int *rank, void *x) {
  if ((int)M[0] != __TOKEN_TAG__) return LIBHIFIR_BAD_PREC;
  const bool           tran      = trans ? *trans : false;
  const int            rnk       = rank ? *rank : 0;
  void *               prec      = nullptr;
  const int            prec_type = M[1];
  const std::uintptr_t prec_tag  = *(std::uintptr_t *)&M[2];
  libhifir::PrecCoder  encoder;
  encoder.tag = prec_tag;
  prec        = encoder.ptr;
  try {
    if (!libhifir::is_mixed(prec_type)) {
      using value_t = double;
      if (!libhifir::is_complex(prec_type)) {
        if (!libhifir::is_int64(prec_type))
          libhifir::apply<true, value_t>(*(hif::HIF<value_t, int> *)prec, *n, b,
                                         tran, rnk, x);
        else
          libhifir::apply<true, value_t>(
              *(hif::HIF<value_t, std::int64_t> *)prec, *n, b, tran, rnk, x);
      } else {
        if (!libhifir::is_int64(prec_type))
          libhifir::apply<true, std::complex<value_t>>(
              *(hif::HIF<std::complex<value_t>, int> *)prec, *n, b, tran, rnk,
              x);
        else
          libhifir::apply<true, std::complex<value_t>>(
              *(hif::HIF<std::complex<value_t>, std::int64_t> *)prec, *n, b,
              tran, rnk, x);
      }
    } else {
      // NOTE: The input is assumed to be double-precision
      using value_t = float;
      if (!libhifir::is_complex(prec_type)) {
        if (!libhifir::is_int64(prec_type))
          libhifir::apply<true, double>(*(hif::HIF<value_t, int> *)prec, *n, b,
                                        tran, rnk, x);
        else
          libhifir::apply<true, double>(
              *(hif::HIF<value_t, std::int64_t> *)prec, *n, b, tran, rnk, x);
      } else {
        if (!libhifir::is_int64(prec_type))
          libhifir::apply<true, std::complex<double>>(
              *(hif::HIF<std::complex<value_t>, int> *)prec, *n, b, tran, rnk,
              x);
        else
          libhifir::apply<true, std::complex<double>>(
              *(hif::HIF<std::complex<value_t>, std::int64_t> *)prec, *n, b,
              tran, rnk, x);
      }
    }
  } catch (const std::exception &e) {
    // internal error
    libhifir::error_msgs[1] = e.what();
    return LIBHIFIR_HIFIR_ERROR;
  }
  return LIBHIFIR_SUCCESS;
}

// multiply
int libhifir_mmultiply(const char M[], const long long *n, const void *b,
                       const int *trans, const int *rank, void *x) {
  if ((int)M[0] != __TOKEN_TAG__) return LIBHIFIR_BAD_PREC;
  const bool           tran      = trans ? *trans : false;
  const int            rnk       = rank ? *rank : 0;
  void *               prec      = nullptr;
  const int            prec_type = M[1];
  const std::uintptr_t prec_tag  = *(std::uintptr_t *)&M[2];
  libhifir::PrecCoder  encoder;
  encoder.tag = prec_tag;
  prec        = encoder.ptr;
  try {
    if (!libhifir::is_mixed(prec_type)) {
      using value_t = double;
      if (!libhifir::is_complex(prec_type)) {
        if (!libhifir::is_int64(prec_type))
          libhifir::apply<false, value_t>(*(hif::HIF<value_t, int> *)prec, *n,
                                          b, tran, rnk, x);
        else
          libhifir::apply<false, value_t>(
              *(hif::HIF<value_t, std::int64_t> *)prec, *n, b, tran, rnk, x);
      } else {
        if (!libhifir::is_int64(prec_type))
          libhifir::apply<false, std::complex<value_t>>(
              *(hif::HIF<std::complex<value_t>, int> *)prec, *n, b, tran, rnk,
              x);
        else
          libhifir::apply<false, std::complex<value_t>>(
              *(hif::HIF<std::complex<value_t>, std::int64_t> *)prec, *n, b,
              tran, rnk, x);
      }
    } else {
      // NOTE: The input is assumed to be double-precision
      using value_t = float;
      if (!libhifir::is_complex(prec_type)) {
        if (!libhifir::is_int64(prec_type))
          libhifir::apply<false, double>(*(hif::HIF<value_t, int> *)prec, *n, b,
                                         tran, rnk, x);
        else
          libhifir::apply<false, double>(
              *(hif::HIF<value_t, std::int64_t> *)prec, *n, b, tran, rnk, x);
      } else {
        if (!libhifir::is_int64(prec_type))
          libhifir::apply<false, std::complex<double>>(
              *(hif::HIF<std::complex<value_t>, int> *)prec, *n, b, tran, rnk,
              x);
        else
          libhifir::apply<false, std::complex<double>>(
              *(hif::HIF<std::complex<value_t>, std::int64_t> *)prec, *n, b,
              tran, rnk, x);
      }
    }
  } catch (const std::exception &e) {
    // internal error
    libhifir::error_msgs[1] = e.what();
    return LIBHIFIR_HIFIR_ERROR;
  }
  return LIBHIFIR_SUCCESS;
}

/*
void hifir(const PrecType &M, const std::size_t n,
                         const void *ind_ptr, const void *indices,
                         const void *vals, const bool is_crs, const void *b,
                         const int nirs, const double *betas, const bool trans,
                         const int rank, void *x, int *iters, int *ir_status)
*/

// hifir operation
int libhifir_hifir(const char M[], const long long *n, const void *ind_start,
                   const void *indices, const void *vals, const int *is_crs,
                   const void *b, const int *nirs, const double *betas,
                   const int *trans, const int *rank, void *x, int *iters,
                   int *ir_status) {
  if (!nirs || *nirs <= 1) return libhifir_solve(M, n, b, trans, rank, x);
  const bool           tran      = trans ? *trans : false;
  const int            rnk       = rank ? *rank : 0;
  const bool           iscrs     = is_crs ? *is_crs : true;
  void *               prec      = nullptr;
  const int            prec_type = M[1];
  const std::uintptr_t prec_tag  = *(std::uintptr_t *)&M[2];
  libhifir::PrecCoder  encoder;
  encoder.tag = prec_tag;
  prec        = encoder.ptr;
  try {
    if (!libhifir::is_mixed(prec_type)) {
      using value_t = double;
      if (!libhifir::is_complex(prec_type)) {
        if (!libhifir::is_int64(prec_type))
          libhifir::hifir<value_t>(*(hif::HIF<value_t, int> *)prec, *n,
                                   ind_start, indices, vals, iscrs, b, *nirs,
                                   betas, tran, rnk, x, iters, ir_status);
        else
          libhifir::hifir<value_t>(*(hif::HIF<value_t, std::int64_t> *)prec, *n,
                                   ind_start, indices, vals, iscrs, b, *nirs,
                                   betas, tran, rnk, x, iters, ir_status);
      } else {
        if (!libhifir::is_int64(prec_type))
          libhifir::hifir<std::complex<value_t>>(
              *(hif::HIF<std::complex<value_t>, int> *)prec, *n, ind_start,
              indices, vals, iscrs, b, *nirs, betas, tran, rnk, x, iters,
              ir_status);
        else
          libhifir::hifir<std::complex<value_t>>(
              *(hif::HIF<std::complex<value_t>, std::int64_t> *)prec, *n,
              ind_start, indices, vals, iscrs, b, *nirs, betas, tran, rnk, x,
              iters, ir_status);
      }
    } else {
      // NOTE: The input is assumed to be double-precision
      using value_t = float;
      if (!libhifir::is_complex(prec_type)) {
        if (!libhifir::is_int64(prec_type))
          libhifir::hifir<double>(*(hif::HIF<value_t, int> *)prec, *n,
                                  ind_start, indices, vals, iscrs, b, *nirs,
                                  betas, tran, rnk, x, iters, ir_status);
        else
          libhifir::hifir<double>(*(hif::HIF<value_t, std::int64_t> *)prec, *n,
                                  ind_start, indices, vals, iscrs, b, *nirs,
                                  betas, tran, rnk, x, iters, ir_status);
      } else {
        if (!libhifir::is_int64(prec_type))
          libhifir::hifir<std::complex<double>>(
              *(hif::HIF<std::complex<value_t>, int> *)prec, *n, ind_start,
              indices, vals, iscrs, b, *nirs, betas, tran, rnk, x, iters,
              ir_status);
        else
          libhifir::hifir<std::complex<double>>(
              *(hif::HIF<std::complex<value_t>, std::int64_t> *)prec, *n,
              ind_start, indices, vals, iscrs, b, *nirs, betas, tran, rnk, x,
              iters, ir_status);
      }
    }
  } catch (const std::exception &e) {
    // internal error
    libhifir::error_msgs[1] = e.what();
    return LIBHIFIR_HIFIR_ERROR;
  }
  return LIBHIFIR_SUCCESS;
}

#ifndef DOXYGEN_SHOULD_SKIP_THIS

void libhifir_version_(int vs[]) { libhifir_version(vs); }
void libhifir_version__(int vs[]) { libhifir_version(vs); }
void LIBHIFIR_VERSION(int vs[]) { libhifir_version(vs); }
void LIBHIFIR_VERSION_(int vs[]) { libhifir_version(vs); }
void LIBHIFIR_VERSION__(int vs[]) { libhifir_version(vs); }

void libhifir_enable_warning_() { libhifir_enable_warning(); }
void libhifir_enable_warning__(void) { libhifir_enable_warning(); }
void LIBHIFIR_ENABLE_WARNING(void) { libhifir_enable_warning(); }
void LIBHIFIR_ENABLE_WARNING_(void) { libhifir_enable_warning(); }
void LIBHIFIR_ENABLE_WARNING__(void) { libhifir_enable_warning(); }
void libhifir_disable_warning_(void) { libhifir_disable_warning(); }
void libhifir_disable_warning__(void) { libhifir_disable_warning(); }
void LIBHIFIR_DISABLE_WARNING(void) { libhifir_disable_warning(); }
void LIBHIFIR_DISABLE_WARNING_(void) { libhifir_disable_warning(); }
void LIBHIFIR_DISABLE_WARNING__(void) { libhifir_disable_warning(); }

void libhifir_setup_params_(double params[]) { libhifir_setup_params(params); }
void libhifir_setup_params__(double params[]) { libhifir_setup_params(params); }
void LIBHIFIR_SETUP_PARAMS(double params[]) { libhifir_setup_params(params); }
void LIBHIFIR_SETUP_PARAMS_(double params[]) { libhifir_setup_params(params); }
void LIBHIFIR_SETUP_PARAMS__(double params[]) { libhifir_setup_params(params); }

int libhifir_create_(const int *is_mixed, const int *is_complex,
                     const int *is_int64, char M[]) {
  return libhifir_create(is_mixed, is_complex, is_int64, M);
}
int libhifir_create__(const int *is_mixed, const int *is_complex,
                      const int *is_int64, char M[]) {
  return libhifir_create(is_mixed, is_complex, is_int64, M);
}
int LIBHIFIR_CREATE(const int *is_mixed, const int *is_complex,
                    const int *is_int64, char M[]) {
  return libhifir_create(is_mixed, is_complex, is_int64, M);
}
int LIBHIFIR_CREATE_(const int *is_mixed, const int *is_complex,
                     const int *is_int64, char M[]) {
  return libhifir_create(is_mixed, is_complex, is_int64, M);
}
int LIBHIFIR_CREATE__(const int *is_mixed, const int *is_complex,
                      const int *is_int64, char M[]) {
  return libhifir_create(is_mixed, is_complex, is_int64, M);
}

int libhifir_check_(const char M[], int *is_mixed, int *is_complex,
                    int *is_int64) {
  return libhifir_check(M, is_mixed, is_complex, is_int64);
}
int libhifir_check__(const char M[], int *is_mixed, int *is_complex,
                     int *is_int64) {
  return libhifir_check(M, is_mixed, is_complex, is_int64);
}
int LIBHIFIR_CHECK(const char M[], int *is_mixed, int *is_complex,
                   int *is_int64) {
  return libhifir_check(M, is_mixed, is_complex, is_int64);
}
int LIBHIFIR_CHECK_(const char M[], int *is_mixed, int *is_complex,
                    int *is_int64) {
  return libhifir_check(M, is_mixed, is_complex, is_int64);
}
int LIBHIFIR_CHECK__(const char M[], int *is_mixed, int *is_complex,
                     int *is_int64) {
  return libhifir_check(M, is_mixed, is_complex, is_int64);
}

int libhifir_destroy_(char M[]) { return libhifir_destroy(M); }
int libhifir_destroy__(char M[]) { return libhifir_destroy(M); }
int LIBHIFIR_DESTROY(char M[]) { return libhifir_destroy(M); }
int LIBHIFIR_DESTROY_(char M[]) { return libhifir_destroy(M); }
int LIBHIFIR_DESTROY__(char M[]) { return libhifir_destroy(M); }

int libhifir_empty_(const char M[]) { return libhifir_empty(M); }
int libhifir_empty__(const char M[]) { return libhifir_empty(M); }
int LIBHIFIR_EMPTY(const char M[]) { return libhifir_empty(M); }
int LIBHIFIR_EMPTY_(const char M[]) { return libhifir_empty(M); }
int LIBHIFIR_EMPTY__(const char M[]) { return libhifir_empty(M); }

void libhifir_query_stats_(const char M[], long long stats[]) {
  libhifir_query_stats(M, stats);
}
void libhifir_query_stats__(const char M[], long long stats[]) {
  libhifir_query_stats(M, stats);
}
void LIBHIFIR_QUERY_STATS(const char M[], long long stats[]) {
  libhifir_query_stats(M, stats);
}
void LIBHIFIR_QUERY_STATS_(const char M[], long long stats[]) {
  libhifir_query_stats(M, stats);
}
void LIBHIFIR_QUERY_STATS__(const char M[], long long stats[]) {
  libhifir_query_stats(M, stats);
}

/* factorize */
int libhifir_factorize_(const char M[], const long long *n,
                        const void *ind_start, const void *indices,
                        const void *vals, const double params[]) {
  return libhifir_factorize(M, n, ind_start, indices, vals, params);
}
int libhifir_factorize__(const char M[], const long long *n,
                         const void *ind_start, const void *indices,
                         const void *vals, const double params[]) {
  return libhifir_factorize(M, n, ind_start, indices, vals, params);
}
int LIBHIFIR_FACTORIZE(const char M[], const long long *n,
                       const void *ind_start, const void *indices,
                       const void *vals, const double params[]) {
  return libhifir_factorize(M, n, ind_start, indices, vals, params);
}
int LIBHIFIR_FACTORIZE_(const char M[], const long long *n,
                        const void *ind_start, const void *indices,
                        const void *vals, const double params[]) {
  return libhifir_factorize(M, n, ind_start, indices, vals, params);
}
int LIBHIFIR_FACTORIZE__(const char M[], const long long *n,
                         const void *ind_start, const void *indices,
                         const void *vals, const double params[]) {
  return libhifir_factorize(M, n, ind_start, indices, vals, params);
}

int libhifir_solve_(const char M[], const long long *n, const void *b,
                    const int *trans, const int *rank, void *x) {
  return libhifir_solve(M, n, b, trans, rank, x);
}
int libhifir_solve__(const char M[], const long long *n, const void *b,
                     const int *trans, const int *rank, void *x) {
  return libhifir_solve(M, n, b, trans, rank, x);
}
int LIBHIFIR_SOLVE(const char M[], const long long *n, const void *b,
                   const int *trans, const int *rank, void *x) {
  return libhifir_solve(M, n, b, trans, rank, x);
}
int LIBHIFIR_SOLVE_(const char M[], const long long *n, const void *b,
                    const int *trans, const int *rank, void *x) {
  return libhifir_solve(M, n, b, trans, rank, x);
}
int LIBHIFIR_SOLVE__(const char M[], const long long *n, const void *b,
                     const int *trans, const int *rank, void *x) {
  return libhifir_solve(M, n, b, trans, rank, x);
}

int libhifir_mmultiply_(const char M[], const long long *n, const void *b,
                        const int *trans, const int *rank, void *x) {
  return libhifir_mmultiply(M, n, b, trans, rank, x);
}
int libhifir_mmultiply__(const char M[], const long long *n, const void *b,
                         const int *trans, const int *rank, void *x) {
  return libhifir_mmultiply(M, n, b, trans, rank, x);
}
int LIBHIFIR_MMULTIPLY(const char M[], const long long *n, const void *b,
                       const int *trans, const int *rank, void *x) {
  return libhifir_mmultiply(M, n, b, trans, rank, x);
}
int LIBHIFIR_MMULTIPLY_(const char M[], const long long *n, const void *b,
                        const int *trans, const int *rank, void *x) {
  return libhifir_mmultiply(M, n, b, trans, rank, x);
}
int LIBHIFIR_MMULTIPLY__(const char M[], const long long *n, const void *b,
                         const int *trans, const int *rank, void *x) {
  return libhifir_mmultiply(M, n, b, trans, rank, x);
}

int libhifir_hifir_(const char M[], const long long *n, const void *ind_start,
                    const void *indices, const void *vals, const int *is_crs,
                    const void *b, const int *nirs, const double *betas,
                    const int *trans, const int *rank, void *x, int *iters,
                    int *ir_status) {
  return libhifir_hifir(M, n, ind_start, indices, vals, is_crs, b, nirs, betas,
                        trans, rank, x, iters, ir_status);
}
int libhifir_hifir__(const char M[], const long long *n, const void *ind_start,
                     const void *indices, const void *vals, const int *is_crs,
                     const void *b, const int *nirs, const double *betas,
                     const int *trans, const int *rank, void *x, int *iters,
                     int *ir_status) {
  return libhifir_hifir(M, n, ind_start, indices, vals, is_crs, b, nirs, betas,
                        trans, rank, x, iters, ir_status);
}
int LIBHIFIR_HIFIR(const char M[], const long long *n, const void *ind_start,
                   const void *indices, const void *vals, const int *is_crs,
                   const void *b, const int *nirs, const double *betas,
                   const int *trans, const int *rank, void *x, int *iters,
                   int *ir_status) {
  return libhifir_hifir(M, n, ind_start, indices, vals, is_crs, b, nirs, betas,
                        trans, rank, x, iters, ir_status);
}
int LIBHIFIR_HIFIR_(const char M[], const long long *n, const void *ind_start,
                    const void *indices, const void *vals, const int *is_crs,
                    const void *b, const int *nirs, const double *betas,
                    const int *trans, const int *rank, void *x, int *iters,
                    int *ir_status) {
  return libhifir_hifir(M, n, ind_start, indices, vals, is_crs, b, nirs, betas,
                        trans, rank, x, iters, ir_status);
}
int LIBHIFIR_HIFIR__(const char M[], const long long *n, const void *ind_start,
                     const void *indices, const void *vals, const int *is_crs,
                     const void *b, const int *nirs, const double *betas,
                     const int *trans, const int *rank, void *x, int *iters,
                     int *ir_status) {
  return libhifir_hifir(M, n, ind_start, indices, vals, is_crs, b, nirs, betas,
                        trans, rank, x, iters, ir_status);
}

#endif  // DOXYGEN_SHOULD_SKIP_THIS
}