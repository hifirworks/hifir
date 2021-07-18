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

#include <hifir.hpp>
#include "libhifir.h"

#define _LHF_CANNOT_ACCEPT_NULL(__mat) \
  if (!__mat) return LHF_NULL_OBJ

#define _LHF_SET_HIFIR_ERROR(__e) libhifir_impl::error_msgs[1] = __e.what()

#define _LHF_RETURN_FROM_HIFIR_ERROR(__e) \
  _LHF_SET_HIFIR_ERROR(__e);              \
  return LHF_HIFIR_ERROR

namespace libhifir_impl {
// handle internal C++ HIFIR errors
static std::string error_msgs[2];

// get factorization result statistics
template <class PrecType>
static inline void get_factor_stats(const PrecType &M, std::size_t stats[]) {
  if (M.empty())
    for (int i = 0; i < 9; ++i) stats[i] = 0;
  else {
    stats[0]  = M.nnz();
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

// create parameters from C parameter array
static inline hif::Params create_params(const double params[]) {
  hif::Params p   = hif::get_default_params();
  p.tau_L         = params[LHF_DROPTOL_L];
  p.tau_U         = params[LHF_DROPTOL_U];
  p.kappa_d       = params[LHF_COND_D];
  p.kappa         = params[LHF_COND];
  p.alpha_L       = params[LHF_ALPHA_L];
  p.alpha_U       = params[LHF_ALPHA_U];
  p.verbose       = params[LHF_VERBOSE];
  p.reorder       = params[LHF_REORDER];
  p.symm_pre_lvls = params[LHF_SYMMPRELVLS];
  p.threads       = params[LHF_THREADS];
  p.rrqr_cond     = params[LHF_RRQR_COND];
  p.pivot         = params[LHF_PIVOT];
  p.beta          = params[LHF_BETA];
  p.is_symm       = params[LHF_ISSYMM];
  p.no_pre        = params[LHF_NOPRE];
  p.nzp_thres     = params[LHF_NZP_THRES];
  p.dense_thres   = params[LHF_DENSE_THRES];
  return p;
}

// factorization implementation
template <class PrecType, class MatTypeHdl>
static inline void factorize(PrecType &M, const MatTypeHdl mat,
                             const double params[]) {
  if (params) {
    if (mat->is_rowmajor)
      M.template factorize<true>(mat->n, mat->indptr, mat->indices, mat->vals);
    else
      M.template factorize<false>(mat->n, mat->indptr, mat->indices, mat->vals);
  } else {
    if (mat->is_rowmajor)
      M.template factorize<true>(mat->n, mat->indptr, mat->indices, mat->vals,
                                 create_params(params));
    else
      M.template factorize<false>(mat->n, mat->indptr, mat->indices, mat->vals,
                                  create_params(params));
  }
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

}  // namespace libhifir_impl

extern "C" {

// get versions
void lhfGetVersions(int versions[]) {
  versions[0] = HIF_GLOBAL_VERSION;
  versions[1] = HIF_MAJOR_VERSION;
  versions[2] = HIF_MINOR_VERSION;
}

// enabling warning
void lhfEnableWarning() { hif::warn_flag(1); }

// disable warning
void lhfDisableWarning() { hif::warn_flag(0); }

// error message
const char *lhfGetErrorMsg() {
  libhifir_impl::error_msgs[0] = libhifir_impl::error_msgs[1];
  libhifir_impl::error_msgs[1].clear();
  if (libhifir_impl::error_msgs[0].empty()) return NULL;
  return libhifir_impl::error_msgs[0].c_str();
}

// setup parameters
LhfStatus lhfSetDefaultParams(double params[]) {
  params[LHF_DROPTOL_L]   = hif::DEFAULT_PARAMS.tau_L;
  params[LHF_DROPTOL_U]   = hif::DEFAULT_PARAMS.tau_U;
  params[LHF_COND_D]      = hif::DEFAULT_PARAMS.kappa_d;
  params[LHF_COND]        = hif::DEFAULT_PARAMS.kappa;
  params[LHF_ALPHA_L]     = hif::DEFAULT_PARAMS.alpha_L;
  params[LHF_ALPHA_U]     = hif::DEFAULT_PARAMS.alpha_U;
  params[LHF_VERBOSE]     = hif::DEFAULT_PARAMS.verbose;
  params[LHF_REORDER]     = hif::DEFAULT_PARAMS.reorder;
  params[LHF_SYMMPRELVLS] = hif::DEFAULT_PARAMS.symm_pre_lvls;
  params[LHF_THREADS]     = hif::DEFAULT_PARAMS.threads;
  params[LHF_RRQR_COND]   = hif::DEFAULT_PARAMS.rrqr_cond;
  params[LHF_PIVOT]       = hif::DEFAULT_PARAMS.pivot;
  params[LHF_BETA]        = hif::DEFAULT_PARAMS.beta;
  params[LHF_ISSYMM]      = hif::DEFAULT_PARAMS.is_symm;
  params[LHF_NOPRE]       = hif::DEFAULT_PARAMS.no_pre;
  params[LHF_NZP_THRES]   = hif::DEFAULT_PARAMS.nzp_thres;
  params[LHF_DENSE_THRES] = hif::DEFAULT_PARAMS.dense_thres;
  return LHF_SUCCESS;
}

// set droptol
LhfStatus lhfSetDroptol(const double droptol, double params[]) {
  params[LHF_DROPTOL_L] = params[LHF_DROPTOL_U] = droptol;
  return LHF_SUCCESS;
}

// set fill factors
LhfStatus lhfSetAlpha(const double alpha, double params[]) {
  params[LHF_ALPHA_L] = params[LHF_ALPHA_U] = alpha;
  return LHF_SUCCESS;
}

LhfStatus lhfSetKappa(const double kappa, double params[]) {
  params[LHF_COND] = params[LHF_COND_D] = kappa;
  return LHF_SUCCESS;
}

// double precision matrix
struct LhfdMatrix {
  LhfInt *    indptr, *indices;
  double *    vals;
  std::size_t n;
  bool        is_rowmajor;
};

// create matrix
LhfdMatrixHdl lhfdCreateMatrix(const int is_rowmajor, const size_t n,
                               const LhfInt *indptr, const LhfInt *indices,
                               const double *vals) {
  LhfdMatrixHdl mat = (LhfdMatrixHdl)std::malloc(sizeof(LhfdMatrix));
  if (!mat) return nullptr;
  mat->is_rowmajor = is_rowmajor;
  mat->indptr = mat->indices = nullptr;
  mat->vals                  = nullptr;
  mat->n                     = 0;
  if (n && indptr && indices && vals) {
    mat->indptr  = (LhfInt *)indptr;
    mat->indices = (LhfInt *)indices;
    mat->vals    = (double *)vals;
    mat->n       = n;
  }
  return mat;
}

// destroy a matrix
LhfStatus lhfdDestroyMatrix(LhfdMatrixHdl mat) {
  if (mat) std::free(mat);
  return LHF_SUCCESS;
}

// get size
size_t lhfdGetMatrixSize(const LhfdMatrixHdl mat) {
  return mat ? mat->n : std::size_t(0);
}

// get nnz
size_t lhfdGetMatrixNnz(const LhfdMatrixHdl mat) {
  return mat ? std::size_t(mat->indptr[mat->n] - mat->indptr[0])
             : std::size_t(0);
}

// wrap external matrix
LhfStatus lhfdWrapMatrix(LhfdMatrixHdl mat, const size_t n,
                         const LhfInt *indptr, const LhfInt *indices,
                         const double *vals) {
  _LHF_CANNOT_ACCEPT_NULL(mat);
  if (n && indptr && indices && vals) {
    mat->indptr  = (LhfInt *)indptr;
    mat->indices = (LhfInt *)indices;
    mat->vals    = (double *)vals;
    mat->n       = n;
    return LHF_SUCCESS;
  }
  return LHF_NULL_OBJ;
}

// HIF structure
struct LhfdHif {
  hif::HIF<double, LhfInt> M;
  LhfdMatrixHdl            A;
};

// create function
LhfdHifHdl lhfdCreate(const LhfdMatrixHdl A, const LhfdMatrixHdl S,
                      const double params[]) {
  LhfdHifHdl M = (LhfdHifHdl)std::malloc(sizeof(LhfdHif));
  if (!M) return nullptr;
  const auto status = lhfdSetup(M, A, S, params);
  if (status == LHF_HIFIR_ERROR) {
    lhfdDestroy(M);
    return nullptr;
  }
  return M;
}

// destroy HIF
LhfStatus lhfdDestroy(LhfdHifHdl hif) {
  if (hif) {
    hif->M.~HIF<double, LhfInt>();
    std::free(hif);
  }
  return LHF_SUCCESS;
}

// setup function
LhfStatus lhfdSetup(LhfdHifHdl hif, const LhfdMatrixHdl A,
                    const LhfdMatrixHdl S, const double params[]) {
  _LHF_CANNOT_ACCEPT_NULL(hif);
  hif->A             = A ? A : S;
  LhfdMatrixHdl facA = S ? S : A;
  _LHF_CANNOT_ACCEPT_NULL(facA);
  try {
    libhifir_impl::factorize(hif->M, facA, params);
  } catch (const std::exception &e) {
    _LHF_RETURN_FROM_HIFIR_ERROR(e);
  }
  return LHF_SUCCESS;
}

// Update A
LhfStatus lhfdUpdate(LhfdHifHdl hif, const LhfdMatrixHdl A) {
  _LHF_CANNOT_ACCEPT_NULL(hif);
  hif->A = A;
  if (hif->A && !hif->M.empty())
    if (hif->A->n != hif->M.nrows()) return LHF_MISMATCHED_SIZES;
  return LHF_SUCCESS;
}

// refactorize S
LhfStatus lhfdRefactorize(LhfdHifHdl hif, const LhfdMatrixHdl S,
                          const double params[]) {
  _LHF_CANNOT_ACCEPT_NULL(hif);
  _LHF_CANNOT_ACCEPT_NULL(S);
  try {
    libhifir_impl::factorize(hif->M, S, params);
  } catch (const std::exception &e) {
    _LHF_RETURN_FROM_HIFIR_ERROR(e);
  }
  return LHF_SUCCESS;
}
}