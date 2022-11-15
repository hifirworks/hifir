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

#include <complex>
#include <cstdint>
#include <cstdlib>
#include <string>

// always throw instead of calling abort
#ifndef HIF_THROW
#  define HIF_THROW
#endif

#include <hifir.hpp>
#include "libhifir.h"

#define _LHF_CANNOT_ACCEPT_NULL(__obj) \
  if (!(__obj)) return LHF_NULL_OBJ

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
    stats[0] = M.nnz();
    stats[1] = M.stats(0);
    stats[2] = M.stats(1);
    stats[3] = M.stats(4);
    stats[4] = M.stats(5);
    stats[5] = M.levels();
    stats[6] = M.rank();
    stats[7] = M.schur_rank();
    stats[8] = M.schur_size();
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

// read sparse matrix from matrix market file
template <class ValueType, class MatHdlType>
static inline LhfStatus mm_read_sparse(const char *fname, MatHdlType mat) {
  _LHF_CANNOT_ACCEPT_NULL(mat);
  if (mat->is_rowmajor) {
    using matrix_t = hif::CRS<ValueType, LhfInt, LhfIndPtr>;
    matrix_t A;
    try {
      A = matrix_t::from_mm(fname);
    } catch (const std::exception &e) {
      _LHF_RETURN_FROM_HIFIR_ERROR(e);
    }
    if (A.nrows() != mat->n || A.ncols() != mat->n) return LHF_MISMATCHED_SIZES;
    std::copy(A.ind_start().cbegin(), A.ind_start().cend(), mat->indptr);
    std::copy(A.inds().cbegin(), A.inds().cend(), mat->indices);
    std::copy(A.vals().cbegin(), A.vals().cend(), mat->vals);
  } else {
    using matrix_t = hif::CCS<ValueType, LhfInt, LhfIndPtr>;
    matrix_t A;
    try {
      A = matrix_t::from_mm(fname);
    } catch (const std::exception &e) {
      _LHF_RETURN_FROM_HIFIR_ERROR(e);
    }
    if (A.nrows() != mat->n || A.ncols() != mat->n) return LHF_MISMATCHED_SIZES;
    std::copy(A.ind_start().cbegin(), A.ind_start().cend(), mat->indptr);
    std::copy(A.inds().cbegin(), A.inds().cend(), mat->indices);
    std::copy(A.vals().cbegin(), A.vals().cend(), mat->vals);
  }
  return LHF_SUCCESS;
}

// factorization implementation
template <class PrecType, class MatHdlType>
static inline void factorize(PrecType &M, const MatHdlType mat,
                             const double params[]) {
  if (!params) {
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

// solve implementation
template <class PrecType, class ValueType>
static inline void solve(PrecType &M, const ValueType *b, ValueType *x,
                         const bool trans = false) {
  using array_t              = hif::Array<ValueType>;
  static constexpr bool WRAP = true;

  const array_t rhs(M.nrows(), (ValueType *)b, WRAP);
  array_t       lhs(M.nrows(), x, WRAP);
  M.solve(rhs, lhs, trans);
}

// multiplication implementation
template <class PrecType, class ValueType>
static inline void mmultiply(PrecType &M, const ValueType *b, ValueType *x,
                             const bool trans = false) {
  using array_t              = hif::Array<ValueType>;
  static constexpr bool WRAP = true;

  const array_t rhs(M.nrows(), (ValueType *)b, WRAP);
  array_t       lhs(M.nrows(), x, WRAP);
  M.mmultiply(rhs, lhs, trans);
}

template <class PrecType, class MatHdlType, class ValueType>
static inline void hifir(const PrecType &M, const MatHdlType A,
                         const ValueType *b, const int nirs,
                         const double *betas, const bool trans, const int rank,
                         ValueType *x, int *ir_status) {
  using array_t = hif::Array<ValueType>;

  array_t bb(A->n, (ValueType *)b, true);
  array_t xx(A->n, (ValueType *)x, true);
  if (!betas) {
    // no residual bounds
    if (A->is_rowmajor) {
      const auto AA(hif::wrap_const_crs(A->n, A->n, A->indptr, A->indices,
                                        (const ValueType *)A->vals, false));
      M.hifir(AA, bb, nirs, xx, trans, rank);
    } else {
      const auto AA(hif::wrap_const_ccs(A->n, A->n, A->indptr, A->indices,
                                        (const ValueType *)A->vals, false));
      M.hifir(AA, bb, nirs, xx, trans, rank);
    }
  } else {
    if (A->is_rowmajor) {
      const auto AA(hif::wrap_const_crs(A->n, A->n, A->indptr, A->indices,
                                        (const ValueType *)A->vals, false));
      const auto info = M.hifir(AA, bb, nirs, betas, xx, trans, rank);
      if (ir_status) {
        ir_status[0] = info.first;
        ir_status[1] = info.second;
      }
    } else {
      const auto AA(hif::wrap_const_ccs(A->n, A->n, A->indptr, A->indices,
                                        (const ValueType *)A->vals, false));
      const auto info = M.hifir(AA, bb, nirs, betas, xx, trans, rank);
      if (ir_status) {
        ir_status[0] = info.first;
        ir_status[1] = info.second;
      }
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

// query mm file information
LhfStatus lhfQueryMmFile(const char *fname, int *is_sparse, int *is_real,
                         size_t *nrows, size_t *ncols, size_t *nnz) {
  // open the file
  std::FILE *f = std::fopen(fname, "r");
  if (!f) return LHF_NULL_OBJ;
  bool is_sparse2, is_real2;
  int  type_id;
  // parse the first line
  hif::internal::mm_read_firstline(f, is_sparse2, is_real2, type_id);
  *is_sparse = is_sparse2;
  *is_real   = is_real2;
  (void)type_id;  // not used
  if (is_sparse2)
    hif::internal::mm_read_sparse_size(f, *nrows, *ncols, *nnz);
  else {
    *nnz = 0u;
    hif::internal::mm_read_dense_size(f, *nrows, *ncols);
  }
  std::fclose(f);
  return LHF_SUCCESS;
}

///////////////////////////
// double precision
///////////////////////////

struct LhfdMatrix {
  LhfIndPtr * indptr;
  LhfInt *    indices;
  double *    vals;
  std::size_t n;
  bool        is_rowmajor;
};

// create matrix
LhfdMatrixHdl lhfdCreateMatrix(const int is_rowmajor, const size_t n,
                               const LhfIndPtr *indptr, const LhfInt *indices,
                               const double *vals) {
  LhfdMatrixHdl mat = (LhfdMatrixHdl)std::malloc(sizeof(LhfdMatrix));
  if (!mat) return nullptr;
  mat->is_rowmajor = is_rowmajor;
  mat->indptr      = nullptr;
  mat->indices     = nullptr;
  mat->vals        = nullptr;
  mat->n           = 0;
  if (n && indptr && indices && vals) {
    mat->indptr  = (LhfIndPtr *)indptr;
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
                         const LhfIndPtr *indptr, const LhfInt *indices,
                         const double *vals) {
  _LHF_CANNOT_ACCEPT_NULL(mat);
  if (n && indptr && indices && vals) {
    mat->indptr  = (LhfIndPtr *)indptr;
    mat->indices = (LhfInt *)indices;
    mat->vals    = (double *)vals;
    mat->n       = n;
    return LHF_SUCCESS;
  }
  return LHF_NULL_OBJ;
}

// load sparse matrix
LhfStatus lhfdReadSparse(const char *fname, LhfdMatrixHdl mat) {
  return libhifir_impl::mm_read_sparse<double>(fname, mat);
}

// double vector mm io
LhfStatus lhfdReadVector(const char *fname, const size_t n, double *v) {
  using vector_t = hif::Array<double>;
  vector_t vec;
  try {
    vec = vector_t::from_mm(fname);
  } catch (const std::exception &e) {
    _LHF_RETURN_FROM_HIFIR_ERROR(e);
  }
  if (n != vec.size()) return LHF_MISMATCHED_SIZES;
  std::copy(vec.cbegin(), vec.cend(), v);
  return LHF_SUCCESS;
}

// HIF structure
struct LhfdHif {
  hif::HIF<double, LhfInt, LhfIndPtr> *M;
  LhfdMatrixHdl                        A;
};

// create function
LhfdHifHdl lhfdCreate(const LhfdMatrixHdl A, const LhfdMatrixHdl S,
                      const double params[]) {
  LhfdHifHdl M = (LhfdHifHdl)std::malloc(sizeof(LhfdHif));
  if (!M) return nullptr;
  M->M = new (std::nothrow) hif::HIF<double, LhfInt, LhfIndPtr>();
  if (!M->M) {
    std::free(M);
    return nullptr;
  }
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
    delete hif->M;
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
    libhifir_impl::factorize(*hif->M, facA, params);
  } catch (const std::exception &e) {
    _LHF_RETURN_FROM_HIFIR_ERROR(e);
  }
  return LHF_SUCCESS;
}

// Update A
LhfStatus lhfdUpdate(LhfdHifHdl hif, const LhfdMatrixHdl A) {
  _LHF_CANNOT_ACCEPT_NULL(hif);
  hif->A = A;
  if (hif->A && !hif->M->empty())
    if (hif->A->n != hif->M->nrows()) return LHF_MISMATCHED_SIZES;
  return LHF_SUCCESS;
}

// refactorize S
LhfStatus lhfdRefactorize(LhfdHifHdl hif, const LhfdMatrixHdl S,
                          const double params[]) {
  _LHF_CANNOT_ACCEPT_NULL(hif);
  _LHF_CANNOT_ACCEPT_NULL(S);
  try {
    libhifir_impl::factorize(*hif->M, S, params);
  } catch (const std::exception &e) {
    _LHF_RETURN_FROM_HIFIR_ERROR(e);
  }
  return LHF_SUCCESS;
}

// Apply
LhfStatus lhfdApply(const LhfdHifHdl hif, const LhfOperationType op,
                    const double *b, const int nirs, const double *betas,
                    const int rank, double *x, int *ir_status) {
  _LHF_CANNOT_ACCEPT_NULL(hif);
  const bool trans = op == LHF_SH || op == LHF_MH;
  // determine rank
  int rnk(rank);
  if (rank == LHF_DEFAULT_RANK)
    rnk = ((op != LHF_S && op != LHF_SH) || nirs > 1) ? -1 : 0;
  try {
    if (nirs <= 1 || op == LHF_M || op == LHF_MH) {
      if (op == LHF_M || op == LHF_MH)
        libhifir_impl::mmultiply(*hif->M, b, x, trans);
      else
        libhifir_impl::solve(*hif->M, b, x, trans);
    } else {
      _LHF_CANNOT_ACCEPT_NULL(hif->A);
      if (hif->M->nrows() != hif->A->n) return LHF_MISMATCHED_SIZES;
      libhifir_impl::hifir(*hif->M, hif->A, b, nirs, betas, trans, rnk, x,
                           ir_status);
    }
  } catch (const std::exception &e) {
    _LHF_RETURN_FROM_HIFIR_ERROR(e);
  }
  return LHF_SUCCESS;
}

// Solve
LhfStatus lhfdSolve(const LhfdHifHdl hif, const double *b, double *x) {
  _LHF_CANNOT_ACCEPT_NULL(hif);
  try {
    libhifir_impl::solve(*hif->M, b, x);
  } catch (const std::exception &e) {
    _LHF_RETURN_FROM_HIFIR_ERROR(e);
  }
  return LHF_SUCCESS;
}

// query status
LhfStatus lhfdGetStats(const LhfdHifHdl hif, size_t stats[]) {
  if (!hif)
    for (int i = 0; i < 9; ++i) stats[i] = 0;
  else
    libhifir_impl::get_factor_stats(*hif->M, stats);
  return LHF_SUCCESS;
}

// utilities
size_t lhfdGetNnz(const LhfdHifHdl hif) {
  return hif ? hif->M->nnz() : std::size_t(0);
}

size_t lhfdGetLevels(const LhfdHifHdl hif) {
  return hif ? hif->M->levels() : std::size_t(0);
}

size_t lhfdGetSchurSize(const LhfdHifHdl hif) {
  return hif ? hif->M->schur_size() : std::size_t(0);
}

size_t lhfdGetSchurRank(const LhfdHifHdl hif) {
  return hif ? hif->M->schur_rank() : std::size_t(0);
}

///////////////////////////
// single precision
///////////////////////////

struct LhfsMatrix {
  LhfIndPtr * indptr;
  LhfInt *    indices;
  float *     vals;
  std::size_t n;
  bool        is_rowmajor;
};

// create matrix
LhfsMatrixHdl lhfsCreateMatrix(const int is_rowmajor, const size_t n,
                               const LhfIndPtr *indptr, const LhfInt *indices,
                               const float *vals) {
  LhfsMatrixHdl mat = (LhfsMatrixHdl)std::malloc(sizeof(LhfsMatrix));
  if (!mat) return nullptr;
  mat->is_rowmajor = is_rowmajor;
  mat->indptr      = nullptr;
  mat->indices     = nullptr;
  mat->vals        = nullptr;
  mat->n           = 0;
  if (n && indptr && indices && vals) {
    mat->indptr  = (LhfIndPtr *)indptr;
    mat->indices = (LhfInt *)indices;
    mat->vals    = (float *)vals;
    mat->n       = n;
  }
  return mat;
}

// destroy a matrix
LhfStatus lhfsDestroyMatrix(LhfsMatrixHdl mat) {
  if (mat) std::free(mat);
  return LHF_SUCCESS;
}

// get size
size_t lhfsGetMatrixSize(const LhfsMatrixHdl mat) {
  return mat ? mat->n : std::size_t(0);
}

// get nnz
size_t lhfsGetMatrixNnz(const LhfsMatrixHdl mat) {
  return mat ? std::size_t(mat->indptr[mat->n] - mat->indptr[0])
             : std::size_t(0);
}

// wrap external matrix
LhfStatus lhfsWrapMatrix(LhfsMatrixHdl mat, const size_t n,
                         const LhfIndPtr *indptr, const LhfInt *indices,
                         const float *vals) {
  _LHF_CANNOT_ACCEPT_NULL(mat);
  if (n && indptr && indices && vals) {
    mat->indptr  = (LhfIndPtr *)indptr;
    mat->indices = (LhfInt *)indices;
    mat->vals    = (float *)vals;
    mat->n       = n;
    return LHF_SUCCESS;
  }
  return LHF_NULL_OBJ;
}

// load sparse matrix
LhfStatus lhfsReadSparse(const char *fname, LhfsMatrixHdl mat) {
  return libhifir_impl::mm_read_sparse<float>(fname, mat);
}

// single vector mm io
LhfStatus lhfsReadVector(const char *fname, const size_t n, float *v) {
  using vector_t = hif::Array<float>;
  vector_t vec;
  try {
    vec = vector_t::from_mm(fname);
  } catch (const std::exception &e) {
    _LHF_RETURN_FROM_HIFIR_ERROR(e);
  }
  if (n != vec.size()) return LHF_MISMATCHED_SIZES;
  std::copy(vec.cbegin(), vec.cend(), v);
  return LHF_SUCCESS;
}

// HIF structure
struct LhfsHif {
  hif::HIF<float, LhfInt, LhfIndPtr> *M;
  LhfsMatrixHdl                       A;
  LhfdMatrixHdl                       Ad;
};

// create function
LhfsHifHdl lhfsCreate(const LhfsMatrixHdl A, const LhfsMatrixHdl S,
                      const double params[]) {
  LhfsHifHdl M = (LhfsHifHdl)std::malloc(sizeof(LhfsHif));
  if (!M) return nullptr;
  M->M = new (std::nothrow) hif::HIF<float, LhfInt, LhfIndPtr>();
  if (!M->M) {
    std::free(M);
    return nullptr;
  }
  const auto status = lhfsSetup(M, A, S, params);
  if (status == LHF_HIFIR_ERROR) {
    lhfsDestroy(M);
    return nullptr;
  }
  return M;
}

// destroy HIF
LhfStatus lhfsDestroy(LhfsHifHdl hif) {
  if (hif) {
    delete hif->M;
    std::free(hif);
  }
  return LHF_SUCCESS;
}

// setup function
LhfStatus lhfsSetup(LhfsHifHdl hif, const LhfsMatrixHdl A,
                    const LhfsMatrixHdl S, const double params[]) {
  _LHF_CANNOT_ACCEPT_NULL(hif);
  hif->A             = A ? A : S;
  LhfsMatrixHdl facA = S ? S : A;
  hif->Ad            = nullptr;
  _LHF_CANNOT_ACCEPT_NULL(facA);
  try {
    libhifir_impl::factorize(*hif->M, facA, params);
  } catch (const std::exception &e) {
    _LHF_RETURN_FROM_HIFIR_ERROR(e);
  }
  return LHF_SUCCESS;
}

// Update A
LhfStatus lhfsUpdate(LhfsHifHdl hif, const LhfsMatrixHdl A) {
  _LHF_CANNOT_ACCEPT_NULL(hif);
  hif->A = A;
  if (hif->A && !hif->M->empty())
    if (hif->A->n != hif->M->nrows()) return LHF_MISMATCHED_SIZES;
  return LHF_SUCCESS;
}

// refactorize S
LhfStatus lhfsRefactorize(LhfsHifHdl hif, const LhfsMatrixHdl S,
                          const double params[]) {
  _LHF_CANNOT_ACCEPT_NULL(hif);
  _LHF_CANNOT_ACCEPT_NULL(S);
  try {
    libhifir_impl::factorize(*hif->M, S, params);
  } catch (const std::exception &e) {
    _LHF_RETURN_FROM_HIFIR_ERROR(e);
  }
  return LHF_SUCCESS;
}

// Apply
LhfStatus lhfsApply(const LhfsHifHdl hif, const LhfOperationType op,
                    const float *b, const int nirs, const double *betas,
                    const int rank, float *x, int *ir_status) {
  _LHF_CANNOT_ACCEPT_NULL(hif);
  const bool trans = op == LHF_SH || op == LHF_MH;
  // determine rank
  int rnk(rank);
  if (rank == LHF_DEFAULT_RANK)
    rnk = ((op != LHF_S && op != LHF_SH) || nirs > 1) ? -1 : 0;
  try {
    if (nirs <= 1 || op == LHF_M || op == LHF_MH) {
      if (op == LHF_M || op == LHF_MH)
        libhifir_impl::mmultiply(*hif->M, b, x, trans);
      else
        libhifir_impl::solve(*hif->M, b, x, trans);
    } else {
      _LHF_CANNOT_ACCEPT_NULL(hif->A);
      if (hif->M->nrows() != hif->A->n) return LHF_MISMATCHED_SIZES;
      libhifir_impl::hifir(*hif->M, hif->A, b, nirs, betas, trans, rnk, x,
                           ir_status);
    }
  } catch (const std::exception &e) {
    _LHF_RETURN_FROM_HIFIR_ERROR(e);
  }
  return LHF_SUCCESS;
}

// Solve
LhfStatus lhfsSolve(const LhfsHifHdl hif, const float *b, float *x) {
  _LHF_CANNOT_ACCEPT_NULL(hif);
  try {
    libhifir_impl::solve(*hif->M, b, x);
  } catch (const std::exception &e) {
    _LHF_RETURN_FROM_HIFIR_ERROR(e);
  }
  return LHF_SUCCESS;
}

// query status
LhfStatus lhfsGetStats(const LhfsHifHdl hif, size_t stats[]) {
  if (!hif)
    for (int i = 0; i < 9; ++i) stats[i] = 0;
  else
    libhifir_impl::get_factor_stats(*hif->M, stats);
  return LHF_SUCCESS;
}

// utilities
size_t lhfsGetNnz(const LhfsHifHdl hif) {
  return hif ? hif->M->nnz() : std::size_t(0);
}

size_t lhfsGetLevels(const LhfsHifHdl hif) {
  return hif ? hif->M->levels() : std::size_t(0);
}

size_t lhfsGetSchurSize(const LhfsHifHdl hif) {
  return hif ? hif->M->schur_size() : std::size_t(0);
}

size_t lhfsGetSchurRank(const LhfsHifHdl hif) {
  return hif ? hif->M->schur_rank() : std::size_t(0);
}

///////////////////////////
// double complex
///////////////////////////

struct LhfzMatrix {
  LhfIndPtr *           indptr;
  LhfInt *              indices;
  std::complex<double> *vals;
  std::size_t           n;
  bool                  is_rowmajor;
};

// create matrix
LhfzMatrixHdl lhfzCreateMatrix(const int is_rowmajor, const size_t n,
                               const LhfIndPtr *indptr, const LhfInt *indices,
                               const double _Complex *vals) {
  LhfzMatrixHdl mat = (LhfzMatrixHdl)std::malloc(sizeof(LhfzMatrix));
  if (!mat) return nullptr;
  mat->is_rowmajor = is_rowmajor;
  mat->indptr      = nullptr;
  mat->indices     = nullptr;
  mat->vals        = nullptr;
  mat->n           = 0;
  if (n && indptr && indices && vals) {
    mat->indptr  = (LhfIndPtr *)indptr;
    mat->indices = (LhfInt *)indices;
    mat->vals    = (std::complex<double> *)vals;
    mat->n       = n;
  }
  return mat;
}

// destroy a matrix
LhfStatus lhfzDestroyMatrix(LhfzMatrixHdl mat) {
  if (mat) std::free(mat);
  return LHF_SUCCESS;
}

// get size
size_t lhfzGetMatrixSize(const LhfzMatrixHdl mat) {
  return mat ? mat->n : std::size_t(0);
}

// get nnz
size_t lhfzGetMatrixNnz(const LhfzMatrixHdl mat) {
  return mat ? std::size_t(mat->indptr[mat->n] - mat->indptr[0])
             : std::size_t(0);
}

// wrap external matrix
LhfStatus lhfzWrapMatrix(LhfzMatrixHdl mat, const size_t n,
                         const LhfIndPtr *indptr, const LhfInt *indices,
                         const double _Complex *vals) {
  _LHF_CANNOT_ACCEPT_NULL(mat);
  if (n && indptr && indices && vals) {
    mat->indptr  = (LhfIndPtr *)indptr;
    mat->indices = (LhfInt *)indices;
    mat->vals    = (std::complex<double> *)vals;
    mat->n       = n;
    return LHF_SUCCESS;
  }
  return LHF_NULL_OBJ;
}

// load sparse matrix
LhfStatus lhfzReadSparse(const char *fname, LhfzMatrixHdl mat) {
  return libhifir_impl::mm_read_sparse<std::complex<double>>(fname, mat);
}

// double complex vector mm io
LhfStatus lhfzReadVector(const char *fname, const size_t n,
                         double _Complex *v) {
  using vector_t = hif::Array<std::complex<double>>;
  vector_t vec;
  try {
    vec = vector_t::from_mm(fname);
  } catch (const std::exception &e) {
    _LHF_RETURN_FROM_HIFIR_ERROR(e);
  }
  if (n != vec.size()) return LHF_MISMATCHED_SIZES;
  std::copy(vec.cbegin(), vec.cend(), (std::complex<double> *)v);
  return LHF_SUCCESS;
}

// HIF structure
struct LhfzHif {
  hif::HIF<std::complex<double>, LhfInt, LhfIndPtr> *M;
  LhfzMatrixHdl                                      A;
};

// create function
LhfzHifHdl lhfzCreate(const LhfzMatrixHdl A, const LhfzMatrixHdl S,
                      const double params[]) {
  LhfzHifHdl M = (LhfzHifHdl)std::malloc(sizeof(LhfzHif));
  if (!M) return nullptr;
  M->M = new (std::nothrow) hif::HIF<std::complex<double>, LhfInt, LhfIndPtr>();
  if (!M->M) {
    std::free(M);
    return nullptr;
  }
  const auto status = lhfzSetup(M, A, S, params);
  if (status == LHF_HIFIR_ERROR) {
    lhfzDestroy(M);
    return nullptr;
  }
  return M;
}

// destroy HIF
LhfStatus lhfzDestroy(LhfzHifHdl hif) {
  if (hif) {
    delete hif->M;
    std::free(hif);
  }
  return LHF_SUCCESS;
}

// setup function
LhfStatus lhfzSetup(LhfzHifHdl hif, const LhfzMatrixHdl A,
                    const LhfzMatrixHdl S, const double params[]) {
  _LHF_CANNOT_ACCEPT_NULL(hif);
  hif->A             = A ? A : S;
  LhfzMatrixHdl facA = S ? S : A;
  _LHF_CANNOT_ACCEPT_NULL(facA);
  try {
    libhifir_impl::factorize(*hif->M, facA, params);
  } catch (const std::exception &e) {
    _LHF_RETURN_FROM_HIFIR_ERROR(e);
  }
  return LHF_SUCCESS;
}

// Update A
LhfStatus lhfzUpdate(LhfzHifHdl hif, const LhfzMatrixHdl A) {
  _LHF_CANNOT_ACCEPT_NULL(hif);
  hif->A = A;
  if (hif->A && !hif->M->empty())
    if (hif->A->n != hif->M->nrows()) return LHF_MISMATCHED_SIZES;
  return LHF_SUCCESS;
}

// refactorize S
LhfStatus lhfzRefactorize(LhfzHifHdl hif, const LhfzMatrixHdl S,
                          const double params[]) {
  _LHF_CANNOT_ACCEPT_NULL(hif);
  _LHF_CANNOT_ACCEPT_NULL(S);
  try {
    libhifir_impl::factorize(*hif->M, S, params);
  } catch (const std::exception &e) {
    _LHF_RETURN_FROM_HIFIR_ERROR(e);
  }
  return LHF_SUCCESS;
}

// Apply
LhfStatus lhfzApply(const LhfzHifHdl hif, const LhfOperationType op,
                    const double _Complex *b, const int nirs,
                    const double *betas, const int rank, double _Complex *x,
                    int *ir_status) {
  _LHF_CANNOT_ACCEPT_NULL(hif);
  const bool trans = op == LHF_SH || op == LHF_MH;
  // determine rank
  int rnk(rank);
  if (rank == LHF_DEFAULT_RANK)
    rnk = ((op != LHF_S && op != LHF_SH) || nirs > 1) ? -1 : 0;
  try {
    if (nirs <= 1 || op == LHF_M || op == LHF_MH) {
      if (op == LHF_M || op == LHF_MH)
        libhifir_impl::mmultiply(*hif->M, (const std::complex<double> *)b,
                                 (std::complex<double> *)x, trans);
      else
        libhifir_impl::solve(*hif->M, (const std::complex<double> *)b,
                             (std::complex<double> *)x, trans);
    } else {
      _LHF_CANNOT_ACCEPT_NULL(hif->A);
      if (hif->M->nrows() != hif->A->n) return LHF_MISMATCHED_SIZES;
      libhifir_impl::hifir(*hif->M, hif->A, (const std::complex<double> *)b,
                           nirs, betas, trans, rnk, (std::complex<double> *)x,
                           ir_status);
    }
  } catch (const std::exception &e) {
    _LHF_RETURN_FROM_HIFIR_ERROR(e);
  }
  return LHF_SUCCESS;
}

// Solve
LhfStatus lhfzSolve(const LhfzHifHdl hif, const double _Complex *b,
                    double _Complex *x) {
  _LHF_CANNOT_ACCEPT_NULL(hif);
  try {
    libhifir_impl::solve(*hif->M, (const std::complex<double> *)b,
                         (std::complex<double> *)x);
  } catch (const std::exception &e) {
    _LHF_RETURN_FROM_HIFIR_ERROR(e);
  }
  return LHF_SUCCESS;
}

// query status
LhfStatus lhfzGetStats(const LhfzHifHdl hif, size_t stats[]) {
  if (!hif)
    for (int i = 0; i < 9; ++i) stats[i] = 0;
  else
    libhifir_impl::get_factor_stats(*hif->M, stats);
  return LHF_SUCCESS;
}

// utilities
size_t lhfzGetNnz(const LhfzHifHdl hif) {
  return hif ? hif->M->nnz() : std::size_t(0);
}

size_t lhfzGetLevels(const LhfzHifHdl hif) {
  return hif ? hif->M->levels() : std::size_t(0);
}

size_t lhfzGetSchurSize(const LhfzHifHdl hif) {
  return hif ? hif->M->schur_size() : std::size_t(0);
}

size_t lhfzGetSchurRank(const LhfzHifHdl hif) {
  return hif ? hif->M->schur_rank() : std::size_t(0);
}

///////////////////////////
// single complex
///////////////////////////

struct LhfcMatrix {
  LhfIndPtr *          indptr;
  LhfInt *             indices;
  std::complex<float> *vals;
  std::size_t          n;
  bool                 is_rowmajor;
};

// create matrix
LhfcMatrixHdl lhfcCreateMatrix(const int is_rowmajor, const size_t n,
                               const LhfIndPtr *indptr, const LhfInt *indices,
                               const float _Complex *vals) {
  LhfcMatrixHdl mat = (LhfcMatrixHdl)std::malloc(sizeof(LhfcMatrix));
  if (!mat) return nullptr;
  mat->is_rowmajor = is_rowmajor;
  mat->indptr      = nullptr;
  mat->indices     = nullptr;
  mat->vals        = nullptr;
  mat->n           = 0;
  if (n && indptr && indices && vals) {
    mat->indptr  = (LhfIndPtr *)indptr;
    mat->indices = (LhfInt *)indices;
    mat->vals    = (std::complex<float> *)vals;
    mat->n       = n;
  }
  return mat;
}

// destroy a matrix
LhfStatus lhfcDestroyMatrix(LhfcMatrixHdl mat) {
  if (mat) std::free(mat);
  return LHF_SUCCESS;
}

// get size
size_t lhfcGetMatrixSize(const LhfcMatrixHdl mat) {
  return mat ? mat->n : std::size_t(0);
}

// get nnz
size_t lhfcGetMatrixNnz(const LhfcMatrixHdl mat) {
  return mat ? std::size_t(mat->indptr[mat->n] - mat->indptr[0])
             : std::size_t(0);
}

// wrap external matrix
LhfStatus lhfcWrapMatrix(LhfcMatrixHdl mat, const size_t n,
                         const LhfIndPtr *indptr, const LhfInt *indices,
                         const float _Complex *vals) {
  _LHF_CANNOT_ACCEPT_NULL(mat);
  if (n && indptr && indices && vals) {
    mat->indptr  = (LhfIndPtr *)indptr;
    mat->indices = (LhfInt *)indices;
    mat->vals    = (std::complex<float> *)vals;
    mat->n       = n;
    return LHF_SUCCESS;
  }
  return LHF_NULL_OBJ;
}

// load sparse matrix
LhfStatus lhfcReadSparse(const char *fname, LhfcMatrixHdl mat) {
  return libhifir_impl::mm_read_sparse<std::complex<float>>(fname, mat);
}

// double complex vector mm io
LhfStatus lhfcReadVector(const char *fname, const size_t n, float _Complex *v) {
  using vector_t = hif::Array<std::complex<float>>;
  vector_t vec;
  try {
    vec = vector_t::from_mm(fname);
  } catch (const std::exception &e) {
    _LHF_RETURN_FROM_HIFIR_ERROR(e);
  }
  if (n != vec.size()) return LHF_MISMATCHED_SIZES;
  std::copy(vec.cbegin(), vec.cend(), (std::complex<float> *)v);
  return LHF_SUCCESS;
}

// HIF structure
struct LhfcHif {
  hif::HIF<std::complex<float>, LhfInt, LhfIndPtr> *M;
  LhfcMatrixHdl                                     A;
  LhfzMatrixHdl                                     Az;
};

// create function
LhfcHifHdl lhfcCreate(const LhfcMatrixHdl A, const LhfcMatrixHdl S,
                      const double params[]) {
  LhfcHifHdl M = (LhfcHifHdl)std::malloc(sizeof(LhfcHif));
  if (!M) return nullptr;
  M->M = new (std::nothrow) hif::HIF<std::complex<float>, LhfInt, LhfIndPtr>();
  const auto status = lhfcSetup(M, A, S, params);
  if (status == LHF_HIFIR_ERROR) {
    lhfcDestroy(M);
    return nullptr;
  }
  return M;
}

// destroy HIF
LhfStatus lhfcDestroy(LhfcHifHdl hif) {
  if (hif) {
    delete hif->M;
    std::free(hif);
  }
  return LHF_SUCCESS;
}

// setup function
LhfStatus lhfcSetup(LhfcHifHdl hif, const LhfcMatrixHdl A,
                    const LhfcMatrixHdl S, const double params[]) {
  _LHF_CANNOT_ACCEPT_NULL(hif);
  hif->A             = A ? A : S;
  LhfcMatrixHdl facA = S ? S : A;
  hif->Az            = nullptr;
  _LHF_CANNOT_ACCEPT_NULL(facA);
  try {
    libhifir_impl::factorize(*hif->M, facA, params);
  } catch (const std::exception &e) {
    _LHF_RETURN_FROM_HIFIR_ERROR(e);
  }
  return LHF_SUCCESS;
}

// Update A
LhfStatus lhfcUpdate(LhfcHifHdl hif, const LhfcMatrixHdl A) {
  _LHF_CANNOT_ACCEPT_NULL(hif);
  hif->A = A;
  if (hif->A && !hif->M->empty())
    if (hif->A->n != hif->M->nrows()) return LHF_MISMATCHED_SIZES;
  return LHF_SUCCESS;
}

// refactorize S
LhfStatus lhfcRefactorize(LhfcHifHdl hif, const LhfcMatrixHdl S,
                          const double params[]) {
  _LHF_CANNOT_ACCEPT_NULL(hif);
  _LHF_CANNOT_ACCEPT_NULL(S);
  try {
    libhifir_impl::factorize(*hif->M, S, params);
  } catch (const std::exception &e) {
    _LHF_RETURN_FROM_HIFIR_ERROR(e);
  }
  return LHF_SUCCESS;
}

// Apply
LhfStatus lhfcApply(const LhfcHifHdl hif, const LhfOperationType op,
                    const float _Complex *b, const int nirs,
                    const double *betas, const int rank, float _Complex *x,
                    int *ir_status) {
  _LHF_CANNOT_ACCEPT_NULL(hif);
  const bool trans = op == LHF_SH || op == LHF_MH;
  // determine rank
  int rnk(rank);
  if (rank == LHF_DEFAULT_RANK)
    rnk = ((op != LHF_S && op != LHF_SH) || nirs > 1) ? -1 : 0;
  try {
    if (nirs <= 1 || op == LHF_M || op == LHF_MH) {
      if (op == LHF_M || op == LHF_MH)
        libhifir_impl::mmultiply(*hif->M, (const std::complex<float> *)b,
                                 (std::complex<float> *)x, trans);
      else
        libhifir_impl::solve(*hif->M, (const std::complex<float> *)b,
                             (std::complex<float> *)x, trans);
    } else {
      _LHF_CANNOT_ACCEPT_NULL(hif->A);
      if (hif->M->nrows() != hif->A->n) return LHF_MISMATCHED_SIZES;
      libhifir_impl::hifir(*hif->M, hif->A, (const std::complex<float> *)b,
                           nirs, betas, trans, rnk, (std::complex<float> *)x,
                           ir_status);
    }
  } catch (const std::exception &e) {
    _LHF_RETURN_FROM_HIFIR_ERROR(e);
  }
  return LHF_SUCCESS;
}

// Solve
LhfStatus lhfcSolve(const LhfcHifHdl hif, const float _Complex *b,
                    float _Complex *x) {
  _LHF_CANNOT_ACCEPT_NULL(hif);
  try {
    libhifir_impl::solve(*hif->M, (const std::complex<float> *)b,
                         (std::complex<float> *)x);
  } catch (const std::exception &e) {
    _LHF_RETURN_FROM_HIFIR_ERROR(e);
  }
  return LHF_SUCCESS;
}

// query status
LhfStatus lhfcGetStats(const LhfcHifHdl hif, size_t stats[]) {
  if (!hif)
    for (int i = 0; i < 9; ++i) stats[i] = 0;
  else
    libhifir_impl::get_factor_stats(*hif->M, stats);
  return LHF_SUCCESS;
}

// utilities
size_t lhfcGetNnz(const LhfcHifHdl hif) {
  return hif ? hif->M->nnz() : std::size_t(0);
}

size_t lhfcGetLevels(const LhfcHifHdl hif) {
  return hif ? hif->M->levels() : std::size_t(0);
}

size_t lhfcGetSchurSize(const LhfcHifHdl hif) {
  return hif ? hif->M->schur_size() : std::size_t(0);
}

size_t lhfcGetSchurRank(const LhfcHifHdl hif) {
  return hif ? hif->M->schur_rank() : std::size_t(0);
}

/////////////////////////
// mixed precision
/////////////////////////

// update Ad in hif by a double-precision matrix
LhfStatus lhfsdUpdate(LhfsHifHdl hif, LhfdMatrixHdl A) {
  _LHF_CANNOT_ACCEPT_NULL(hif);
  hif->Ad = A;
  if (hif->Ad && !hif->M->empty())
    if (hif->Ad->n != hif->M->nrows()) return LHF_MISMATCHED_SIZES;
  return LHF_SUCCESS;
}

// apply single prec to double solutions
LhfStatus lhfsdApply(const LhfsHifHdl hif, const LhfOperationType op,
                     const double *b, const int nirs, const double *betas,
                     const int rank, double *x, int *ir_status) {
  _LHF_CANNOT_ACCEPT_NULL(hif);
  const bool trans = op == LHF_SH || op == LHF_MH;
  // determine rank
  int rnk(rank);
  if (rank == LHF_DEFAULT_RANK)
    rnk = ((op != LHF_S && op != LHF_SH) || nirs > 1) ? -1 : 0;
  try {
    if (nirs <= 1 || op == LHF_M || op == LHF_MH) {
      if (op == LHF_M || op == LHF_MH)
        libhifir_impl::mmultiply(*hif->M, b, x, trans);
      else
        libhifir_impl::solve(*hif->M, b, x, trans);
    } else {
      // Ad not A
      _LHF_CANNOT_ACCEPT_NULL(hif->Ad);
      if (hif->M->nrows() != hif->Ad->n) return LHF_MISMATCHED_SIZES;
      libhifir_impl::hifir(*hif->M, hif->Ad, b, nirs, betas, trans, rnk, x,
                           ir_status);
    }
  } catch (const std::exception &e) {
    _LHF_RETURN_FROM_HIFIR_ERROR(e);
  }
  return LHF_SUCCESS;
}

// solve
LhfStatus lhfsdSolve(const LhfsHifHdl hif, const double *b, double *x) {
  _LHF_CANNOT_ACCEPT_NULL(hif);
  try {
    libhifir_impl::solve(*hif->M, b, x);
  } catch (const std::exception &e) {
    _LHF_RETURN_FROM_HIFIR_ERROR(e);
  }
  return LHF_SUCCESS;
}

// update Az in hif by a double-precision matrix
LhfStatus lhfczUpdate(LhfcHifHdl hif, LhfzMatrixHdl A) {
  _LHF_CANNOT_ACCEPT_NULL(hif);
  hif->Az = A;
  if (hif->Az && !hif->M->empty())
    if (hif->Az->n != hif->M->nrows()) return LHF_MISMATCHED_SIZES;
  return LHF_SUCCESS;
}

// apply single prec to double solutions (complex)
LhfStatus lhfczApply(const LhfcHifHdl hif, const LhfOperationType op,
                     const double _Complex *b, const int nirs,
                     const double *betas, const int rank, double _Complex *x,
                     int *ir_status) {
  _LHF_CANNOT_ACCEPT_NULL(hif);
  const bool trans = op == LHF_SH || op == LHF_MH;
  // determine rank
  int rnk(rank);
  if (rank == LHF_DEFAULT_RANK)
    rnk = ((op != LHF_S && op != LHF_SH) || nirs > 1) ? -1 : 0;
  try {
    if (nirs <= 1 || op == LHF_M || op == LHF_MH) {
      if (op == LHF_M || op == LHF_MH)
        libhifir_impl::mmultiply(*hif->M, (const std::complex<double> *)b,
                                 (std::complex<double> *)x, trans);
      else
        libhifir_impl::solve(*hif->M, (const std::complex<double> *)b,
                             (std::complex<double> *)x, trans);
    } else {
      // Az not A
      _LHF_CANNOT_ACCEPT_NULL(hif->Az);
      if (hif->M->nrows() != hif->Az->n) return LHF_MISMATCHED_SIZES;
      libhifir_impl::hifir(*hif->M, hif->Az, (const std::complex<double> *)b,
                           nirs, betas, trans, rnk, (std::complex<double> *)x,
                           ir_status);
    }
  } catch (const std::exception &e) {
    _LHF_RETURN_FROM_HIFIR_ERROR(e);
  }
  return LHF_SUCCESS;
}

// Solve (mixed complex single and double)
LhfStatus lhfczSolve(const LhfcHifHdl hif, const double _Complex *b,
                     double _Complex *x) {
  _LHF_CANNOT_ACCEPT_NULL(hif);
  try {
    libhifir_impl::solve(*hif->M, (const std::complex<double> *)b,
                         (std::complex<double> *)x);
  } catch (const std::exception &e) {
    _LHF_RETURN_FROM_HIFIR_ERROR(e);
  }
  return LHF_SUCCESS;
}
}