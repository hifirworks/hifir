/*
///////////////////////////////////////////////////////////////////////////////
//                  This file is part of the HIFIR library                   //
///////////////////////////////////////////////////////////////////////////////
*/

/*!
 * @file libhifir.h
 * @brief C library interface header file
 * @author Qiao Chen

@verbatim
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
@endverbatim

 */

#ifndef _LIBHIFIR_H
#define _LIBHIFIR_H

#include <complex.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*!
 * @addtogroup c
 * @{
 */

#ifndef LIBHIFIR_INT_SIZE
#  define LIBHIFIR_INT_SIZE 32
#endif

#if LIBHIFIR_INT_SIZE != 32 && LIBHIFIR_INT_SIZE != 64
#  error "Unsupported integer size, must be either 32 or 64!"
#endif
#if LIBHIFIR_INT_SIZE == 32
typedef int32_t LhfInt;
#else
typedef int64_t LhfInt;
#endif

/*!
 * We use an anonymous enum to group parameter positions in an array. The first
 * thing one needs to do is to create a @a double array of length
 * LHF_NUMBER_PARAMSs.
 *
 * @code{.c}
 *  double params[LHF_NUMBER_PARAMS];
 *  lhfSetDefaultParams(params);
 *  params[LHF_DROPTOL_L] = params[LHF_DROPTOL_U] = 3.0;
 * @endcode
 */
enum {
  LHF_DROPTOL_L = 0, /*!< Drop tolerance for L factor */
  LHF_DROPTOL_U,     /*!< Drop tolerance for U factor */
  LHF_COND_D,        /*!< Conditioning threshold for diagonal */
  LHF_COND,          /*!< Conditioning threshold for L and U factors */
  LHF_ALPHA_L,       /*!< Scalability-oriented dropping factor for L */
  LHF_ALPHA_U,       /*!< Scalability-oriented dropping factor for U */
  LHF_VERBOSE,       /*!< Verbose level */
  LHF_REORDER,       /*!< Reorder option */
  LHF_SYMMPRELVLS,   /*!< Option for tuning symmetric preprocessing levels */
  LHF_THREADS,       /*!< Option for choosing nthreads for Schur computation */
  LHF_RRQR_COND,     /*!< Condition number threshold used in the final RRQR */
  LHF_PIVOT,         /*!< Pivoting option */
  LHF_BETA, /*!< Option for the threshold used in preventing bad scaling factors
               from equlibriation in preprocessing. */
  LHF_ISSYMM,        /*!< Flag to tunning on symmetric/Hermitian input */
  LHF_NOPRE,         /*!< Option to turn on/off preprocessing */
  LHF_NZP_THRES,     /*!< Nonzero pattern symmetry threshold for symmetric
                        preprocessing */
  LHF_DENSE_THRES,   /*!< Dense size threshold for the Schur complement */
  LHF_NUMBER_PARAMS, /*!< Total number of parameters */
};

// verbose levels
enum {
  LHF_VERBOSE_NULL     = 0,
  LHF_VERBOSE_INFO     = 1,
  LHF_VERBOSE_PRE      = 2,
  LHF_VERBOSE_FAC      = 4,
  LHF_VERBOSE_PRE_TIME = 8,
  LHF_VERBOSE_MEM      = 16,
};

// reordering options
enum {
  LHF_REORDER_OFF = 0,
  LHF_REORDER_AUTO,
  LHF_REORDER_AMD,
  LHF_REORDER_RCM,
};

// rook pivoting options
enum {
  LHF_PIVOTING_OFF = 0,
  LHF_PIVOTING_ON,
  LHF_PIVOTING_AUTO,
};

/*!
 * @typedef LhfStatus
 * @brief Return status
 */
typedef enum LhfStatus {
  LHF_SUCCESS = 0,      /*!< Successful return */
  LHF_NULL_OBJ,         /*!< Calling functions on @a NULL objects */
  LHF_MISMATCHED_SIZES, /*!< Mismatched sizes */
  LHF_BAD_PREC,         /*!< Improper use of a preconditioner */
  LHF_HIFIR_ERROR,      /*!< Internal HIFIR error, call @ref lhfGetErrorMsg */
} LhfStatus;

/*!
 * @typedef LhfOperationType
 * @brief Operation tags for the @a apply functions
 */
typedef enum LhfOperationType {
  LHF_S = 0, /*!< Standard triangular solve */
  LHF_SH,    /*!< Hermitian/transpose solve */
  LHF_M,     /*!< Multilevel matrix-vector product */
  LHF_MH,    /*!< Hermitian/transpose matrix-vector product */
} LhfOperationType;

static const int LHF_DEFAULT_RANK = -2;
/*!< Default numerical rank in @a apply routines */

/*!
 * @}
 */

/*!
 * @typedef LhfdMatrixHdl
 * @brief Pointer to double precision sparse matrix in @a libhifir
 * @ingroup cdouble
 */
typedef struct LhfdMatrix* LhfdMatrixHdl;

/*!
 * @typedef LhfsMatrixHdl
 * @brief Pointer to single precision sparse matrix in @a libhifir
 * @ingroup csingle
 */
typedef struct LhfsMatrix* LhfsMatrixHdl;

/*!
 * @typedef LhfzMatrixHdl
 * @brief Pointer to double precision complex sparse matrix in @a libhifir
 * @ingroup ccomplexdouble
 */
typedef struct LhfzMatrix* LhfzMatrixHdl;

/*!
 * @typedef LhfcMatrixHdl
 * @brief Pointer to single precision complex sparse matrix in @a libhifir
 * @ingroup ccomplexsingle
 */
typedef struct LhfcMatrix* LhfcMatrixHdl;

/*!
 * @typedef LhfdHifHdl
 * @brief Pointer to double precision HIF structure in @a libhifir
 * @ingroup cdouble
 */
typedef struct LhfdHif* LhfdHifHdl;

/*!
 * @typedef LhfsHifHdl
 * @brief Pointer to single precision HIF structure in @a libhifir
 * @ingroup csingle
 */
typedef struct LhfsHif* LhfsHifHdl;

/*!
 * @typedef LhfzHifHdl
 * @brief Pointer to double precision complex HIF structure in @a libhifir
 * @ingroup ccomplexdouble
 */
typedef struct LhfzHif* LhfzHifHdl;

/*!
 * @typedef LhfcHifHdl
 * @brief Pointer to single precision complex HIF structure in @a libhifir
 * @ingroup ccomplexsingle
 */
typedef struct LhfcHif* LhfcHifHdl;

/*!
 * @addtogroup c
 * @{
 */

/*!
 * @brief Get versions of the C++ HIFIR package
 * @param[out] versions A length-three array storing global, major, and minor
 *                      versions
 */
void lhfGetVersions(int versions[]);

/*!
 * @brief Enable warning
 */
void lhfEnableWarning(void);

/*!
 * @brief Disable warning
 */
void lhfDisableWarning(void);

/*!
 * @brief Get global error message if the return status is @ref LHF_HIFIR_ERROR
 */
const char* lhfGetErrorMsg(void);

/*!
 * @brief Initialize parameters with default options
 * @param[out] params Parameter array of legnth @ref LHF_NUMBER_PARAMS
 * @sa lhfSetDroptol, lhfSetAlpha
 */
LhfStatus lhfSetDefaultParams(double params[]);

/*!
 * @brief Set unified droptol for both L and U factors
 * @param[in] droptol Drop tolerance value
 * @param[out] params Parameter array of legnth @ref LHF_NUMBER_PARAMS
 * @sa lhfSetDefaultParams
 */
LhfStatus lhfSetDroptol(const double droptol, double params[]);

/*!
 * @brief Set unified fill factors for both L and U factors
 * @param[in] alpha Fill factor value
 * @param[out] params Parameter array of legnth @ref LHF_NUMBER_PARAMS
 * @sa lhfSetDefaultParams
 */
LhfStatus lhfSetAlpha(const double alpha, double params[]);

/*!
 * @brief Set unified inverse-norm threshold for L, U, and D factors
 * @param[in] kappa Inverse-norm threshold value
 * @param[out] params Parameter array of legnth @ref LHF_NUMBER_PARAMS
 * @sa lhfSetDefaultParams
 */
LhfStatus lhfSetKappa(const double kappa, double params[]);

/*!
 * @}
 */

// Matrices

/*!
 * @addtogroup cdouble
 * @{
 */

/*!
 * @brief Create an instance of double-precision sparse matrix
 * @param[in] is_rowmajor If true, then we use CRS, otherwise CCS is assumed
 * @param[in] n Size of the squared matrix
 * @param[in] indptr Index pointer array
 * @param[in] indices Index list
 * @param[in] vals Numerical value array
 * @note The last three entries can be @a NULL, which will create an empty
 *       instance of sparse matrix.
 */
LhfdMatrixHdl lhfdCreateMatrix(const int is_rowmajor, const size_t n,
                               const LhfInt* indptr, const LhfInt* indices,
                               const double* vals);

/*!
 * @brief Destroy a double-precision matrix instance
 * @param[in,out] mat Matrix instance
 */
LhfStatus lhfdDestroyMatrix(LhfdMatrixHdl mat);

/*!
 * @brief Get the matrix size
 */
size_t lhfdGetMatrixSize(const LhfdMatrixHdl mat);

/*!
 * @brief Get the matrix number of nonzeros
 */
size_t lhfdGetMatrixNnz(const LhfdMatrixHdl mat);

/*!
 * @}
 */

/*!
 * @addtogroup csingle
 * @{
 */

/*!
 * @brief Create an instance of single-precision sparse matrix
 * @param[in] is_rowmajor If true, then we use CRS, otherwise CCS is assumed
 * @param[in] n Size of the squared matrix
 * @param[in] indptr Index pointer array
 * @param[in] indices Index list
 * @param[in] vals Numerical value array
 * @note The last three entries can be @a NULL, which will create an empty
 *       instance of sparse matrix.
 */
LhfsMatrixHdl lhfsCreateMatrix(const int is_rowmajor, const size_t n,
                               const LhfInt* indptr, const LhfInt* indices,
                               const float* vals);

/*!
 * @brief Destroy a single-precision matrix instance
 * @param[in,out] mat Matrix instance
 */
LhfStatus lhfsDestroyMatrix(LhfsMatrixHdl mat);

/*!
 * @brief Get the matrix size
 */
size_t lhfsGetMatrixSize(const LhfsMatrixHdl mat);

/*!
 * @brief Get the matrix number of nonzeros
 */
size_t lhfsGetMatrixNnz(const LhfsMatrixHdl mat);

/*!
 * @}
 */

/*!
 * @addtogroup ccomplexdouble
 * @{
 */

/*!
 * @brief Create an instance of double-precision complex sparse matrix
 * @param[in] is_rowmajor If true, then we use CRS, otherwise CCS is assumed
 * @param[in] n Size of the squared matrix
 * @param[in] indptr Index pointer array
 * @param[in] indices Index list
 * @param[in] vals Numerical value array
 * @note The last three entries can be @a NULL, which will create an empty
 *       instance of sparse matrix.
 */
LhfzMatrixHdl lhfzCreateMatrix(const int is_rowmajor, const size_t n,
                               const LhfInt* indptr, const LhfInt* indices,
                               const double _Complex* vals);

/*!
 * @brief Destroy a double-precision complex matrix instance
 * @param[in,out] mat Matrix instance
 */
LhfStatus lhfzDestroyMatrix(LhfzMatrixHdl mat);

/*!
 * @brief Get the matrix size
 */
size_t lhfzGetMatrixSize(const LhfzMatrixHdl mat);

/*!
 * @brief Get the matrix number of nonzeros
 */
size_t lhfzGetMatrixNnz(const LhfzMatrixHdl mat);

/*!
 * @}
 */

/*!
 * @addtogroup ccomplexsingle
 * @{
 */

/*!
 * @brief Create an instance of single-precision complex sparse matrix
 * @param[in] is_rowmajor If true, then we use CRS, otherwise CCS is assumed
 * @param[in] n Size of the squared matrix
 * @param[in] indptr Index pointer array
 * @param[in] indices Index list
 * @param[in] vals Numerical value array
 * @note The last three entries can be @a NULL, which will create an empty
 *       instance of sparse matrix.
 */
LhfcMatrixHdl lhfcCreateMatrix(const int is_rowmajor, const size_t n,
                               const LhfInt* indptr, const LhfInt* indices,
                               const float _Complex* vals);

/*!
 * @brief Destroy a single-precision complex matrix instance
 * @param[in,out] mat Matrix instance
 */
LhfStatus lhfcDestroyMatrix(LhfcMatrixHdl mat);

/*!
 * @brief Get the matrix size
 */
size_t lhfcGetMatrixSize(const LhfcMatrixHdl mat);

/*!
 * @brief Get the matrix number of nonzeros
 */
size_t lhfcGetMatrixNnz(const LhfcMatrixHdl mat);

/*!
 * @}
 */

/*!
 * @brief Wrap external data into a double-precision sparse matrix
 * @param[out] mat Matrix that holds the external data
 * @param[in] n Size the squared matrix
 * @param[in] indptr Index pointer array
 * @param[in] indices Index list
 * @param[in] vals Numerical value array
 * @ingroup cdouble
 */
LhfStatus lhfdWrapMatrix(LhfdMatrixHdl mat, const size_t n,
                         const LhfInt* indptr, const LhfInt* indices,
                         const double* vals);

/*!
 * @brief Wrap external data into a single-precision sparse matrix
 * @param[out] mat Matrix that holds the external data
 * @param[in] n Size the squared matrix
 * @param[in] indptr Index pointer array
 * @param[in] indices Index list
 * @param[in] vals Numerical value array
 * @ingroup csingle
 */
LhfStatus lhfsWrapMatrix(LhfsMatrixHdl mat, const size_t n,
                         const LhfInt* indptr, const LhfInt* indices,
                         const float* vals);

/*!
 * @brief Wrap external data into a double-precision complex sparse matrix
 * @param[out] mat Matrix that holds the external data
 * @param[in] n Size the squared matrix
 * @param[in] indptr Index pointer array
 * @param[in] indices Index list
 * @param[in] vals Numerical value array
 * @ingroup ccomplexdouble
 */
LhfStatus lhfzWrapMatrix(LhfzMatrixHdl mat, const size_t n,
                         const LhfInt* indptr, const LhfInt* indices,
                         const double _Complex* vals);

/*!
 * @brief Wrap external data into a single-precision complex sparse matrix
 * @param[out] mat Matrix that holds the external data
 * @param[in] n Size the squared matrix
 * @param[in] indptr Index pointer array
 * @param[in] indices Index list
 * @param[in] vals Numerical value array
 * @ingroup ccomplexsingle
 */
LhfStatus lhfcWrapMatrix(LhfcMatrixHdl mat, const size_t n,
                         const LhfInt* indptr, const LhfInt* indices,
                         const float _Complex* vals);

// HIF preconditioners

/*!
 * @addtogroup cdouble
 * @{
 */

/*!
 * @brief Create a double-precision HIF instance
 * @param[in] A Input coefficient matrix (used in iterative refinement)
 * @param[in] S Sparsifier input
 * @param[in] params Control parameters, see @ref lhfSetDefaultParams
 *
 * This function create an instance of double-precision HIF (see @ref hif::HIF)
 * preconditioner. Note that both @a A and @a S can be @a NULL, which then will
 * create an empty instance. If both operators are provided, then the
 * preconditioner will be factorized based on @a S. If only @a A is provided,
 * then HIF is computed on @a A. If only @a S is provided, then @a A is set to
 * be @a S besides computing the factorization on it. In addition, if @a params
 * is @a NULL, then the default parameters will be used.
 *
 * @sa lhfdDestroy
 */
LhfdHifHdl lhfdCreate(const LhfdMatrixHdl A, const LhfdMatrixHdl S,
                      const double params[]);

/*!
 * @brief Destroy a double-precision HIF instance
 * @param[out] hif HIF preconditioner
 * @sa lhfdCreate
 */
LhfStatus lhfdDestroy(LhfdHifHdl hif);

/*!
 * @brief Setup a double-precision HIF instance
 * @param[out] hif A HIF instance
 * @param[in] A Input coefficient matrix (used in iterative refinement)
 * @param[in] S Sparsifier input
 * @param[in] params Control parameters, see @ref lhfSetDefaultParams
 *
 * This function serves similar to @ref lhfdCreate, except that @a A and @a S
 * cannot be both @a NULL. This function is to defer the factorization from
 * construction of HIF preconditioners.
 *
 * @sa lhfdCreate
 */
LhfStatus lhfdSetup(LhfdHifHdl hif, const LhfdMatrixHdl A,
                    const LhfdMatrixHdl S, const double params[]);

/*!
 * @brief Update the @a A matrix in HIF
 * @param[out] hif A HIF instance
 * @param[in] A A new matrix
 * @note This function will not call factorization on @a A.
 * @sa lhfdRefactorize
 */
LhfStatus lhfdUpdate(LhfdHifHdl hif, const LhfdMatrixHdl A);

/*!
 * @brief Refactorize a HIF preconditioner
 * @param[out] hif A HIF instance
 * @param[in] S A new sparsifier
 * @param[in] params Control parameters
 * @note This function will not update @a A inside @a hif
 * @note If @a params is @a NULL, then the default parameters will be used.
 * @sa lhfdUpdate
 */
LhfStatus lhfdRefactorize(LhfdHifHdl hif, const LhfdMatrixHdl S,
                          const double params[]);

/*!
 * @brief Apply a preconditioner with a certian operation mode
 * @param[in] hif A HIF instance
 * @param[in] op Operation tag
 * @param[in] b The RHS vector
 * @param[in] nirs Number of iterative refinements
 * @param[in] betas Relative residual norm bounds for IR with op=solve
 * @param[in] rank Numerical rank
 * @param[out] x Computed solution vector
 * @param[out] ir_status (optional) If IR and @a betas is provided, then this
 *                        records the number of actual IR ( @a ir_status[0] )
 *                        and the IR status ( @a ir_status[1] ).
 *
 * This is the core function of HIF. A HIF instance can be applied in four
 * different modes, namely triangular solve and its Hermitian/tranpose variant,
 * as well as matrix-vector multiplication and its Hermitian/tranpose variant.
 * For the triangular solve modes, @a nirs can be greater than one, which means
 * iterative refinement is enabled. With IR enabled, if @a betas is provided,
 * i.e., not @a NULL, then we will iterative until (1) res<=betas[0], (2)
 * res>betas[1], or (3) it=nirs, and the actual iteratoin steps and return
 * status are stored in @a ir_status if provided (i.e., not @a NULL ). On the
 * other side, if @a betas=NULL, then the a fixed amount of IR ( @a nirs ) are
 * performed and @a ir_status is not accessed. Regarding @a rank, in general, it
 * should use @a LHF_DEFAULT_RANK.
 *
 * @sa lhfdSolve
 */
LhfStatus lhfdApply(const LhfdHifHdl hif, const LhfOperationType op,
                    const double* b, const int nirs, const double* betas,
                    const int rank, double* x, int* ir_status);

/*!
 * @brief Triangular solve
 *
 * For the sake of convenience, we provide the following routine as it is the
 * most commonly used interface. The following routine is equivalent to calling
 * @ref lhfdApply with @a op=LHF_S, @a nirs=1, and @a rank=LHF_DEFAULT_RANK.
 *
 * @sa lhfdApply
 */
LhfStatus lhfdSolve(const LhfdHifHdl hif, const double* b, double* x);

/*!
 * @brief Get the statistics of a computed HIF instance
 * @param[in] hif A HIF instance
 * @param[out] stats A length-9 array stores certain useful information
 *
 * Regarding the output @a stats:
 *  - stats[0]: Number of nonzeros of a preconditioner
 *  - stats[1]: Total deferals
 *  - stats[2]: Dynamic deferals (status[1]-status[2] are static deferals)
 *  - stats[3]: Total droppings
 *  - stats[4]: Droppings due to scalability-oriented strategy
 *  - stats[5]: Number of levels
 *  - stats[6]: Numerical rank of the whole preconditioner
 *  - stats[7]: The numerical rank of the final Schur complement
 *  - stats[8]: The size of the final Schur complement (status[7]<=status[8])
 */
LhfStatus lhfdGetStatus(const LhfdHifHdl hif, size_t stats[]);

/*!
 * @brief Get number of nonzeros of a HIF preconditioner
 * @sa lhfdGetStatus
 */
size_t lhfdGetNnz(const LhfdHifHdl hif);

/*!
 * @brief Get number of levels
 * @sa lhfdGetStatus
 */
size_t lhfdGetLevels(const LhfdHifHdl hif);

/*!
 * @brief Get the Schur complement size
 * @sa lhfdGetStatus
 */
size_t lhfdGetSchurSize(const LhfdHifHdl hif);

/*!
 * @brief Get the Schur complement rank
 * @sa lhfdGetStatus
 */
size_t lhfdGetSchurRank(const LhfdHifHdl hif);

/*!
 * @}
 */

/*!
 * @addtogroup csingle
 * @{
 */

/*!
 * @brief Create a single-precision HIF instance
 * @param[in] A Input coefficient matrix (used in iterative refinement)
 * @param[in] S Sparsifier input
 * @param[in] params Control parameters, see @ref lhfSetDefaultParams
 *
 * This function create an instance of single-precision HIF (see @ref hif::HIF)
 * preconditioner. Note that both @a A and @a S can be @a NULL, which then will
 * create an empty instance. If both operators are provided, then the
 * preconditioner will be factorized based on @a S. If only @a A is provided,
 * then HIF is computed on @a A. If only @a S is provided, then @a A is set to
 * be @a S besides computing the factorization on it. In addition, if @a params
 * is @a NULL, then the default parameters will be used.
 *
 * @sa lhfsDestroy
 */
LhfsHifHdl lhfsCreate(const LhfsMatrixHdl A, const LhfsMatrixHdl S,
                      const double params[]);

/*!
 * @brief Destroy a single-precision HIF instance
 * @param[out] hif HIF preconditioner
 * @sa lhfsCreate
 */
LhfStatus lhfsDestroy(LhfsHifHdl hif);

/*!
 * @brief Setup a single-precision HIF instance
 * @param[out] hif A HIF instance
 * @param[in] A Input coefficient matrix (used in iterative refinement)
 * @param[in] S Sparsifier input
 * @param[in] params Control parameters, see @ref lhfSetDefaultParams
 *
 * This function serves similar to @ref lhfsCreate, except that @a A and @a S
 * cannot be both @a NULL. This function is to defer the factorization from
 * construction of HIF preconditioners.
 *
 * @sa lhfsCreate
 */
LhfStatus lhfsSetup(LhfsHifHdl hif, const LhfsMatrixHdl A,
                    const LhfsMatrixHdl S, const double params[]);

/*!
 * @brief Update the @a A matrix in HIF
 * @param[out] hif A HIF instance
 * @param[in] A A new matrix
 * @note This function will not call factorization on @a A.
 * @sa lhfsRefactorize
 */
LhfStatus lhfsUpdate(LhfsHifHdl hif, const LhfsMatrixHdl A);

/*!
 * @brief Refactorize a HIF preconditioner
 * @param[out] hif A HIF instance
 * @param[in] S A new sparsifier
 * @param[in] params Control parameters
 * @note This function will not update @a A inside @a hif
 * @note If @a params is @a NULL, then the default parameters will be used.
 * @sa lhfsUpdate
 */
LhfStatus lhfsRefactorize(LhfsHifHdl hif, const LhfsMatrixHdl S,
                          const double params[]);

/*!
 * @brief Apply a preconditioner with a certian operation mode
 * @param[in] hif A HIF instance
 * @param[in] op Operation tag
 * @param[in] b The RHS vector
 * @param[in] nirs Number of iterative refinements
 * @param[in] betas Relative residual norm bounds for IR with op=solve
 * @param[in] rank Numerical rank
 * @param[out] x Computed solution vector
 * @param[out] ir_status (optional) If IR and @a betas is provided, then this
 *                        records the number of actual IR ( @a ir_status[0] )
 *                        and the IR status ( @a ir_status[1] ).
 *
 * This is the core function of HIF. A HIF instance can be applied in four
 * different modes, namely triangular solve and its Hermitian/tranpose variant,
 * as well as matrix-vector multiplication and its Hermitian/tranpose variant.
 * For the triangular solve modes, @a nirs can be greater than one, which means
 * iterative refinement is enabled. With IR enabled, if @a betas is provided,
 * i.e., not @a NULL, then we will iterative until (1) res<=betas[0], (2)
 * res>betas[1], or (3) it=nirs, and the actual iteratoin steps and return
 * status are stored in @a ir_status if provided (i.e., not @a NULL ). On the
 * other side, if @a betas=NULL, then the a fixed amount of IR ( @a nirs ) are
 * performed and @a ir_status is not accessed. Regarding @a rank, in general, it
 * should use @a LHF_DEFAULT_RANK.
 *
 * @sa lhfsSolve
 */
LhfStatus lhfsApply(const LhfsHifHdl hif, const LhfOperationType op,
                    const float* b, const int nirs, const double* betas,
                    const int rank, float* x, int* ir_status);

/*!
 * @brief Triangular solve
 *
 * For the sake of convenience, we provide the following routine as it is the
 * most commonly used interface. The following routine is equivalent to calling
 * @ref lhfsApply with @a op=LHF_S, @a nirs=1, and @a rank=LHF_DEFAULT_RANK.
 *
 * @sa lhfsApply
 */
LhfStatus lhfsSolve(const LhfsHifHdl hif, const float* b, float* x);

/*!
 * @brief Get the statistics of a computed HIF instance
 * @param[in] hif A HIF instance
 * @param[out] stats A length-9 array stores certain useful information
 *
 * Regarding the output @a stats:
 *  - stats[0]: Number of nonzeros of a preconditioner
 *  - stats[1]: Total deferals
 *  - stats[2]: Dynamic deferals (status[1]-status[2] are static deferals)
 *  - stats[3]: Total droppings
 *  - stats[4]: Droppings due to scalability-oriented strategy
 *  - stats[5]: Number of levels
 *  - stats[6]: Numerical rank of the whole preconditioner
 *  - stats[7]: The numerical rank of the final Schur complement
 *  - stats[8]: The size of the final Schur complement (status[7]<=status[8])
 */
LhfStatus lhfsGetStatus(const LhfsHifHdl hif, size_t stats[]);

/*!
 * @brief Get number of nonzeros of a HIF preconditioner
 * @sa lhfsGetStatus
 */
size_t lhfsGetNnz(const LhfsHifHdl hif);

/*!
 * @brief Get number of levels
 * @sa lhfsGetStatus
 */
size_t lhfsGetLevels(const LhfsHifHdl hif);

/*!
 * @brief Get the Schur complement size
 * @sa lhfsGetStatus
 */
size_t lhfsGetSchurSize(const LhfsHifHdl hif);

/*!
 * @brief Get the Schur complement rank
 * @sa lhfsGetStatus
 */
size_t lhfsGetSchurRank(const LhfsHifHdl hif);

/*!
 * @}
 */

/*!
 * @addtogroup ccomplexdouble
 * @{
 */

/*!
 * @brief Create a double-precision complex HIF instance
 * @param[in] A Input coefficient matrix (used in iterative refinement)
 * @param[in] S Sparsifier input
 * @param[in] params Control parameters, see @ref lhfSetDefaultParams
 *
 * This function create an instance of double-precision HIF (see @ref hif::HIF)
 * preconditioner. Note that both @a A and @a S can be @a NULL, which then will
 * create an empty instance. If both operators are provided, then the
 * preconditioner will be factorized based on @a S. If only @a A is provided,
 * then HIF is computed on @a A. If only @a S is provided, then @a A is set to
 * be @a S besides computing the factorization on it. In addition, if @a params
 * is @a NULL, then the default parameters will be used.
 *
 * @sa lhfdDestroy
 */
LhfzHifHdl lhfzCreate(const LhfzMatrixHdl A, const LhfzMatrixHdl S,
                      const double params[]);

/*!
 * @brief Destroy a double-precision complex HIF instance
 * @param[out] hif HIF preconditioner
 * @sa lhfzCreate
 */
LhfStatus lhfzDestroy(LhfzHifHdl hif);

/*!
 * @brief Setup a double-precision complex HIF instance
 * @param[out] hif A HIF instance
 * @param[in] A Input coefficient matrix (used in iterative refinement)
 * @param[in] S Sparsifier input
 * @param[in] params Control parameters, see @ref lhfSetDefaultParams
 *
 * This function serves similar to @ref lhfzCreate, except that @a A and @a S
 * cannot be both @a NULL. This function is to defer the factorization from
 * construction of HIF preconditioners.
 *
 * @sa lhfzCreate
 */
LhfStatus lhfzSetup(LhfzHifHdl hif, const LhfzMatrixHdl A,
                    const LhfzMatrixHdl S, const double params[]);

/*!
 * @brief Update the @a A matrix in HIF
 * @param[out] hif A HIF instance
 * @param[in] A A new matrix
 * @note This function will not call factorization on @a A.
 * @sa lhfzRefactorize
 */
LhfStatus lhfzUpdate(LhfzHifHdl hif, const LhfzMatrixHdl A);

/*!
 * @brief Refactorize a HIF preconditioner
 * @param[out] hif A HIF instance
 * @param[in] S A new sparsifier
 * @param[in] params Control parameters
 * @note This function will not update @a A inside @a hif
 * @note If @a params is @a NULL, then the default parameters will be used.
 * @sa lhfzUpdate
 */
LhfStatus lhfzRefactorize(LhfzHifHdl hif, const LhfzMatrixHdl S,
                          const double params[]);

/*!
 * @brief Apply a preconditioner with a certian operation mode
 * @param[in] hif A HIF instance
 * @param[in] op Operation tag
 * @param[in] b The RHS vector
 * @param[in] nirs Number of iterative refinements
 * @param[in] betas Relative residual norm bounds for IR with op=solve
 * @param[in] rank Numerical rank
 * @param[out] x Computed solution vector
 * @param[out] ir_status (optional) If IR and @a betas is provided, then this
 *                        records the number of actual IR ( @a ir_status[0] )
 *                        and the IR status ( @a ir_status[1] ).
 *
 * This is the core function of HIF. A HIF instance can be applied in four
 * different modes, namely triangular solve and its Hermitian/tranpose variant,
 * as well as matrix-vector multiplication and its Hermitian/tranpose variant.
 * For the triangular solve modes, @a nirs can be greater than one, which means
 * iterative refinement is enabled. With IR enabled, if @a betas is provided,
 * i.e., not @a NULL, then we will iterative until (1) res<=betas[0], (2)
 * res>betas[1], or (3) it=nirs, and the actual iteratoin steps and return
 * status are stored in @a ir_status if provided (i.e., not @a NULL ). On the
 * other side, if @a betas=NULL, then the a fixed amount of IR ( @a nirs ) are
 * performed and @a ir_status is not accessed. Regarding @a rank, in general, it
 * should use @a LHF_DEFAULT_RANK.
 *
 * @sa lhfzSolve
 */
LhfStatus lhfzApply(const LhfzHifHdl hif, const LhfOperationType op,
                    const double _Complex* b, const int nirs,
                    const double* betas, const int rank, double _Complex* x,
                    int* ir_status);

/*!
 * @brief Triangular solve
 *
 * For the sake of convenience, we provide the following routine as it is the
 * most commonly used interface. The following routine is equivalent to calling
 * @ref lhfzApply with @a op=LHF_S, @a nirs=1, and @a rank=LHF_DEFAULT_RANK.
 *
 * @sa lhfzApply
 */
LhfStatus lhfzSolve(const LhfzHifHdl hif, const double _Complex* b,
                    double _Complex* x);

/*!
 * @brief Get the statistics of a computed HIF instance
 * @param[in] hif A HIF instance
 * @param[out] stats A length-9 array stores certain useful information
 *
 * Regarding the output @a stats:
 *  - stats[0]: Number of nonzeros of a preconditioner
 *  - stats[1]: Total deferals
 *  - stats[2]: Dynamic deferals (status[1]-status[2] are static deferals)
 *  - stats[3]: Total droppings
 *  - stats[4]: Droppings due to scalability-oriented strategy
 *  - stats[5]: Number of levels
 *  - stats[6]: Numerical rank of the whole preconditioner
 *  - stats[7]: The numerical rank of the final Schur complement
 *  - stats[8]: The size of the final Schur complement (status[7]<=status[8])
 */
LhfStatus lhfzGetStatus(const LhfzHifHdl hif, size_t stats[]);

/*!
 * @brief Get number of nonzeros of a HIF preconditioner
 * @sa lhfzGetStatus
 */
size_t lhfzGetNnz(const LhfzHifHdl hif);

/*!
 * @brief Get number of levels
 * @sa lhfzGetStatus
 */
size_t lhfzGetLevels(const LhfzHifHdl hif);

/*!
 * @brief Get the Schur complement size
 * @sa lhfzGetStatus
 */
size_t lhfzGetSchurSize(const LhfzHifHdl hif);

/*!
 * @brief Get the Schur complement rank
 * @sa lhfzGetStatus
 */
size_t lhfzGetSchurRank(const LhfzHifHdl hif);

/*!
 * @}
 */

/*!
 * @addtogroup ccomplexsingle
 * @{
 */

/*!
 * @brief Create a single-precision complex HIF instance
 * @param[in] A Input coefficient matrix (used in iterative refinement)
 * @param[in] S Sparsifier input
 * @param[in] params Control parameters, see @ref lhfSetDefaultParams
 *
 * This function create an instance of single-precision HIF (see @ref hif::HIF)
 * preconditioner. Note that both @a A and @a S can be @a NULL, which then will
 * create an empty instance. If both operators are provided, then the
 * preconditioner will be factorized based on @a S. If only @a A is provided,
 * then HIF is computed on @a A. If only @a S is provided, then @a A is set to
 * be @a S besides computing the factorization on it. In addition, if @a params
 * is @a NULL, then the default parameters will be used.
 *
 * @sa lhfcDestroy
 */
LhfcHifHdl lhfcCreate(const LhfcMatrixHdl A, const LhfcMatrixHdl S,
                      const double params[]);

/*!
 * @brief Destroy a single-precision complex HIF instance
 * @param[out] hif HIF preconditioner
 * @sa lhfcCreate
 */
LhfStatus lhfcDestroy(LhfcHifHdl hif);

/*!
 * @brief Setup a single-precision complex HIF instance
 * @param[out] hif A HIF instance
 * @param[in] A Input coefficient matrix (used in iterative refinement)
 * @param[in] S Sparsifier input
 * @param[in] params Control parameters, see @ref lhfSetDefaultParams
 *
 * This function serves similar to @ref lhfcCreate, except that @a A and @a S
 * cannot be both @a NULL. This function is to defer the factorization from
 * construction of HIF preconditioners.
 *
 * @sa lhfcCreate
 */
LhfStatus lhfcSetup(LhfcHifHdl hif, const LhfcMatrixHdl A,
                    const LhfcMatrixHdl S, const double params[]);

/*!
 * @brief Update the @a A matrix in HIF
 * @param[out] hif A HIF instance
 * @param[in] A A new matrix
 * @note This function will not call factorization on @a A.
 * @sa lhfcRefactorize
 */
LhfStatus lhfcUpdate(LhfcHifHdl hif, const LhfcMatrixHdl A);

/*!
 * @brief Refactorize a HIF preconditioner
 * @param[out] hif A HIF instance
 * @param[in] S A new sparsifier
 * @param[in] params Control parameters
 * @note This function will not update @a A inside @a hif
 * @note If @a params is @a NULL, then the default parameters will be used.
 * @sa lhfcUpdate
 */
LhfStatus lhfcRefactorize(LhfcHifHdl hif, const LhfcMatrixHdl S,
                          const double params[]);

/*!
 * @brief Apply a preconditioner with a certian operation mode
 * @param[in] hif A HIF instance
 * @param[in] op Operation tag
 * @param[in] b The RHS vector
 * @param[in] nirs Number of iterative refinements
 * @param[in] betas Relative residual norm bounds for IR with op=solve
 * @param[in] rank Numerical rank
 * @param[out] x Computed solution vector
 * @param[out] ir_status (optional) If IR and @a betas is provided, then this
 *                        records the number of actual IR ( @a ir_status[0] )
 *                        and the IR status ( @a ir_status[1] ).
 *
 * This is the core function of HIF. A HIF instance can be applied in four
 * different modes, namely triangular solve and its Hermitian/tranpose variant,
 * as well as matrix-vector multiplication and its Hermitian/tranpose variant.
 * For the triangular solve modes, @a nirs can be greater than one, which means
 * iterative refinement is enabled. With IR enabled, if @a betas is provided,
 * i.e., not @a NULL, then we will iterative until (1) res<=betas[0], (2)
 * res>betas[1], or (3) it=nirs, and the actual iteratoin steps and return
 * status are stored in @a ir_status if provided (i.e., not @a NULL ). On the
 * other side, if @a betas=NULL, then the a fixed amount of IR ( @a nirs ) are
 * performed and @a ir_status is not accessed. Regarding @a rank, in general, it
 * should use @a LHF_DEFAULT_RANK.
 *
 * @sa lhfcSolve
 */
LhfStatus lhfcApply(const LhfcHifHdl hif, const LhfOperationType op,
                    const float _Complex* b, const int nirs,
                    const double* betas, const int rank, float _Complex* x,
                    int* ir_status);

/*!
 * @brief Triangular solve
 *
 * For the sake of convenience, we provide the following routine as it is the
 * most commonly used interface. The following routine is equivalent to calling
 * @ref lhfcApply with @a op=LHF_S, @a nirs=1, and @a rank=LHF_DEFAULT_RANK.
 *
 * @sa lhfcApply
 */
LhfStatus lhfcSolve(const LhfcHifHdl hif, const float _Complex* b,
                    float _Complex* x);

/*!
 * @brief Get the statistics of a computed HIF instance
 * @param[in] hif A HIF instance
 * @param[out] stats A length-9 array stores certain useful information
 *
 * Regarding the output @a stats:
 *  - stats[0]: Number of nonzeros of a preconditioner
 *  - stats[1]: Total deferals
 *  - stats[2]: Dynamic deferals (status[1]-status[2] are static deferals)
 *  - stats[3]: Total droppings
 *  - stats[4]: Droppings due to scalability-oriented strategy
 *  - stats[5]: Number of levels
 *  - stats[6]: Numerical rank of the whole preconditioner
 *  - stats[7]: The numerical rank of the final Schur complement
 *  - stats[8]: The size of the final Schur complement (status[7]<=status[8])
 */
LhfStatus lhfcGetStatus(const LhfcHifHdl hif, size_t stats[]);

/*!
 * @brief Get number of nonzeros of a HIF preconditioner
 * @sa lhfcGetStatus
 */
size_t lhfcGetNnz(const LhfcHifHdl hif);

/*!
 * @brief Get number of levels
 * @sa lhfcGetStatus
 */
size_t lhfcGetLevels(const LhfcHifHdl hif);

/*!
 * @brief Get the Schur complement size
 * @sa lhfcGetStatus
 */
size_t lhfcGetSchurSize(const LhfcHifHdl hif);

/*!
 * @brief Get the Schur complement rank
 * @sa lhfcGetStatus
 */
size_t lhfcGetSchurRank(const LhfcHifHdl hif);

/*!
 * @}
 */

/*!
 * @addtogroup cmixed
 * @{
 */

/*!
 * @brief Update the @a A matrix in HIF (double precision)
 * @param[out] hif A HIF instance (single-precision)
 * @param[in] A A new matrix (double-precision)
 * @note This function will not call factorization on @a A.
 * @note This function will update the double-precision operator in @a hif.
 * @sa lhfsdApply
 */
LhfStatus lhfsdUpdate(LhfsHifHdl hif, LhfdMatrixHdl A);

/*!
 * @brief Apply a single preconditioner with a certian operation mode to obtain
 *        double solutions
 * @param[in] hif A single-precision HIF instance
 * @param[in] op Operation tag
 * @param[in] b The RHS vector
 * @param[in] nirs Number of iterative refinements
 * @param[in] betas Relative residual norm bounds for IR with op=solve
 * @param[in] rank Numerical rank
 * @param[out] x Computed solution vector
 * @param[out] ir_status (optional) If IR and @a betas is provided, then this
 *                        records the number of actual IR ( @a ir_status[0] )
 *                        and the IR status ( @a ir_status[1] ).
 *
 * This is the core function of HIF. A HIF instance can be applied in four
 * different modes, namely triangular solve and its Hermitian/tranpose variant,
 * as well as matrix-vector multiplication and its Hermitian/tranpose variant.
 * For the triangular solve modes, @a nirs can be greater than one, which means
 * iterative refinement is enabled. With IR enabled, if @a betas is provided,
 * i.e., not @a NULL, then we will iterative until (1) res<=betas[0], (2)
 * res>betas[1], or (3) it=nirs, and the actual iteratoin steps and return
 * status are stored in @a ir_status if provided (i.e., not @a NULL ). On the
 * other side, if @a betas=NULL, then the a fixed amount of IR ( @a nirs ) are
 * performed and @a ir_status is not accessed. Regarding @a rank, in general, it
 * should use @a LHF_DEFAULT_RANK.
 *
 * In addition, this function is for mixed-precision computation, in that a
 * single-precision HIF is used to obtain double-precision solutions. Note that
 * if iterative refinement is on (i.e., @a nirs>1 and op=solve), then @a A must
 * be passed in separately via @ref lhfsdUpdate, because we need a
 * double-precision matrix for iterations.
 *
 * @sa lhfsdSolve, lhfdApply, lhfsApply, lhfsdUpdate
 */
LhfStatus lhfsdApply(const LhfsHifHdl hif, const LhfOperationType op,
                     const double* b, const int nirs, const double* betas,
                     const int rank, double* x, int* ir_status);

/*!
 * @brief Triangular solve with midex double- and single-precisoin
 *
 * For the sake of convenience, we provide the following routine as it is the
 * most commonly used interface. The following routine is equivalent to calling
 * @ref lhfsdApply with @a op=LHF_S, @a nirs=1, and @a rank=LHF_DEFAULT_RANK.
 *
 * @sa lhfsdApply
 */
LhfStatus lhfsdSolve(const LhfsHifHdl hif, const double* b, double* x);

/*!
 * @brief Update the @a A matrix in HIF (complex double precision)
 * @param[out] hif A HIF instance (complex single-precision)
 * @param[in] A A new matrix (complex double-precision)
 * @note This function will not call factorization on @a A.
 * @note This function will update the double-precision operator in @a hif.
 * @sa lhfczApply
 */
LhfStatus lhfczUpdate(LhfcHifHdl hif, LhfzMatrixHdl A);

/*!
 * @brief Apply a single preconditioner with a certian operation mode to obtain
 *        double solutions (complex arithmetic)
 * @param[in] hif A complex single-precision HIF instance
 * @param[in] op Operation tag
 * @param[in] b The RHS vector
 * @param[in] nirs Number of iterative refinements
 * @param[in] betas Relative residual norm bounds for IR with op=solve
 * @param[in] rank Numerical rank
 * @param[out] x Computed solution vector
 * @param[out] ir_status (optional) If IR and @a betas is provided, then this
 *                        records the number of actual IR ( @a ir_status[0] )
 *                        and the IR status ( @a ir_status[1] ).
 *
 * This is the core function of HIF. A HIF instance can be applied in four
 * different modes, namely triangular solve and its Hermitian/tranpose variant,
 * as well as matrix-vector multiplication and its Hermitian/tranpose variant.
 * For the triangular solve modes, @a nirs can be greater than one, which means
 * iterative refinement is enabled. With IR enabled, if @a betas is provided,
 * i.e., not @a NULL, then we will iterative until (1) res<=betas[0], (2)
 * res>betas[1], or (3) it=nirs, and the actual iteratoin steps and return
 * status are stored in @a ir_status if provided (i.e., not @a NULL ). On the
 * other side, if @a betas=NULL, then the a fixed amount of IR ( @a nirs ) are
 * performed and @a ir_status is not accessed. Regarding @a rank, in general, it
 * should use @a LHF_DEFAULT_RANK.
 *
 * In addition, this function is for mixed-precision computation, in that a
 * single-precision HIF is used to obtain double-precision solutions. Note that
 * if iterative refinement is on (i.e., @a nirs>1 and op=solve), then @a A must
 * be passed in separately via @ref lhfczUpdate, because we need a
 * double-precision matrix for iterations.
 *
 * @sa lhfczSolve, lhfzApply, lhfcApply, lhfczUpdate
 */
LhfStatus lhfczApply(const LhfcHifHdl hif, const LhfOperationType op,
                     const double _Complex* b, const int nirs,
                     const double* betas, const int rank, double _Complex* x,
                     int* ir_status);

/*!
 * @brief Triangular solve with midex double- and single-precisoin (complex)
 *
 * For the sake of convenience, we provide the following routine as it is the
 * most commonly used interface. The following routine is equivalent to calling
 * @ref lhfczApply with @a op=LHF_S, @a nirs=1, and @a rank=LHF_DEFAULT_RANK.
 *
 * @sa lhfczApply
 */
LhfStatus lhfczSolve(const LhfcHifHdl hif, const double _Complex* b,
                     double _Complex* x);

/*!
 * @}
 */

#ifdef __cplusplus
}
#endif

#endif /* _LIBHIFIR_H */
