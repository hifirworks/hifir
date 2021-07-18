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
  LHF_NULL_MAT,         /*!< Calling functions on @a NULL matrices */
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
 * @struct LhfdMatrixStruct
 * @brief Double precision sparse matrix in @a libhifir
 * @ingroup cdouble
 */
struct LhfdMatrixStruct;
typedef struct LhfdMatrixStruct* LhfdMatrix; /*!< Pointer of sparse matrix */

/*!
 * @struct LhfsMatrixStruct
 * @brief Single precision sparse matrix in @a libhifir
 * @ingroup csingle
 */
struct LhfsMatrixStruct;
typedef struct LhfsMatrixStruct* LhfsMatrix; /*!< Pointer of sparse matrix */

/*!
 * @struct LhfzMatrixStruct
 * @brief Double precision complex sparse matrix in @a libhifir
 * @ingroup ccomplexdouble
 */
struct LhfzMatrixStruct;
typedef struct LhfzMatrixStruct* LhfzMatrix; /*!< Pointer of sparse matrix */

/*!
 * @struct LhfcMatrixStruct
 * @brief Single precision complex sparse matrix in @a libhifir
 * @ingroup ccomplexsingle
 */
struct LhfcMatrixStruct;
typedef struct LhfcMatrixStruct* LhfcMatrix; /*!< Pointer of sparse matrix */

/*!
 * @struct LhfdHIFStruct
 * @brief Double precision HIF structure in @a libhifir
 * @ingroup cdouble
 */
struct LhfdHIFStruct;
typedef struct LhfdHIFStruct* LhfdHIF; /*!< Pointer of HIF */

/*!
 * @struct LhfsHIFStruct
 * @brief Single precision HIF structure in @a libhifir
 * @ingroup csingle
 */
struct LhfsHIFStruct;
typedef struct LhfsHIFStruct* LhfsHIF; /*!< Pointer of HIF */

/*!
 * @struct LhfzHIFStruct
 * @brief Double precision complex HIF structure in @a libhifir
 * @ingroup ccomplexdouble
 */
struct LhfzHIFStruct;
typedef struct LhfzHIFStruct* LhfzHIF; /*!< Pointer of HIF */

/*!
 * @struct LhfcHIFStruct
 * @brief Single precision complex HIF structure in @a libhifir
 * @ingroup ccomplexsingle
 */
struct LhfcHIFStruct;
typedef struct LhfcHIFStruct* LhfcHIF; /*!< Pointer of HIF */

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
 * @brief Get global error message if the return status is @ref
 */

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
 * @brief Create an instance of double-precision sparse matrix
 * @param[in] is_rowmajor If true, then we use CRS, otherwise CCS is assumed
 * @param[in] n Size of the squared matrix
 * @param[in] indptr Index pointer array
 * @param[in] indices Index list
 * @param[in] vals Numerical value array
 * @note The last three entries can be @a NULL, which will create an empty
 *       instance of sparse matrix.
 * @ingroup cdouble
 */
LhfdMatrix lhfdCreateMatrix(const int is_rowmajor, const size_t n,
                            const LhfInt* indptr, const LhfInt* indices,
                            const double* vals);

/*!
 * @brief Destroy a double-precision matrix instance
 * @param[in,out] mat Matrix instance
 * @ingroup cdouble
 */
LhfStatus lhfdDestroyMatrix(LhfdMatrix mat);

/*!
 * @brief Get the matrix size
 */
size_t lhfdGetMatrixSize(const LhfdMatrix mat);

/*!
 * @brief Get the matrix number of nonzeros
 */
size_t lhfdGetMatrixNnz(const LhfdMatrix mat);

/*!
 * @brief Create an instance of single-precision sparse matrix
 * @param[in] is_rowmajor If true, then we use CRS, otherwise CCS is assumed
 * @param[in] n Size of the squared matrix
 * @param[in] indptr Index pointer array
 * @param[in] indices Index list
 * @param[in] vals Numerical value array
 * @note The last three entries can be @a NULL, which will create an empty
 *       instance of sparse matrix.
 * @ingroup csingle
 */
LhfsMatrix lhfsCreateMatrix(const int is_rowmajor, const size_t n,
                            const LhfInt* indptr, const LhfInt* indices,
                            const float* vals);

/*!
 * @brief Destroy a single-precision matrix instance
 * @param[in,out] mat Matrix instance
 * @ingroup csingle
 */
LhfStatus lhfsDestroyMatrix(LhfsMatrix mat);

/*!
 * @brief Get the matrix size
 */
size_t lhfsGetMatrixSize(const LhfsMatrix mat);

/*!
 * @brief Get the matrix number of nonzeros
 */
size_t lhfsGetMatrixNnz(const LhfsMatrix mat);

/*!
 * @brief Create an instance of double-precision complex sparse matrix
 * @param[in] is_rowmajor If true, then we use CRS, otherwise CCS is assumed
 * @param[in] n Size of the squared matrix
 * @param[in] indptr Index pointer array
 * @param[in] indices Index list
 * @param[in] vals Numerical value array
 * @note The last three entries can be @a NULL, which will create an empty
 *       instance of sparse matrix.
 * @ingroup ccomplexdouble
 */
LhfzMatrix lhfzCreateMatrix(const int is_rowmajor, const size_t n,
                            const LhfInt* indptr, const LhfInt* indices,
                            const void* vals);

/*!
 * @brief Destroy a double-precision complex matrix instance
 * @param[in,out] mat Matrix instance
 * @ingroup ccomplexdouble
 */
LhfStatus lhfzDestroyMatrix(LhfzMatrix mat);

/*!
 * @brief Get the matrix size
 */
size_t lhfzGetMatrixSize(const LhfzMatrix mat);

/*!
 * @brief Get the matrix number of nonzeros
 */
size_t lhfzGetMatrixNnz(const LhfzMatrix mat);

/*!
 * @brief Create an instance of single-precision complex sparse matrix
 * @param[in] is_rowmajor If true, then we use CRS, otherwise CCS is assumed
 * @param[in] n Size of the squared matrix
 * @param[in] indptr Index pointer array
 * @param[in] indices Index list
 * @param[in] vals Numerical value array
 * @note The last three entries can be @a NULL, which will create an empty
 *       instance of sparse matrix.
 * @ingroup ccomplexsingle
 */
LhfcMatrix lhfcCreateMatrix(const int is_rowmajor, const size_t n,
                            const LhfInt* indptr, const LhfInt* indices,
                            const void* vals);

/*!
 * @brief Destroy a single-precision complex matrix instance
 * @param[in,out] mat Matrix instance
 * @ingroup ccomplexsingle
 */
LhfStatus lhfcDestroyMatrix(LhfcMatrix mat);

/*!
 * @brief Get the matrix size
 */
size_t lhfcGetMatrixSize(const LhfdMatrix mat);

/*!
 * @brief Get the matrix number of nonzeros
 */
size_t lhfcGetMatrixNnz(const LhfdMatrix mat);

/*!
 * @brief Wrap external data into a double-precision sparse matrix
 * @param[out] mat Matrix that holds the external data
 * @param[in] n Size the squared matrix
 * @param[in] indptr Index pointer array
 * @param[in] indices Index list
 * @param[in] vals Numerical value array
 * @ingroup cdouble
 */
LhfStatus lhfdWrapMatrix(LhfdMatrix mat, const size_t n, const LhfInt* indptr,
                         const LhfInt* indices, const double* vals);

/*!
 * @brief Wrap external data into a single-precision sparse matrix
 * @param[out] mat Matrix that holds the external data
 * @param[in] n Size the squared matrix
 * @param[in] indptr Index pointer array
 * @param[in] indices Index list
 * @param[in] vals Numerical value array
 * @ingroup csingle
 */
LhfStatus lhfsWrapMatrix(LhfsMatrix mat, const size_t n, const LhfInt* indptr,
                         const LhfInt* indices, const float* vals);

/*!
 * @brief Wrap external data into a double-precision complex sparse matrix
 * @param[out] mat Matrix that holds the external data
 * @param[in] n Size the squared matrix
 * @param[in] indptr Index pointer array
 * @param[in] indices Index list
 * @param[in] vals Numerical value array
 * @ingroup ccomplexdouble
 */
LhfStatus lhfzWrapMatrix(LhfzMatrix mat, const size_t n, const LhfInt* indptr,
                         const LhfInt* indices, const void* vals);

/*!
 * @brief Wrap external data into a single-precision complex sparse matrix
 * @param[out] mat Matrix that holds the external data
 * @param[in] n Size the squared matrix
 * @param[in] indptr Index pointer array
 * @param[in] indices Index list
 * @param[in] vals Numerical value array
 * @ingroup ccomplexsingle
 */
LhfStatus lhfcWrapMatrix(LhfcMatrix mat, const size_t n, const LhfInt* indptr,
                         const LhfInt* indices, const void* vals);

// HIF preconditioners

/*!
 * @addtogroup
 * {
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
LhfdHIF lhfdCreate(const LhfdMatrix A, const LhfdMatrix S,
                   const double params[]);

/*!
 * @brief Destroy a double-precision HIF instance
 * @param[out] hif HIF preconditioner
 * @sa lhfdCreate
 */
LhfStatus lhfdDestroy(LhfdHIF hif);

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
LhfStatus lhfdSetup(LhfdHIF hif, const LhfdMatrix A, const LhfdMatrix S,
                    const double params[]);

/*!
 * @brief Update the @a A matrix in HIF
 * @param[out] hif A HIF instance
 * @param[in] A A new matrix
 * @note This function will not call factorization on @a A.
 * @sa lhfdRefactorize
 */
LhfStatus lhfdUpdate(LhfdHIF hif, const LhfdMatrix A);

/*!
 * @brief Refactorize a HIF preconditioner
 * @param[out] hif A HIF instance
 * @param[in] S A new sparsifier
 * @param[in] params Control parameters
 * @note This function will not update @a A inside @a hif
 * @note If @a params is @a NULL, then the default parameters will be used.
 * @sa lhfdUpdate
 */
LhfStatus lhfdRefactorize(LhfdHIF hif, const LhfdMatrix S,
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
LhfStatus lhfdApply(const LhfdHIF hif, const LhfOperationType op,
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
LhfStatus lhfdSolve(const LhfdHIF hif, const double* b, double* x);

/*!
 * @brief Get the statistics of a computed HIF instance
 * @param[in] hif A HIF instance
 * @param[out] status A length-9 array stores certain useful information
 *
 * Regarding the output @a status:
 *  - status[0]: Number of nonzeros of a preconditioner
 *  - status[1]: Total deferals
 *  - status[2]: Dynamic deferals (status[1]-status[2] are static deferals)
 *  - status[3]: Total droppings
 *  - status[4]: Droppings due to scalability-oriented strategy
 *  - status[5]: Number of levels
 *  - status[6]: Numerical rank of the whole preconditioner
 *  - status[7]: The numerical rank of the final Schur complement
 *  - status[8]: The size of the final Schur complement (status[7]<=status[8])
 */
LhfStatus lhfdGetStatus(const LhfdHIF hif, size_t status[]);

/*!
 * @brief Get number of nonzeros of a HIF preconditioner
 * @sa lhfdGetStatus
 */
size_t lhfdGetNnz(const LhfdHIF hif);

/*!
 * @brief Get number of levels
 * @sa lhfdGetStatus
 */
size_t lhfdGetLevels(const LhfdHIF hif);

/*!
 * @brief Get the Schur complement size
 * @sa lhfdGetStatus
 */
size_t lhfdGetSchurSize(const LhfdHIF hif);

/*!
 * @brief Get the Schur complement rank
 * @sa lhfdGetStatus
 */
size_t lhfdGetSchurRank(const LhfdHIF hif);

/*!
 * @}
 */

#ifdef __cplusplus
}
#endif

#endif /* _LIBHIFIR_H */
