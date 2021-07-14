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

#ifdef __cplusplus
extern "C" {
#endif

/*!
 * @addtogroup c
 */

/*!
 * @def LIBHIFIR_NUMBER_PARAMS
 * @brief Total number of parameters
 * @note In @a libhifir, we use a double array of size LIBHIFIR_NUMBER_PARAMS
 *       for control parameters in factorization.
 */
#define LIBHIFIR_NUMBER_PARAMS 40 /* reserved for the future */

/*!
 * @def LIBHIFIR_MATRIX_TYPE
 * @brief Matrix type option
 */
#define LIBHIFIR_MATRIX_TYPE 0

/*!
 * @def LIBHIFIR_MATRIX_CRS
 * @brief CRS input option
 */
#define LIBHIFIR_MATRIX_CRS 1

/*!
 * @def LIBHIFIR_MATRIX_CCS
 * @brief CCS input option
 */
#define LIBHIFIR_MATRIX_CCS 0

/*!
 * @def LIBHIFIR_DROPTOL_L
 * @brief Drop tolerance for L factor
 */
#define LIBHIFIR_DROPTOL_L 1

/*!
 * @def LIBHIFIR_DROPTOL_U
 * @brief Drop tolerance for U factor
 */
#define LIBHIFIR_DROPTOL_U 2

/*!
 * @def LIBHIFIR_COND_D
 * @brief Conditioning threshold for diagonal
 */
#define LIBHIFIR_COND_D 3

/*!
 * @def LIBHIFIR_COND
 * @brief Conditioning threshold for L and U factors
 */
#define LIBHIFIR_COND 4

/*!
 * @def LIBHIFIR_ALPHA_L
 * @brief Scalability-oriented dropping factor for L
 */
#define LIBHIFIR_ALPHA_L 5

/*!
 * @def LIBHIFIR_ALPHA_U
 * @brief Scalability-oriented dropping factor for U
 */
#define LIBHIFIR_ALPHA_U 6

/*!
 * @def LIBHIFIR_VERBOSE
 * @brief Verbose level
 */
#define LIBHIFIR_VERBOSE 7

/*!
 * @def LIBHIFIR_VERBOSE_NULL
 */
#define LIBHIFIR_VERBOSE_NULL 0

/*!
 * @def LIBHIFIR_VERBOSE_INFO
 */
#define LIBHIFIR_VERBOSE_INFO 1

/*!
 * @def LIBHIFIR_VERBOSE_PRE
 */
#define LIBHIFIR_VERBOSE_PRE 2

/*!
 * @def LIBHIFIR_VERBOSE_FAC
 */
#define LIBHIFIR_VERBOSE_FAC 4

/*!
 * @def LIBHIFIR_VERBOSE_PRE_TIME
 */
#define LIBHIFIR_VERBOSE_PRE_TIME 8

/*!
 * @def LIBHIFIR_VERBOSE_MEM
 */
#define LIBHIFIR_VERBOSE_MEM 16

/*!
 * @def LIBHIFIR_REORDER
 * @brief Reorder option
 */
#define LIBHIFIR_REORDER 8

/*!
 * @def LIBHIFIR_REORDER_OFF
 */
#define LIBHIFIR_REORDER_OFF 0

/*!
 * @def LIBHIFIR_REORDER_AUTO
 */
#define LIBHIFIR_REORDER_AUTO 1

/*!
 * @def LIBHIFIR_REORDER_AMD
 */
#define LIBHIFIR_REORDER_AMD 2

/*!
 * @def LIBHIFIR_REORDER_RCM
 */
#define LIBHIFIR_REORDER_RCM 3

/*!
 * @def LIBHIFIR_SYMMPRELVLS
 * @brief Option for tuning symmetric preprocessing levels
 */
#define LIBHIFIR_SYMMPRELVLS 9

/*!
 * @def LIBHIFIR_THREADS
 * @brief Option for choosing number of threads for Schur computation
 */
#define LIBHIFIR_THREADS 10

/*!
 * @def LIBHIFIR_RRQR_COND
 * @brief condition number threshold used in the final RRQR
 */
#define LIBHIFIR_RRQR_COND 11

/*!
 * @def LIBHIFIR_PIVOT
 * @brief Pivoting option
 */
#define LIBHIFIR_PIVOT 12

/*!
 * @def LIBHIFIR_PIVOTING_OFF
 */
#define LIBHIFIR_PIVOTING_OFF 0

/*!
 * @def LIBHIFIR_PIVOTING_ON
 */
#define LIBHIFIR_PIVOTING_ON 1

/*!
 * @def LIBHIFIR_PIVOTING_AUTO
 */
#define LIBHIFIR_PIVOTING_AUTO 0

/*!
 * @def LIBHIFIR_BETA
 * @brief Option for the threshold used in preventing bad scaling factors from
 *        equlibriation in preprocessing.
 */
#define LIBHIFIR_BETA 13

/*!
 * @def LIBHIFIR_ISSYMM
 */
#define LIBHIFIR_ISSYMM 14

/*!
 * @def LIBHIFIR_NOPRE
 * @brief Option to turn on/off preprocessing
 */
#define LIBHIFIR_NOPRE 15

/*!
 * @def LIBHIFIR_PREC_LEN
 * @brief Array size to "encode" a preconditioner
 * @note char M[LIBHIFIR_PREC] creates a preconditioner token M.
 */
#define LIBHIFIR_PREC_LEN 10

/*!
 * @def LIBHIFIR_SUCCESS
 * @brief Successful return
 */
#define LIBHIFIR_SUCCESS 0

/*!
 * @def LIBHIFIR_PREC_EXIST
 * @brief Existing preconditioner while calling @ref libhifir_create
 */
#define LIBHIFIR_PREC_EXIST 1

/*!
 * @def LIBHIFIR_BAD_PREC
 */
#define LIBHIFIR_BAD_PREC 2

/*!
 * @def LIBHIFIR_HIFIR_ERROR
 * @note Internal HIFIR error, call @ref libhifir_error_msg
 */
#define LIBHIFIR_HIFIR_ERROR 3

/*!
 * @brief Get the version of HIFIR
 * @param[out] vs Global version
 * @note The input should be a length-three array, in which the first, second,
 *       and third entries store the global, major, and minor versions, resp.
 */
void libhifir_version(int vs[]);

/*!
 * @brief Enable global warning
 * @sa libhifir_disable_warning
 */
void libhifir_enable_warning(void);

/*!
 * @brief Disable global warning
 * @sa libhifir_enable_warning
 */
void libhifir_disable_warning(void);

/*!
 * @brief Get internal error message from HIFIR
 * @note NULL indicates no error (or error message expired)
 * @note Call right after obtaining LIBHIFIR_HIFIR_ERROR
 * @todo How do handle this in Fortran?
 */
const char *libhifir_error_msg(void);

/*!
 * @brief Initialize and setup parameters
 * @param[out] params Control parameters, length must be no smaller than
 *                    @ref LIBHIFIR_NUMBER_PARAMS
 */
void libhifir_setup_params(double params[]);

/*!
 * @brief Create a preconditioner token
 * @param[in,out] M Preconditioner token
 * @param[in] is_mixed Whether or not mixed-precision is used
 * @param[in] is_complex Whether or not complex number is used
 * @param[in] is_int64 Whether or not using 64-bit integer
 *
 * Notice that if @a is_mixed is @a NULL, then double precision preconditioner
 * is used. If @a is_complex is @a NULL, then real arithmetic is used. If
 * @a is_int64 is @a NULL, then 32-bit integer is used. All these parameters
 * are integers pass-by-ref if they are not @a NULL and are treated as Boolean
 * tags (i.e., either 1 or 0).
 *
 * @note @a M should be at least length of @ref LIBHIFIR_PREC
 * @warning One should not manually modify @a M
 * @sa libhifir_destroy
 */
int libhifir_create(const int *is_mixed, const int *is_complex,
                    const int *is_int64, char M[]);

/*!
 * @brief Check the configurations of a preconditioner
 * @param[in] M Preconditioner token
 * @param[out] is_mixed Mixed-precision tag
 * @param[out] is_complex Complex-arithmetic tag
 * @param[out] is_int64 64bit-integer tag
 * @sa libhifir_create
 */
int libhifir_check(const char M[], int *is_mixed, int *is_complex,
                   int *is_int64);

/*!
 * @brief Destroy a preconditioner
 * @param[in,out] M Preconditioner token
 * @sa libhifir_create
 */
int libhifir_destroy(char M[]);

/*!
 * @brief Check if a preconditioner is empty
 * @param[in] M Preconditioner token
 */
int libhifir_empty(const char M[]);

/*!
 * @brief Query preconditioner statistics
 * @param[in] M Preconditioner token
 * @param[out] stats Statistics output
 * @sa libhifir_empty
 */
void libhifir_query_stats(const char M[], long long stats[]);

/*!
 * @brief Factorize a HIF preconditioner
 * @param[in] M Preconditioner token
 * @param[in] n Input matrix size
 * @param[in] ind_start Pointer start array in compressed format
 * @param[in] indices Index array in compressed format
 * @param[in] vals Value array in compressed format
 * @param[in] params Control parameters
 *
 * The input matrix can be either CRS or CCS, depending on the matrix type
 * option @ref LIBHIFIR_MATRIX_TYPE.
 *
 * @sa libhifir_solve, libhifir_mmultiply
 */
int libhifir_factorize(const char M[], const long long *n,
                       const void *ind_start, const void *indices,
                       const void *vals, const double params[]);

/*!
 * @brief Perform multilevel triangular solve
 * @param[in] M Preconditioner token
 * @param[in] n Input size of the vector
 * @param[in] b Input RHS vector
 * @param[in] trans Transpose/Hermitian tag (NULL treated as no-tran)
 * @param[in] rank Rank for the final Schur complement (NULL treated as 0)
 * @param[out] x Solution output of
 *             @f$\boldsymbol{x}=\boldsymbol{M}^{g}\boldsymbol{b}@f$ or
 *             @f$\boldsymbol{x}=\boldsymbol{M}^{gH}\boldsymbol{b}@f$
 * @sa libhifir_mmultiply, libhifir_hifir
 */
int libhifir_solve(const char M[], const long long *n, const void *b,
                   const int *trans, const int *rank, void *x);

/*!
 * @brief Perform multilevel matrix-vector multiplication
 * @param[in] M Preconditioner token
 * @param[in] n Input size of the vector
 * @param[in] b Input vector
 * @param[in] trans Transpose/Hermitian tag (NULL treated as no-tran)
 * @param[in] rank Rank for the final Schur complement (NULL treated as 0)
 * @param[out] x Solution output of
 *             @f$\boldsymbol{x}=\boldsymbol{M}\boldsymbol{b}@f$ or
 *             @f$\boldsymbol{x}=\boldsymbol{M}^{H}\boldsymbol{b}@f$
 * @sa libhifir_solve
 */
int libhifir_mmultiply(const char M[], const long long *n, const void *b,
                       const int *trans, const int *rank, void *x);

/*!
 * @brief Perform multilevel triangular solve with iterative refinement
 * @param[in] M Preconditioner token
 * @param[in] n Input size of the vector
 * @param[in] ind_start Pointer start array in compressed format
 * @param[in] indices Index array in compressed format
 * @param[in] vals Value array in compressed format
 * @param[in] is_crs Boolean tag indicating matrix types (CRS,true), (CCS,false)
 * @param[in] b Input vector
 * @param[in] nirs Number of iterative refinement. If the input is 1 or @a NULL,
 *                 then this function is equivalent to @ref libhifir_solve.
 * @param[in] betas Residual bound in IR in 2-norm. This is a length-two array,
 *                  where the first entry beta[0] (@f$\beta_L@f$) is the lower
 *                  bound, and the second entry beta[1] (@f$\beta_U@f$) is the
 *                  upper bound. If @a NULL is passed in, then unbounded IR
 *                  will be performed, i.e., running IR with fixed @a nirs
 *                  iterations.
 * @param[in] trans Transpose/Hermitian tag (NULL treated as no-tran)
 * @param[in] rank Rank for the final Schur complement (NULL treated as -1)
 * @param[out] x Solution vector from iterative refinement
 * @param[out] iters If @a betas is not @a NULL, then the actual iterations
 *             of refinement is stored in this variable if it is supplied.
 *             Otherwise, this function does not touch this variable.
 * @param[out] ir_status Similar to @a iters, this varaiable indicates the
 *                       status of bounded IR (i.e., @a betas is not empty).
 *                       If @a ir_status is zero, then the IR converges; if
 *                       if @a ir_status > 0, then it diverges; otherwise, it
 *                       reaches maxit bound.
 * @sa libhifir_solve, libhifir_mmultiply
 */
int libhifir_hifir(const char M[], const long long *n, const void *ind_start,
                   const void *indices, const void *vals, const int *is_crs,
                   const void *b, const int *nirs, const double *betas,
                   const int *trans, const int *rank, void *x, int *iters,
                   int *ir_status);

/*!
 * @}
 */ /* group c */

#ifndef DOXYGEN_SHOULD_SKIP_THIS

/* versions */
void libhifir_version_(int vs[]);
void libhifir_version__(int vs[]);
void LIBHIFIR_VERSION(int vs[]);
void LIBHIFIR_VERSION_(int vs[]);
void LIBHIFIR_VERSION__(int vs[]);

/* warnings */
void libhifir_enable_warning_(void);
void libhifir_enable_warning__(void);
void LIBHIFIR_ENABLE_WARNING(void);
void LIBHIFIR_ENABLE_WARNING_(void);
void LIBHIFIR_ENABLE_WARNING__(void);
void libhifir_disable_warning_(void);
void libhifir_disable_warning__(void);
void LIBHIFIR_DISABLE_WARNING(void);
void LIBHIFIR_DISABLE_WARNING_(void);
void LIBHIFIR_DISABLE_WARNING__(void);

/* parameters */
void libhifir_setup_params_(double params[]);
void libhifir_setup_params__(double params[]);
void LIBHIFIR_SETUP_PARAMS(double params[]);
void LIBHIFIR_SETUP_PARAMS_(double params[]);
void LIBHIFIR_SETUP_PARAMS__(double params[]);

/* create */
int libhifir_create_(const int *is_mixed, const int *is_complex,
                     const int *is_int64, char M[]);
int libhifir_create__(const int *is_mixed, const int *is_complex,
                      const int *is_int64, char M[]);
int LIBHIFIR_CREATE(const int *is_mixed, const int *is_complex,
                    const int *is_int64, char M[]);
int LIBHIFIR_CREATE_(const int *is_mixed, const int *is_complex,
                     const int *is_int64, char M[]);
int LIBHIFIR_CREATE__(const int *is_mixed, const int *is_complex,
                      const int *is_int64, char M[]);

/* check */
int libhifir_check_(const char M[], int *is_mixed, int *is_complex,
                    int *is_int64);
int libhifir_check__(const char M[], int *is_mixed, int *is_complex,
                     int *is_int64);
int LIBHIFIR_CHECK(const char M[], int *is_mixed, int *is_complex,
                   int *is_int64);
int LIBHIFIR_CHECK_(const char M[], int *is_mixed, int *is_complex,
                    int *is_int64);
int LIBHIFIR_CHECK__(const char M[], int *is_mixed, int *is_complex,
                     int *is_int64);

/* destroy */
int libhifir_destroy_(char M[]);
int libhifir_destroy__(char M[]);
int LIBHIFIR_DESTROY(char M[]);
int LIBHIFIR_DESTROY_(char M[]);
int LIBHIFIR_DESTROY__(char M[]);

/* empty */
int libhifir_empty_(const char M[]);
int libhifir_empty__(const char M[]);
int LIBHIFIR_EMPTY(const char M[]);
int LIBHIFIR_EMPTY_(const char M[]);
int LIBHIFIR_EMPTY__(const char M[]);

/* stats */
void libhifir_query_stats_(const char M[], long long stats[]);
void libhifir_query_stats__(const char M[], long long stats[]);
void LIBHIFIR_QUERY_STATS(const char M[], long long stats[]);
void LIBHIFIR_QUERY_STATS_(const char M[], long long stats[]);
void LIBHIFIR_QUERY_STATS__(const char M[], long long stats[]);

/* factorize */
int libhifir_factorize_(const char M[], const long long *n,
                        const void *ind_start, const void *indices,
                        const void *vals, const double params[]);
int libhifir_factorize__(const char M[], const long long *n,
                         const void *ind_start, const void *indices,
                         const void *vals, const double params[]);
int LIBHIFIR_FACTORIZE(const char M[], const long long *n,
                       const void *ind_start, const void *indices,
                       const void *vals, const double params[]);
int LIBHIFIR_FACTORIZE_(const char M[], const long long *n,
                        const void *ind_start, const void *indices,
                        const void *vals, const double params[]);
int LIBHIFIR_FACTORIZE__(const char M[], const long long *n,
                         const void *ind_start, const void *indices,
                         const void *vals, const double params[]);

/* solve */
int libhifir_solve_(const char M[], const long long *n, const void *b,
                    const int *trans, const int *rank, void *x);
int libhifir_solve__(const char M[], const long long *n, const void *b,
                     const int *trans, const int *rank, void *x);
int LIBHIFIR_SOLVE(const char M[], const long long *n, const void *b,
                   const int *trans, const int *rank, void *x);
int LIBHIFIR_SOLVE_(const char M[], const long long *n, const void *b,
                    const int *trans, const int *rank, void *x);
int LIBHIFIR_SOLVE__(const char M[], const long long *n, const void *b,
                     const int *trans, const int *rank, void *x);

/* multiply */
int libhifir_mmultiply_(const char M[], const long long *n, const void *b,
                        const int *trans, const int *rank, void *x);
int libhifir_mmultiply__(const char M[], const long long *n, const void *b,
                         const int *trans, const int *rank, void *x);
int LIBHIFIR_MMULTIPLY(const char M[], const long long *n, const void *b,
                       const int *trans, const int *rank, void *x);
int LIBHIFIR_MMULTIPLY_(const char M[], const long long *n, const void *b,
                        const int *trans, const int *rank, void *x);
int LIBHIFIR_MMULTIPLY__(const char M[], const long long *n, const void *b,
                         const int *trans, const int *rank, void *x);

/* hifir */
int libhifir_hifir_(const char M[], const long long *n, const void *ind_start,
                    const void *indices, const void *vals, const int *is_crs,
                    const void *b, const int *nirs, const double *betas,
                    const int *trans, const int *rank, void *x, int *iters,
                    int *ir_status);
int libhifir_hifir__(const char M[], const long long *n, const void *ind_start,
                     const void *indices, const void *vals, const int *is_crs,
                     const void *b, const int *nirs, const double *betas,
                     const int *trans, const int *rank, void *x, int *iters,
                     int *ir_status);
int LIBHIFIR_HIFIR(const char M[], const long long *n, const void *ind_start,
                   const void *indices, const void *vals, const int *is_crs,
                   const void *b, const int *nirs, const double *betas,
                   const int *trans, const int *rank, void *x, int *iters,
                   int *ir_status);
int LIBHIFIR_HIFIR_(const char M[], const long long *n, const void *ind_start,
                    const void *indices, const void *vals, const int *is_crs,
                    const void *b, const int *nirs, const double *betas,
                    const int *trans, const int *rank, void *x, int *iters,
                    int *ir_status);
int LIBHIFIR_HIFIR__(const char M[], const long long *n, const void *ind_start,
                     const void *indices, const void *vals, const int *is_crs,
                     const void *b, const int *nirs, const double *betas,
                     const int *trans, const int *rank, void *x, int *iters,
                     int *ir_status);

#endif /* DOXYGEN_SHOULD_SKIP_THIS */

#ifdef __cplusplus
}
#endif

#endif /* _LIBHIFIR_H */
