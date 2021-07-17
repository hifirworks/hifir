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

#ifdef __cplusplus
extern "C" {
#endif

/*!
 * @addtogroup c
 */

#ifndef LIBHIFIR_INT
#  define LIBHIFIR_INT int
#endif
typedef LIBHIFIR_INT LhfInt;

/*!
 * @def LIBHIFIR_NUMBER_PARAMS
 * @brief Total number of parameters
 * @note In @a libhifir, we use a double array of size LIBHIFIR_NUMBER_PARAMS
 *       for control parameters in factorization.
 */
#define LIBHIFIR_NUMBER_PARAMS 40 /* reserved for the future */

/*!
 * @def LIBHIFIR_DROPTOL_L
 * @brief Drop tolerance for L factor
 */
#define LIBHIFIR_DROPTOL_L 0

/*!
 * @def LIBHIFIR_DROPTOL_U
 * @brief Drop tolerance for U factor
 */
#define LIBHIFIR_DROPTOL_U 1

/*!
 * @def LIBHIFIR_COND_D
 * @brief Conditioning threshold for diagonal
 */
#define LIBHIFIR_COND_D 2

/*!
 * @def LIBHIFIR_COND
 * @brief Conditioning threshold for L and U factors
 */
#define LIBHIFIR_COND 3

/*!
 * @def LIBHIFIR_ALPHA_L
 * @brief Scalability-oriented dropping factor for L
 */
#define LIBHIFIR_ALPHA_L 4

/*!
 * @def LIBHIFIR_ALPHA_U
 * @brief Scalability-oriented dropping factor for U
 */
#define LIBHIFIR_ALPHA_U 5

/*!
 * @def LIBHIFIR_VERBOSE
 * @brief Verbose level
 */
#define LIBHIFIR_VERBOSE 6

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
#define LIBHIFIR_REORDER 7

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
#define LIBHIFIR_SYMMPRELVLS 8

/*!
 * @def LIBHIFIR_THREADS
 * @brief Option for choosing number of threads for Schur computation
 */
#define LIBHIFIR_THREADS 9

/*!
 * @def LIBHIFIR_RRQR_COND
 * @brief condition number threshold used in the final RRQR
 */
#define LIBHIFIR_RRQR_COND 10

/*!
 * @def LIBHIFIR_PIVOT
 * @brief Pivoting option
 */
#define LIBHIFIR_PIVOT 11

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
#define LIBHIFIR_BETA 12

/*!
 * @def LIBHIFIR_ISSYMM
 */
#define LIBHIFIR_ISSYMM 13

/*!
 * @def LIBHIFIR_NOPRE
 * @brief Option to turn on/off preprocessing
 */
#define LIBHIFIR_NOPRE 14

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
 * @struct LhfdMatrixStruct
 * @brief Double precision sparse matrix in @a libhifir
 */
struct LhfdMatrixStruct;
typedef struct LhfdMatrixStruct* LhfdMatrix; /*!< Pointer of sparse matrix */

/*!
 * @struct LhfsMatrixStruct
 * @brief Single precision sparse matrix in @a libhifir
 */
struct LhfsMatrixStruct;
typedef struct LhfsMatrixStruct* LhfsMatrix; /*!< Pointer of sparse matrix */

/*!
 * @struct LhfzMatrixStruct
 * @brief Double precision complex sparse matrix in @a libhifir
 */
struct LhfzMatrixStruct;
typedef struct LhfzMatrixStruct* LhfzMatrix; /*!< Pointer of sparse matrix */

/*!
 * @struct LhfcMatrixStruct
 * @brief Single precision complex sparse matrix in @a libhifir
 */
struct LhfcMatrixStruct;
typedef struct LhfcMatrixStruct* LhfcMatrix; /*!< Pointer of sparse matrix */

/*!
 * @struct LhfdHIFStruct
 * @brief Double precision HIF structure in @a libhifir
 */
struct LhfdHIFStruct;
typedef struct LhfdHIFStruct* LhfdHIF; /*!< Pointer of HIF */

/*!
 * @struct LhfsHIFStruct
 * @brief Single precision HIF structure in @a libhifir
 */
struct LhfsHIFStruct;
typedef struct LhfsHIFStruct* LhfsHIF; /*!< Pointer of HIF */

/*!
 * @struct LhfzHIFStruct
 * @brief Double precision complex HIF structure in @a libhifir
 */
struct LhfzHIFStruct;
typedef struct LhfzHIFStruct* LhfzHIF; /*!< Pointer of HIF */

/*!
 * @struct LhfcHIFStruct
 * @brief Single precision complex HIF structure in @a libhifir
 */
struct LhfcHIFStruct;
typedef struct LhfcHIFStruct* LhfcHIF; /*!< Pointer of HIF */

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
LhfdMatrix lhfdCreateMatrix(const int is_rowmajor, const size_t n,
                            const LhfInt* indptr, const LhfInt* indices,
                            const double* vals);

/*!
 * @brief Destroy a double-precision matrix instance
 * @param[in,out] mat Matrix instance
 */
int lhfdDestroyMatrix(LhfdMatrix mat);

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
LhfsMatrix lhfsCreateMatrix(const int is_rowmajor, const size_t n,
                            const LhfInt* indptr, const LhfInt* indices,
                            const float* vals);

/*!
 * @brief Destroy a single-precision matrix instance
 * @param[in,out] mat Matrix instance
 */
int lhfsDestroyMatrix(LhfsMatrix mat);

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
LhfzMatrix lhfzCreateMatrix(const int is_rowmajor, const size_t n,
                            const LhfInt* indptr, const LhfInt* indices,
                            const void* vals);

/*!
 * @brief Destroy a double-precision complex matrix instance
 * @param[in,out] mat Matrix instance
 */
int lhfzDestroyMatrix(LhfzMatrix mat);

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
LhfcMatrix lhfcCreateMatrix(const int is_rowmajor, const size_t n,
                            const LhfInt* indptr, const LhfInt* indices,
                            const void* vals);

/*!
 * @brief Destroy a single-precision complex matrix instance
 * @param[in,out] mat Matrix instance
 */
int lhfcDestroyMatrix(LhfcMatrix mat);

/*!
 * @brief Wrap external data into a double-precision sparse matrix
 * @param[in] n Size the squared matrix
 * @param[in] indptr Index pointer array
 * @param[in] indices Index list
 * @param[in] vals Numerical value array
 */
int lhfdWrapMatrix(const size_t n, const LhfInt* indptr, const LhfInt* indices,
                   const double* vals);

/*!
 * @brief Wrap external data into a single-precision sparse matrix
 * @param[in] n Size the squared matrix
 * @param[in] indptr Index pointer array
 * @param[in] indices Index list
 * @param[in] vals Numerical value array
 */
int lhfsWrapMatrix(const size_t n, const LhfInt* indptr, const LhfInt* indices,
                   const float* vals);

/*!
 * @brief Wrap external data into a double-precision complex sparse matrix
 * @param[in] n Size the squared matrix
 * @param[in] indptr Index pointer array
 * @param[in] indices Index list
 * @param[in] vals Numerical value array
 */
int lhfzWrapMatrix(const size_t n, const LhfInt* indptr, const LhfInt* indices,
                   const void* vals);

/*!
 * @brief Wrap external data into a single-precision complex sparse matrix
 * @param[in] n Size the squared matrix
 * @param[in] indptr Index pointer array
 * @param[in] indices Index list
 * @param[in] vals Numerical value array
 */
int lhfcWrapMatrix(const size_t n, const LhfInt* indptr, const LhfInt* indices,
                   const void* vals);

#ifdef __cplusplus
}
#endif

#endif /* _LIBHIFIR_H */
