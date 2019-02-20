//@HEADER
//----------------------------------------------------------------------------
//                Copyright (C) 2019 The PSMILU AUTHORS
//----------------------------------------------------------------------------
//@HEADER

/// \file psmilu_native_io.hpp
/// \brief Read and write native IO format
/// \authors Qiao,

#ifndef _PSMILU_NATIVE_IO_HPP
#define _PSMILU_NATIVE_IO_HPP

#include <algorithm>
#include <complex>
#include <cstdint>
#include <fstream>
#include <tuple>
#include <type_traits>

#include "psmilu_Array.hpp"
#include "psmilu_log.hpp"

namespace psmilu {
namespace internal {

/// \brief check if the os is litter endian
/// \ingroup util
inline bool is_little_endian() {
  int a = 1;
  return *(char *)&a;
}

/// \brief check the word size
/// \ingroup util
inline constexpr std::size_t get_word_size() { return sizeof(void *); }

/// \brief read the general information
/// \tparam InStream input stream type, should be \a std::ifstream
/// \param[in,out] i_str in-file streamer
/// \ingroup util
/// \return a \a tuple of args, from left to right, endianness, word size,
///         integer size, row-major flag, C index flag, double precision flag,
///         real/complex flag.
/// \sa write_general_info
template <class InStream>
inline std::tuple<bool, std::size_t, std::size_t, bool, bool, bool, bool>
read_general_info(InStream &i_str) {
  bool        is_little;
  std::size_t word_size, int_size;
  bool        is_row, is_c, is_double, is_real;

  char buf;
  i_str.read(&buf, 1);
  is_little = buf;
  i_str.read(&buf, 1);
  word_size = buf;
  i_str.read(&buf, 1);
  int_size = buf;
  i_str.read(&buf, 1);
  is_row = buf;
  i_str.read(&buf, 1);
  is_c = buf;
  i_str.read(&buf, 1);
  is_double = buf;
  i_str.read(&buf, 1);
  is_real = buf;

  return std::make_tuple(is_little, word_size, int_size, is_row, is_c,
                         is_double, is_real);
}

/// \brief read the matrix size information
/// \tparam InStream input stream type, should be \a std::ifstream
/// \param[in,out] i_str in-file streamer
/// \ingroup util
/// \return a \a tuple of args, from left to right, row size, column size,
///         number of nonzeros, and leading symmetric block size.
/// \sa write_matrix_sizes
///
/// Be aware that we use the fixed size integer (\a std::uint64_t) to store
/// size information.
template <class InStream>
inline std::tuple<std::uint64_t, std::uint64_t, std::uint64_t, std::uint64_t>
read_matrix_sizes(InStream &i_str) {
  std::uint64_t row, col, nnz, m;
  i_str.read(reinterpret_cast<char *>(&row), 8);
  i_str.read(reinterpret_cast<char *>(&col), 8);
  i_str.read(reinterpret_cast<char *>(&nnz), 8);
  i_str.read(reinterpret_cast<char *>(&m), 8);
  return std::make_tuple(row, col, nnz, m);
}

/// \brief read matrix data attributes
/// \tparam InStream input stream type, should be \a std::ifstream
/// \param[in,out] i_str in-file streamer
/// \param[in] int_size integer size in bytes
/// \param[in] real_size value data type size in bytes
/// \param[in] start_size primary size, i.e. row for CRS and col for CCS
/// \param[in] nnz total number of nonzeros
/// \param[out] ind_start raw data of size at least int_size*(start_size+1)
/// \param[out] indices raw data of size at least int_size*nnz
/// \param[out] vals raw data of size at least real_size*nnz
/// \ingroup util
/// \sa write_matrix_data_attrs
template <class InStream>
inline void read_matrix_data_attrs(InStream &i_str, const std::size_t int_size,
                                   const std::size_t   real_size,
                                   const std::uint64_t start_size,
                                   const std::uint64_t nnz, char *ind_start,
                                   char *indices, char *vals) {
  i_str.read(ind_start, (start_size + 1) * int_size);
  i_str.read(indices, nnz * int_size);
  i_str.read(vals, nnz * real_size);
}

/// \brief write general information
/// \tparam OutStream output stream type, should be \a std::ofstream
/// \param[in,out] o_str out-file streamer
/// \param[in] int_size integer size
/// \param[in] is_row row-major flag
/// \param[in] is_c C index flag
/// \param[in] is_double double precision flag
/// \param[in] is_real real/complex flag
/// \ingroup util
/// \sa read_general_info
template <class OutStream>
inline void write_general_info(OutStream &o_str, const std::size_t int_size,
                               const bool is_row, const bool is_c,
                               const bool is_double, const bool is_real) {
  char buf = is_little_endian();
  o_str.write(&buf, 1);
  buf = (char)get_word_size();
  o_str.write(&buf, 1);
  buf = (char)int_size;
  o_str.write(&buf, 1);
  buf = is_row;
  o_str.write(&buf, 1);
  buf = is_c;
  o_str.write(&buf, 1);
  buf = is_double;
  o_str.write(&buf, 1);
  buf = is_real;
  o_str.write(&buf, 1);
}

/// \brief write matrix size information
/// \tparam OutStream output stream type, should be \a std::ofstream
/// \param[in,out] o_str out-file streamer
/// \param[in] row row size
/// \param[in] col column size
/// \param[in] nnz total number of nonzeros
/// \param[in] m leading symmetric block size
/// \ingroup util
/// \sa read_matrix_sizes
template <class OutStream>
inline void write_matrix_sizes(OutStream &o_str, const std::uint64_t row,
                               const std::uint64_t col, const std::uint64_t nnz,
                               const std::uint64_t m) {
  o_str.write(reinterpret_cast<const char *>(&row), 8);
  o_str.write(reinterpret_cast<const char *>(&col), 8);
  o_str.write(reinterpret_cast<const char *>(&nnz), 8);
  o_str.write(reinterpret_cast<const char *>(&m), 8);
}

/// \brief write matrix data attributes
/// \tparam OutStream output stream type, should be \a std::ofstream
/// \param[in,out] o_str out-file streamer
/// \param[in] int_size integer size
/// \param[in] real_size value data type size
/// \param[in] start_size primary size
/// \param[in] nnz total number of nonzeros
/// \param[in] ind_start opaque pointer to start positions
/// \param[in] indices opaque pointer to indices
/// \param[in] vals opaque pointer to value array
/// \ingroup util
/// \sa read_matrix_data_attrs
template <class OutStream>
inline void write_matrix_data_attrs(
    OutStream &o_str, const std::size_t int_size, const std::size_t real_size,
    const std::uint64_t start_size, const std::uint64_t nnz,
    const void *ind_start, const void *indices, const void *vals) {
  o_str.write(reinterpret_cast<const char *>(ind_start),
              (start_size + 1) * int_size);
  o_str.write(reinterpret_cast<const char *>(indices), nnz * int_size);
  o_str.write(reinterpret_cast<const char *>(vals), nnz * real_size);
}

#ifndef DOXYGEN_SHOULD_SKIP_THIS

template <std::size_t SrcSize, std::size_t TgtSize>
inline void copy_vals_helper(const std::uint64_t nnz, const void *v1, void *v2);

template <>
inline void copy_vals_helper<4u, 8u>(const std::uint64_t nnz, const void *v1,
                                     void *v2) {
  // file is float, system is double
  const float *src = reinterpret_cast<const float *>(v1);
  double *     tgt = reinterpret_cast<double *>(v2);
  std::copy_n(src, nnz, tgt);
}

template <>
inline void copy_vals_helper<8u, 4u>(const std::uint64_t nnz, const void *v1,
                                     void *v2) {
  // file is double, system is float
  const double *src = reinterpret_cast<const double *>(v1);
  float *       tgt = reinterpret_cast<float *>(v2);
  std::copy_n(src, nnz, tgt);
}

template <>
inline void copy_vals_helper<8u, 16u>(const std::uint64_t nnz, const void *v1,
                                      void *v2) {
  // file is complex<float> system is complex<double>
  const std::complex<float> *src =
      reinterpret_cast<const std::complex<float> *>(v1);
  std::complex<double> *tgt = reinterpret_cast<std::complex<double> *>(v2);
  std::copy_n(src, nnz, tgt);
}

template <>
inline void copy_vals_helper<16u, 8u>(const std::uint64_t nnz, const void *v1,
                                      void *v2) {
  // file is complex<double>, system is complex<float>
  const std::complex<double> *src =
      reinterpret_cast<const std::complex<double> *>(v1);
  std::complex<float> *tgt = reinterpret_cast<std::complex<float> *>(v2);
  std::copy_n(src, nnz, tgt);
}

#endif  // DOXYGEN_SHOULD_SKIP_THIS

}  // namespace internal

/*!
 * \addtogroup util
 * @{
 */

/// \brief write a \ref CRS matrix
/// \tparam CsType Crs type, see \ref CRS
/// \param[in] filename file name
/// \param[in] A matrix for writing to file
/// \param[in] m leading symmetric block size
/// \sa read_native_bin
template <class CsType>
inline typename std::enable_if<CsType::ROW_MAJOR>::type write_native_bin(
    const char *filename, const CsType &A, const std::uint64_t m) {
  using value_type                       = typename CsType::value_type;
  constexpr static std::size_t real_size = sizeof(value_type);
  constexpr static std::size_t int_size  = sizeof(typename CsType::index_type);
  constexpr static bool is_real = std::is_floating_point<value_type>::value;
  constexpr static bool is_double =
      (is_real && sizeof(value_type) == 8u) || sizeof(value_type) == 16u;

  std::ofstream f(filename, std::ios_base::binary);
  psmilu_error_if(!f.is_open(), "failed to open file %s.", filename);
  internal::write_general_info(f, int_size, true, !CsType::ONE_BASED, is_double,
                               is_real);
  const std::uint64_t nrows(A.nrows()), ncols(A.ncols()), nnz(A.nnz());
  internal::write_matrix_sizes(f, nrows, ncols, nnz, m);
  const void *ind_start = reinterpret_cast<const void *>(A.row_start().data());
  const void *indices   = reinterpret_cast<const void *>(A.col_ind().data());
  const void *vals      = reinterpret_cast<const void *>(A.vals().data());
  internal::write_matrix_data_attrs(f, int_size, real_size, nrows, nnz,
                                    ind_start, indices, vals);
  f.close();
}

/// \brief write a \ref CCS matrix
/// \tparam CsType Ccs type, see \ref CCS
/// \param[in] filename file name
/// \param[in] A matrix for writing to file
/// \param[in] m leading symmetric block size
/// \sa read_native_bin
template <class CsType>
inline typename std::enable_if<!CsType::ROW_MAJOR>::type write_native_bin(
    const char *filename, const CsType &A, const std::uint64_t m) {
  using value_type                       = typename CsType::value_type;
  constexpr static std::size_t real_size = sizeof(value_type);
  constexpr static std::size_t int_size  = sizeof(typename CsType::index_type);
  constexpr static bool is_real = std::is_floating_point<value_type>::value;
  constexpr static bool is_double =
      (is_real && sizeof(value_type) == 8u) || sizeof(value_type) == 16u;

  std::ofstream f(filename, std::ios_base::binary);
  psmilu_error_if(!f.is_open(), "failed to open file %s.", filename);
  internal::write_general_info(f, int_size, false, !CsType::ONE_BASED,
                               is_double, is_real);
  const std::uint64_t nrows(A.nrows()), ncols(A.ncols()), nnz(A.nnz());
  internal::write_matrix_sizes(f, nrows, ncols, nnz, m);
  const void *ind_start = reinterpret_cast<const void *>(A.col_start().data());
  const void *indices   = reinterpret_cast<const void *>(A.row_ind().data());
  const void *vals      = reinterpret_cast<const void *>(A.vals().data());
  internal::write_matrix_data_attrs(f, int_size, real_size, nrows, nnz,
                                    ind_start, indices, vals);
  f.close();
}

// function prototype for ccs case

/// \brief read data from file to a \ref CCS matrix
/// \tparam CsType ccs type
/// \param[in] filename file name
/// \param[out] A matrix to store data from file
/// \sa write_native_bin
template <class CsType, typename T = std::uint64_t>
inline typename std::enable_if<!CsType::ROW_MAJOR, T>::type read_native_bin(
    const char *filename, CsType &A);

/// \brief read data from file to a \ref CRS matrix
/// \tparam CsType crs type
/// \param[in] filename file name
/// \param[out] A matrix to store data from file
/// \sa write_native_bin
template <class CsType, typename T = std::uint64_t>
inline typename std::enable_if<CsType::ROW_MAJOR, T>::type read_native_bin(
    const char *filename, CsType &A) {
  using value_type                       = typename CsType::value_type;
  constexpr static std::size_t real_size = sizeof(value_type);
  constexpr static std::size_t int_size  = sizeof(typename CsType::index_type);
  constexpr static bool is_real = std::is_floating_point<value_type>::value;
  constexpr static bool is_double =
      (is_real && sizeof(value_type) == 8u) || sizeof(value_type) == 16u;
  constexpr static bool is_c = !CsType::ONE_BASED;

  std::ifstream f(filename, std::ios_base::binary);
  psmilu_error_if(!f.is_open(), "failed to open file %s.", filename);

  bool        f_is_little;
  std::size_t f_word_size, f_int_size;
  bool        f_is_row, f_is_c, f_is_double, f_is_real;

  std::tie(f_is_little, f_word_size, f_int_size, f_is_row, f_is_c, f_is_double,
           f_is_real) = internal::read_general_info(f);

  const bool my_endianness  = internal::is_little_endian();
  const bool bad_endianness = f_is_little ^ my_endianness;
  psmilu_error_if(bad_endianness,
                  "current system endianness does not match that of the file "
                  "%s. Endianness converting is not available for now..",
                  filename);

  const bool bad_value_type = is_real ^ f_is_real;
  psmilu_error_if(bad_value_type,
                  "inconsistent value data type detected, i.e. trying to read "
                  "real data to complex matrix or vice versa.");

  psmilu_error_if(f_int_size > int_size,
                  "the integer size in file (%zd) is larger than the integer "
                  "size (%zd) used in the system; this is not allowed.",
                  f_int_size, int_size);

  const std::size_t f_real_size =
      f_is_real ? (f_is_double ? 8 : 4) : (f_is_double ? 16 : 8);
  psmilu_warning_if(
      f_real_size != real_size,
      "the floating size in file (%zd) does not match that used in the system "
      "(%zd), as a result, the precision cannot be preserved.",
      f_real_size, real_size);

  // read sizes
  std::uint64_t nrows, ncols, nnz, m;
  std::tie(nrows, ncols, nnz, m) = internal::read_matrix_sizes(f);

  const bool consistent = real_size == f_real_size && int_size == f_int_size;
  if (f_is_row) {
    A.resize(nrows, ncols);
    A.row_start().resize(nrows + 1);
    psmilu_error_if(A.row_start().status() == DATA_UNDEF,
                    "memory allocation failed");
    A.reserve(nnz);
    psmilu_error_if(
        A.col_ind().status() == DATA_UNDEF || A.vals().status() == DATA_UNDEF,
        "memory allocation failed");
    A.col_ind().resize(nnz);
    A.vals().resize(nnz);
    if (consistent) {
      char *ind_start = reinterpret_cast<char *>(A.row_start().data()),
           *indices   = reinterpret_cast<char *>(A.col_ind().data()),
           *vals      = reinterpret_cast<char *>(A.vals().data());
      internal::read_matrix_data_attrs(f, int_size, real_size, nrows, nnz,
                                       ind_start, indices, vals);
    } else {
      Array<char> ind_start(f_int_size * (nrows + 1)),
          indices(f_int_size * nnz), vals(f_real_size * nnz);
      psmilu_error_if(ind_start.status() == DATA_UNDEF ||
                          indices.status() == DATA_UNDEF ||
                          vals.status() == DATA_UNDEF,
                      "memory allocation failed");
      internal::read_matrix_data_attrs(f, f_int_size, f_real_size, nrows, nnz,
                                       ind_start.data(), indices.data(),
                                       vals.data());
      if (int_size != f_int_size) {
        if (f_int_size == 1u) {
          std::copy(ind_start.cbegin(), ind_start.cend(),
                    A.row_start().begin());
          std::copy(indices.cbegin(), indices.cend(), A.col_ind().begin());
        } else if (f_int_size == 2u) {
          const std::int16_t *i = reinterpret_cast<const std::int16_t *>(
                                 ind_start.data()),
                             *j = reinterpret_cast<const std::int16_t *>(
                                 indices.data());
          std::copy_n(i, nrows + 1, A.row_start().begin());
          std::copy_n(j, nnz, A.col_ind().begin());
        } else {
          // can only be 4
          const std::int32_t *i = reinterpret_cast<const std::int32_t *>(
                                 ind_start.data()),
                             *j = reinterpret_cast<const std::int32_t *>(
                                 indices.data());
          std::copy_n(i, nrows + 1, A.row_start().begin());
          std::copy_n(j, nnz, A.col_ind().begin());
        }
      } else {
        std::copy_n(reinterpret_cast<const typename CsType::index_type *>(
                        ind_start.data()),
                    nrows + 1, A.row_start().begin());
        std::copy_n(reinterpret_cast<const typename CsType::index_type *>(
                        indices.data()),
                    nnz, A.col_ind().begin());
      }
      if (real_size != f_real_size) {
        if (is_real) {
          if (is_double)
            internal::copy_vals_helper<4u, 8u>(nnz, vals.data(),
                                               A.vals().data());
          else
            internal::copy_vals_helper<8u, 4u>(nnz, vals.data(),
                                               A.vals().data());
        } else {
          if (is_double)
            internal::copy_vals_helper<8u, 16u>(nnz, vals.data(),
                                                A.vals().data());
          else
            internal::copy_vals_helper<16u, 8u>(nnz, vals.data(),
                                                A.vals().data());
        }
      } else
        std::copy_n(reinterpret_cast<const value_type *>(vals.data()), nnz,
                    A.vals().begin());
    }
    const int shift = f_is_c - is_c;
    if (shift != 0) {
      for (auto &v : A.row_start()) v += shift;
      for (auto &v : A.col_ind()) v += shift;
    }
    f.close();  // not needed but I'd like to have it
    psmilu_error_if(A.nnz() != nnz, "inconsistent nnz");
    return m;
  }
  f.close();
  using ccs_type = typename CsType::other_type;
  ccs_type Ap;
  read_native_bin(filename, Ap);
  A = CsType(Ap);
  return m;
}

// impl of ccs case

template <class CsType, typename T>
inline typename std::enable_if<!CsType::ROW_MAJOR, T>::type read_native_bin(
    const char *filename, CsType &A) {
  using value_type                       = typename CsType::value_type;
  constexpr static std::size_t real_size = sizeof(value_type);
  constexpr static std::size_t int_size  = sizeof(typename CsType::index_type);
  constexpr static bool is_real = std::is_floating_point<value_type>::value;
  constexpr static bool is_double =
      (is_real && sizeof(value_type) == 8u) || sizeof(value_type) == 16u;
  constexpr static bool is_c = !CsType::ONE_BASED;

  std::ifstream f(filename, std::ios_base::binary);
  psmilu_error_if(!f.is_open(), "failed to open file %s.", filename);

  bool        f_is_little;
  std::size_t f_word_size, f_int_size;
  bool        f_is_row, f_is_c, f_is_double, f_is_real;

  std::tie(f_is_little, f_word_size, f_int_size, f_is_row, f_is_c, f_is_double,
           f_is_real) = internal::read_general_info(f);

  const bool my_endianness  = internal::is_little_endian();
  const bool bad_endianness = f_is_little ^ my_endianness;
  psmilu_error_if(bad_endianness,
                  "current system endianness does not match that of the file "
                  "%s. Endianness converting is not available for now..",
                  filename);

  const bool bad_value_type = is_real ^ f_is_real;
  psmilu_error_if(bad_value_type,
                  "inconsistent value data type detected, i.e. trying to read "
                  "real data to complex matrix or vice versa.");

  psmilu_error_if(f_int_size > int_size,
                  "the integer size in file (%zd) is larger than the integer "
                  "size (%zd) used in the system; this is not allowed.",
                  f_int_size, int_size);

  const std::size_t f_real_size =
      f_is_real ? (f_is_double ? 8 : 4) : (f_is_double ? 16 : 8);
  psmilu_warning_if(
      f_real_size != real_size,
      "the floating size in file (%zd) does not match that used in the system "
      "(%zd), as a result, the precision cannot be preserved.",
      f_real_size, real_size);

  // read sizes
  std::uint64_t nrows, ncols, nnz, m;
  std::tie(nrows, ncols, nnz, m) = internal::read_matrix_sizes(f);

  const bool consistent = real_size == f_real_size && int_size == f_int_size;
  if (!f_is_row) {
    A.resize(nrows, ncols);
    A.col_start().resize(nrows + 1);
    psmilu_error_if(A.col_start().status() == DATA_UNDEF,
                    "memory allocation failed");
    A.reserve(nnz);
    psmilu_error_if(
        A.row_ind().status() == DATA_UNDEF || A.vals().status() == DATA_UNDEF,
        "memory allocation failed");
    A.row_ind().resize(nnz);
    A.vals().resize(nnz);
    if (consistent) {
      char *ind_start = reinterpret_cast<char *>(A.col_start().data()),
           *indices   = reinterpret_cast<char *>(A.row_ind().data()),
           *vals      = reinterpret_cast<char *>(A.vals().data());
      internal::read_matrix_data_attrs(f, int_size, real_size, nrows, nnz,
                                       ind_start, indices, vals);
    } else {
      Array<char> ind_start(f_int_size * (nrows + 1)),
          indices(f_int_size * nnz), vals(f_real_size * nnz);
      psmilu_error_if(ind_start.status() == DATA_UNDEF ||
                          indices.status() == DATA_UNDEF ||
                          vals.status() == DATA_UNDEF,
                      "memory allocation failed");
      internal::read_matrix_data_attrs(f, f_int_size, f_real_size, nrows, nnz,
                                       ind_start.data(), indices.data(),
                                       vals.data());
      if (int_size != f_int_size) {
        if (f_int_size == 1u) {
          std::copy(ind_start.cbegin(), ind_start.cend(),
                    A.col_start().begin());
          std::copy(indices.cbegin(), indices.cend(), A.row_ind().begin());
        } else if (f_int_size == 2u) {
          const std::int16_t *i = reinterpret_cast<const std::int16_t *>(
                                 ind_start.data()),
                             *j = reinterpret_cast<const std::int16_t *>(
                                 indices.data());
          std::copy_n(i, nrows + 1, A.col_start().begin());
          std::copy_n(j, nnz, A.row_ind().begin());
        } else {
          // can only be 4
          const std::int32_t *i = reinterpret_cast<const std::int32_t *>(
                                 ind_start.data()),
                             *j = reinterpret_cast<const std::int32_t *>(
                                 indices.data());
          std::copy_n(i, nrows + 1, A.col_start().begin());
          std::copy_n(j, nnz, A.row_ind().begin());
        }
      } else {
        std::copy_n(reinterpret_cast<const typename CsType::index_type *>(
                        ind_start.data()),
                    nrows + 1, A.col_start().begin());
        std::copy_n(reinterpret_cast<const typename CsType::index_type *>(
                        indices.data()),
                    nnz, A.row_ind().begin());
      }
      if (real_size != f_real_size) {
        if (is_real) {
          if (is_double)
            internal::copy_vals_helper<4u, 8u>(nnz, vals.data(),
                                               A.vals().data());
          else
            internal::copy_vals_helper<8u, 4u>(nnz, vals.data(),
                                               A.vals().data());
        } else {
          if (is_double)
            internal::copy_vals_helper<8u, 16u>(nnz, vals.data(),
                                                A.vals().data());
          else
            internal::copy_vals_helper<16u, 8u>(nnz, vals.data(),
                                                A.vals().data());
        }
      } else
        std::copy_n(reinterpret_cast<const value_type *>(vals.data()), nnz,
                    A.vals().begin());
    }
    const int shift = f_is_c - is_c;
    if (shift != 0) {
      for (auto &v : A.col_start()) v += shift;
      for (auto &v : A.row_ind()) v += shift;
    }
    f.close();
    psmilu_error_if(A.nnz() != nnz, "inconsistent nnz");
    return m;
  }
  f.close();
  using crs_type = typename CsType::other_type;
  crs_type Ap;
  read_native_bin(filename, Ap);
  A = CsType(Ap);
  return m;
}

/*!
 * @}
 */ // group util

}  // namespace psmilu

#endif  // _PSMILU_NATIVE_IO_HPP
