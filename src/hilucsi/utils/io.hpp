//@HEADER
//----------------------------------------------------------------------------
//                Copyright (C) 2019 The HILUCSI AUTHORS
//----------------------------------------------------------------------------
//@HEADER

/// \file hilucsi/utils/io.hpp
/// \brief Read and write native IO formats
/// \authors Qiao,
///
/// We define two native file formats for exchanging compressed storage data;
/// these two formats are the HILUCSI binary and HILUCSI ASCII formats. The
/// former is for efficient IO, while the latter is for data readability and
/// compatibility.
///
/// First, let's introduce the binary format. There are three data groups that
/// are stored: 1) \a general \a information, which requires 7 bytes in total;
/// sequentially, they are, byte by byte, platform endianness (1 for little,
/// 0 for big), word size (8 for 64bit and 4 for 32bit), integer size, storage
/// scheme (1 for CRS and 0 for CCS), reserved, floating
/// point precision flag (1 for \a double and 0 for \a float), and the value
/// data type (1 for real and 0 for complex). 2) \a matrix \a sizes, the sizes
/// are stored in \b fixed-size integers, which are type of \a uint64_t, thus
/// having 32 bytes in total; four size attributes are stored, which are row
/// size, column size, total number of nonzeros, and the leading block size,
/// resp. 3) The final group is the \a matrix \a data \a attributes, the index
/// position (pointer) array is stored first, followed by index array and
/// the value array.
///
/// \warning OS endianness is not compatible for now, e.g. if a file is
///          written on a Little-Endian machine, then it must be loaded from
///          Little-Endian machines.
///
/// Second, let's take a look at the ASCII file format, which is designed for
/// cross-platform compatibility. Like the binary format, there are three
/// data groups as well. For general information, three characters are stored
/// in a single line, ?$%, where ? is the matrix storage scheme used, 'R' for
/// CRS while 'C' for CCS; $ is reserved, which are
/// for C and Fortran based index systems, resp; finally, % is the data type,
/// which adapts the naming scheme in BLAS, i.e. 'D' for double, 'S' for
/// single, 'Z' for complex double, and 'C' for complex single. For instance,
/// \a RCD defines a CSR matrix with 0-based index of double precision values.
/// Similary to the binary format, the sizes are stored and separated by white
/// space. Finally, for the data attributes, the index pointer is stored first,
/// then indices and values. Regarding numerical value format, we use C++ IO
/// operators with proper precisions so that no information is truncated
/// during IO. Specail note for complex types, the standard format for complex
/// is (real,imag), which should be used if one consider writing reader/writer
/// in other programming languages.
///
/// For ASCII format, the groups are separated by \a newline, while either
/// white space or \a newline can be used to separate the integral and
/// numerical values. Also, the data groups are compactly stored, i.e. no
/// gaps are allowed. Regarding comments (including empty lines), they are
/// only allowed at the beginning of the file; a line of comment starts with
/// character '#', which is a commonly used convension, e.g. shell, make,
/// Python, etc.

#ifndef _HILUCSI_UTILS_IO_HPP
#define _HILUCSI_UTILS_IO_HPP

#include <algorithm>
#include <complex>
#include <cstdint>
#include <fstream>
#include <tuple>
#include <type_traits>

#include "hilucsi/ds/Array.hpp"

namespace hilucsi {
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
/// \sa read_bin
template <class CsType>
inline typename std::enable_if<CsType::ROW_MAJOR>::type write_bin(
    const char *filename, const CsType &A, const std::uint64_t m) {
  using value_type                       = typename CsType::value_type;
  constexpr static std::size_t real_size = sizeof(value_type);
  constexpr static std::size_t int_size  = sizeof(typename CsType::index_type);
  constexpr static bool is_real = std::is_floating_point<value_type>::value;
  constexpr static bool is_double =
      (is_real && sizeof(value_type) == 8u) || sizeof(value_type) == 16u;

  std::ofstream f(filename, std::ios_base::binary);
  hilucsi_error_if(!f.is_open(), "failed to open file %s.", filename);
  internal::write_general_info(f, int_size, true, true, is_double, is_real);
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
/// \sa read_bin
template <class CsType>
inline typename std::enable_if<!CsType::ROW_MAJOR>::type write_bin(
    const char *filename, const CsType &A, const std::uint64_t m) {
  using value_type                       = typename CsType::value_type;
  constexpr static std::size_t real_size = sizeof(value_type);
  constexpr static std::size_t int_size  = sizeof(typename CsType::index_type);
  constexpr static bool is_real = std::is_floating_point<value_type>::value;
  constexpr static bool is_double =
      (is_real && sizeof(value_type) == 8u) || sizeof(value_type) == 16u;

  std::ofstream f(filename, std::ios_base::binary);
  hilucsi_error_if(!f.is_open(), "failed to open file %s.", filename);
  internal::write_general_info(f, int_size, false, true, is_double, is_real);
  const std::uint64_t nrows(A.nrows()), ncols(A.ncols()), nnz(A.nnz());
  internal::write_matrix_sizes(f, nrows, ncols, nnz, m);
  const void *ind_start = reinterpret_cast<const void *>(A.col_start().data());
  const void *indices   = reinterpret_cast<const void *>(A.row_ind().data());
  const void *vals      = reinterpret_cast<const void *>(A.vals().data());
  internal::write_matrix_data_attrs(f, int_size, real_size, nrows, nnz,
                                    ind_start, indices, vals);
  f.close();
}

/// \brief query information of a binary file
/// \param[in] filename binary filename
/// \return tuple of CRS/CCS order, C-index, double/float, is real, nrows,
///         ncols, nnz, m
/// \ingroup cpp
/// \sa query_info_ascii
inline std::tuple<bool, bool, bool, bool, std::uint64_t, std::uint64_t,
                  std::uint64_t, std::uint64_t>
query_info_bin(const char *filename) {
  std::ifstream f(filename, std::ios_base::binary);
  hilucsi_error_if(!f.is_open(), "failed to open file %s.", filename);

  bool        f_is_little;
  std::size_t f_word_size, f_int_size;
  bool        f_is_row, f_is_c, f_is_double, f_is_real;

  std::tie(f_is_little, f_word_size, f_int_size, f_is_row, f_is_c, f_is_double,
           f_is_real) = internal::read_general_info(f);

  const bool my_endianness  = internal::is_little_endian();
  const bool bad_endianness = f_is_little ^ my_endianness;
  hilucsi_error_if(bad_endianness,
                   "current system endianness does not match that of the file "
                   "%s. Endianness converting is not available for now..",
                   filename);

  // read sizes
  std::uint64_t nrows, ncols, nnz, m;
  std::tie(nrows, ncols, nnz, m) = internal::read_matrix_sizes(f);

  f.close();

  return std::make_tuple(f_is_row, f_is_c, f_is_double, f_is_real, nrows, ncols,
                         nnz, m);
}

// function prototype for ccs case

/// \brief read data from file to a \ref CCS matrix
/// \tparam CsType ccs type
/// \param[in] filename file name
/// \param[out] A matrix to store data from file
/// \sa write_bin
template <class CsType, typename T = std::uint64_t>
inline typename std::enable_if<!CsType::ROW_MAJOR, T>::type read_bin(
    const char *filename, CsType &A);

/// \brief read data from file to a \ref CRS matrix
/// \tparam CsType crs type
/// \param[in] filename file name
/// \param[out] A matrix to store data from file
/// \sa write_bin
template <class CsType, typename T = std::uint64_t>
inline typename std::enable_if<CsType::ROW_MAJOR, T>::type read_bin(
    const char *filename, CsType &A) {
  using value_type                       = typename CsType::value_type;
  constexpr static std::size_t real_size = sizeof(value_type);
  constexpr static std::size_t int_size  = sizeof(typename CsType::index_type);
  constexpr static bool is_real = std::is_floating_point<value_type>::value;
  constexpr static bool is_double =
      (is_real && sizeof(value_type) == 8u) || sizeof(value_type) == 16u;

  std::ifstream f(filename, std::ios_base::binary);
  hilucsi_error_if(!f.is_open(), "failed to open file %s.", filename);

  bool        f_is_little;
  std::size_t f_word_size, f_int_size;
  bool        f_is_row, f_is_c, f_is_double, f_is_real;

  std::tie(f_is_little, f_word_size, f_int_size, f_is_row, f_is_c, f_is_double,
           f_is_real) = internal::read_general_info(f);

  const bool my_endianness  = internal::is_little_endian();
  const bool bad_endianness = f_is_little ^ my_endianness;
  hilucsi_error_if(bad_endianness,
                   "current system endianness does not match that of the file "
                   "%s. Endianness converting is not available for now..",
                   filename);

  const bool bad_value_type = is_real ^ f_is_real;
  hilucsi_error_if(bad_value_type,
                   "inconsistent value data type detected, i.e. trying to read "
                   "real data to complex matrix or vice versa.");

  hilucsi_error_if(f_int_size > int_size,
                   "the integer size in file (%zd) is larger than the integer "
                   "size (%zd) used in the system; this is not allowed.",
                   f_int_size, int_size);

  const std::size_t f_real_size =
      f_is_real ? (f_is_double ? 8 : 4) : (f_is_double ? 16 : 8);
  hilucsi_warning_if(
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
    hilucsi_error_if(A.row_start().status() == DATA_UNDEF,
                     "memory allocation failed");
    A.reserve(nnz);
    hilucsi_error_if(
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
      hilucsi_error_if(ind_start.status() == DATA_UNDEF ||
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
    if (!f_is_c) {
      for (auto &v : A.row_start()) --v;
      for (auto &v : A.col_ind()) --v;
    }
    f.close();  // not needed but I'd like to have it
    hilucsi_error_if(A.nnz() != nnz, "inconsistent nnz");
    return m;
  }
  f.close();
  using ccs_type = typename CsType::other_type;
  ccs_type Ap;
  read_bin(filename, Ap);
  A = CsType(Ap);
  return m;
}

// impl of ccs case

template <class CsType, typename T>
inline typename std::enable_if<!CsType::ROW_MAJOR, T>::type read_bin(
    const char *filename, CsType &A) {
  using value_type                       = typename CsType::value_type;
  constexpr static std::size_t real_size = sizeof(value_type);
  constexpr static std::size_t int_size  = sizeof(typename CsType::index_type);
  constexpr static bool is_real = std::is_floating_point<value_type>::value;
  constexpr static bool is_double =
      (is_real && sizeof(value_type) == 8u) || sizeof(value_type) == 16u;

  std::ifstream f(filename, std::ios_base::binary);
  hilucsi_error_if(!f.is_open(), "failed to open file %s.", filename);

  bool        f_is_little;
  std::size_t f_word_size, f_int_size;
  bool        f_is_row, f_is_c, f_is_double, f_is_real;

  std::tie(f_is_little, f_word_size, f_int_size, f_is_row, f_is_c, f_is_double,
           f_is_real) = internal::read_general_info(f);

  const bool my_endianness  = internal::is_little_endian();
  const bool bad_endianness = f_is_little ^ my_endianness;
  hilucsi_error_if(bad_endianness,
                   "current system endianness does not match that of the file "
                   "%s. Endianness converting is not available for now..",
                   filename);

  const bool bad_value_type = is_real ^ f_is_real;
  hilucsi_error_if(bad_value_type,
                   "inconsistent value data type detected, i.e. trying to read "
                   "real data to complex matrix or vice versa.");

  hilucsi_error_if(f_int_size > int_size,
                   "the integer size in file (%zd) is larger than the integer "
                   "size (%zd) used in the system; this is not allowed.",
                   f_int_size, int_size);

  const std::size_t f_real_size =
      f_is_real ? (f_is_double ? 8 : 4) : (f_is_double ? 16 : 8);
  hilucsi_warning_if(
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
    hilucsi_error_if(A.col_start().status() == DATA_UNDEF,
                     "memory allocation failed");
    A.reserve(nnz);
    hilucsi_error_if(
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
      hilucsi_error_if(ind_start.status() == DATA_UNDEF ||
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
    if (!f_is_c) {
      for (auto &v : A.col_start()) --v;
      for (auto &v : A.row_ind()) --v;
    }
    f.close();
    hilucsi_error_if(A.nnz() != nnz, "inconsistent nnz");
    return m;
  }
  f.close();
  using crs_type = typename CsType::other_type;
  crs_type Ap;
  read_bin(filename, Ap);
  A = CsType(Ap);
  return m;
}

/// \brief write to ASCII file
/// \tparam IsRowMajor flag to indicate \ref CRS or \ref CCS
/// \tparam IndexArray index array type, see \ref Array
/// \tparma ValueArray value array type, see \ref Array
/// \param[in] fname file name for output
/// \param[in] ind_start index start array
/// \param[in] other_size column size for \ref CRS, row size for \ref CCS
/// \param[in] indices index array
/// \param[in] vals value array
/// \param[in] m leading symmetric block size
template <bool IsRowMajor, class IndexArray, class ValueArray>
inline void write_ascii(const char *fname, const IndexArray &ind_start,
                        const typename IndexArray::size_type other_size,
                        const IndexArray &indices, const ValueArray &vals,
                        const typename IndexArray::size_type m) {
  constexpr static char is_row    = IsRowMajor ? 'R' : 'C';
  constexpr static char is_c      = 'C';
  using value_type                = typename ValueArray::value_type;
  constexpr static int  data_size = sizeof(value_type);
  constexpr static char dtype     = std::is_floating_point<value_type>::value
                                    ? (data_size == 8 ? 'D' : 'S')
                                    : (data_size == 16 ? 'Z' : 'C');
  constexpr static int prec = dtype == 'D' || dtype == 'Z' ? 16 : 8;

  const static char info[4] = {is_row, is_c, dtype, '\0'};

  if (!ind_start.size())
    hilucsi_error("cannot write matrix with empty ind_start");

  decltype(other_size) nrows(IsRowMajor ? ind_start.size() - 1 : other_size),
      ncols(IsRowMajor ? other_size : ind_start.size() - 1),
      nnz(indices.size());

  hilucsi_error_if(vals.size() != nnz, "inconsistent nnz");
  hilucsi_error_if(nnz != decltype(nnz)(ind_start.back() - ind_start.front()),
                   "inconsistent nnz");

  std::ofstream f(fname);
  hilucsi_error_if(!f.is_open(), "cannot open file %s.", fname);

  f << info << '\n';
  f << nrows << ' ' << ncols << ' ' << nnz << ' ' << m << '\n';

  for (const auto v : ind_start) f << v << '\n';
  for (const auto v : indices) f << v << '\n';
  f.setf(std::ios::scientific);
  f.precision(prec);
  for (const auto v : vals) f << v << '\n';

  f.close();
}

/// \brief query information of an ASCII file
/// \param[in] filename binary filename
/// \return tuple of CRS/CCS order, C-index, double/float, is real, nrows,
///         ncols, nnz, m
/// \ingroup cpp
/// \sa query_info_bin
inline std::tuple<bool, bool, bool, bool, std::uint64_t, std::uint64_t,
                  std::uint64_t, std::uint64_t>
query_info_ascii(const char *filename) {
  const static char dtypes[5] = {'D', 'S', 'Z', 'C', '\0'};

  std::ifstream f(filename);
  hilucsi_error_if(!f.is_open(), "cannot open file %s.", filename);

  bool        is_row, is_c;
  char        d;
  std::size_t nrows, ncols, nnz, m;

  std::string buf;
  for (;;) {
    buf.clear();
    std::getline(f, buf);
    if (!buf.size() || buf[0] == '#') continue;
    break;
  }

  // buf should contain the information
  if (buf.size() != 3u) hilucsi_error("not a valid native hilucsi ascii file");
  if (buf[0] != 'R' && buf[0] != 'C')
    hilucsi_error(
        "the first char should be either R or C (row or column major)");

  // skip comments
  {
    int i = 0;
    for (; i < 4; ++i)
      if (buf.back() == dtypes[i]) break;
    if (i == 4) hilucsi_error("unknown data type");
  }

  is_row = buf.front() == 'R';
  is_c   = buf[1] == 'C';
  d      = buf.back();

  const bool is_double = d == 'D' || d == 'Z';
  const bool is_real   = d == 'D' || d == 'S';

  // read sizes
  f >> nrows >> ncols >> nnz >> m;

  return std::make_tuple(is_row, is_c, is_double, is_real, nrows, ncols, nnz,
                         m);
}

/// \brief read data from ASCII file
/// \tparam IndexArray index array type, see \ref Array
/// \tparma ValueArray value array type, see \ref Array
/// \param[in] fname file name for output
/// \param[in] ind_start index start array
/// \param[in] indices index array
/// \param[in] vals value array
template <class IndexArray, class ValueArray>
inline std::tuple<bool, bool, char, std::size_t, std::size_t, std::size_t,
                  std::size_t>
read_ascii(const char *fname, IndexArray &ind_start, IndexArray &indices,
           ValueArray &vals) {
  using value_type                = typename ValueArray::value_type;
  constexpr static bool is_real   = std::is_floating_point<value_type>::value;
  const static char     dtypes[5] = {'D', 'S', 'Z', 'C', '\0'};

  std::ifstream f(fname);
  hilucsi_error_if(!f.is_open(), "cannot open file %s.", fname);

  bool        is_row, is_c;
  char        d;
  std::size_t nrows, ncols, nnz, m;

  std::string buf;
  for (;;) {
    buf.clear();
    std::getline(f, buf);
    if (!buf.size() || buf[0] == '#') continue;
    break;
  }

  // buf should contain the information
  if (buf.size() != 3u) hilucsi_error("not a valid native hilucsi ascii file");
  if (buf[0] != 'R' && buf[0] != 'C')
    hilucsi_error(
        "the first char should be either R or C (row or column major)");
  if (buf[1] != 'C' && buf[1] != 'F')
    hilucsi_error(
        "the second char should be either C or F (c index or fortran index)");

  // skip comments
  {
    int i = 0;
    for (; i < 4; ++i)
      if (buf.back() == dtypes[i]) break;
    if (i == 4) hilucsi_error("unknown data type");
  }

  is_row = buf.front() == 'R';
  is_c   = buf[1] == 'C';
  d      = buf.back();

  if (is_real) {
    hilucsi_error_if(d == 'Z' || d == 'C',
                     "cannot load complex data to real array");
  } else {
    hilucsi_error_if(d == 'D' || d == 'S',
                     "cannot load real data to complex array");
  }

  // read sizes
  f >> nrows >> ncols >> nnz >> m;

#ifndef NDEBUG
  hilucsi_info("file %s has size attributes: %zd, %zd, %zd, %zd", fname, nrows,
               ncols, nnz, m);
#endif

  const std::size_t primary_size = is_row ? nrows + 1 : ncols + 1;
  ind_start.resize(primary_size);
  hilucsi_error_if(ind_start.status() == DATA_UNDEF,
                   "memory allocation failed");
  indices.resize(nnz);
  vals.resize(nnz);
  hilucsi_error_if(
      indices.status() == DATA_UNDEF || vals.status() == DATA_UNDEF,
      "memory allocation failed");

  for (std::size_t i = 0u; i < primary_size; ++i) f >> ind_start[i];
  for (std::size_t i = 0u; i < nnz; ++i) f >> indices[i];
  for (std::size_t i = 0u; i < nnz; ++i) f >> vals[i];

  if (!is_c) {
    for (auto &v : ind_start) --v;
    for (auto &v : indices) --v;
  }

  f.close();

  return std::make_tuple(is_row, is_c, d, nrows, ncols, nnz, m);
}

/*!
 * @}
 */ // group util

}  // namespace hilucsi

#endif  // _HILUCSI_UTILS_IO_HPP
