///////////////////////////////////////////////////////////////////////////////
//                  This file is part of the HIFIR library                   //
///////////////////////////////////////////////////////////////////////////////

/*!
 * \file hif/utils/io.hpp
 * \brief Read and write sparse matrices in HIFIR
 * \author Qiao Chen
 *
 * If HDF5 is enabled, then HIFIR is able to output sparse matrices into HDF5
 * files with four DataSets:
 *  - "sparse_info": length-four array of [nrows,ncols,isrowmajor,isreal]
 *    the data type is \a std::uint64_t
 *  - "ind_start": indptr array
 *  - "indices": index array
 *  - "vals": value array
 * Note that we use HDF5 C++ interface.
 *
 * In addition, we provide routines to handle the Matrix Market file format.
 * In particular, we support read all sparse matrices. For dense matrix, we
 * will treat it as an array for RHS, thus we only read the first column.

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

#ifndef _HIF_UTILS_IO_HPP
#define _HIF_UTILS_IO_HPP

#include <algorithm>
#include <cctype>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <tuple>
#include <type_traits>

#ifdef HIF_HAS_HDF5
#  include <H5Cpp.h>
#endif

#include "hif/ds/Array.hpp"
#include "hif/utils/math.hpp"
#include "hif/version.h"

namespace hif {

/*!
 * \addtogroup io
 * @{
 */

/// \brief write a \ref CRS matrix
/// \tparam CsType Crs type, see \ref CRS
/// \param[in] filename file name
/// \param[in] A matrix for writing to file
/// \sa read_bin
/// \note This function requires HDF5
template <class CsType>
inline void write_bin(const char *filename, const CsType &A) {
#ifdef HIF_HAS_HDF5
  using value_type  = typename CsType::value_type;
  using index_type  = typename CsType::index_type;
  using indptr_type = typename CsType::indptr_type;
  using scalar_type = typename ValueTypeTrait<value_type>::value_type;

#  ifndef H5_NO_NAMESPACE
  using namespace H5;
#  endif

  static_assert(std::is_floating_point<scalar_type>::value,
                "must be scalar type");
  static_assert(std::is_integral<index_type>::value &&
                    std::is_integral<indptr_type>::value,
                "must be integer types");
  static_assert(sizeof(index_type) >= 4ul && sizeof(indptr_type) >= 4u,
                "must be at least 32bit integers");

  std::uint64_t info[4];
  info[0] = A.nrows();  // nrows
  info[1] = A.ncols();  // ncols
  info[2] = CsType::ROW_MAJOR;
  info[3] = std::is_floating_point<value_type>::value;

  try {
    Exception::dontPrint();
    H5File f(H5std_string(filename), H5F_ACC_TRUNC);
    // write information
    hsize_t   n = 4;
    DataSpace infosp(1, &n);
    auto      ds_info = f.createDataSet(H5std_string("sparse_info"),
                                   PredType::NATIVE_UINT64, infosp);
    ds_info.write((const void *)info, PredType::NATIVE_UINT64);
    // write ind_start
    const void *ind_start = (const void *)A.ind_start().data();
    n                     = A.ind_start().size();
    DataSpace ind_start_sp(1, &n);
    if (std::is_same<indptr_type, long long>::value) {
      const auto &type = PredType::NATIVE_LLONG;
      auto        ds_indptr =
          f.createDataSet(H5std_string("ind_start"), type, ind_start_sp);
      ds_indptr.write(ind_start, type);
    } else if (std::is_same<indptr_type, unsigned long long>::value) {
      const auto &type = PredType::NATIVE_ULLONG;
      auto        ds_indptr =
          f.createDataSet(H5std_string("ind_start"), type, ind_start_sp);
      ds_indptr.write(ind_start, type);
    } else if (std::is_same<indptr_type, long>::value) {
      const auto &type = PredType::NATIVE_LONG;
      auto        ds_indptr =
          f.createDataSet(H5std_string("ind_start"), type, ind_start_sp);
      ds_indptr.write(ind_start, type);
    } else if (std::is_same<indptr_type, unsigned long>::value) {
      const auto &type = PredType::NATIVE_ULONG;
      auto        ds_indptr =
          f.createDataSet(H5std_string("ind_start"), type, ind_start_sp);
      ds_indptr.write(ind_start, type);
    } else if (std::is_same<indptr_type, int>::value) {
      const auto &type = PredType::NATIVE_INT;
      auto        ds_indptr =
          f.createDataSet(H5std_string("ind_start"), type, ind_start_sp);
      ds_indptr.write(ind_start, type);
    } else if (std::is_same<indptr_type, unsigned int>::value) {
      const auto &type = PredType::NATIVE_UINT;
      auto        ds_indptr =
          f.createDataSet(H5std_string("ind_start"), type, ind_start_sp);
      ds_indptr.write(ind_start, type);
    } else
      hif_error("unsupported integer type");
    // write indices
    n = A.inds().size();
    DataSpace   nnz_sp(1, &n);
    const void *inds = (const void *)A.inds().data();
    if (std::is_same<index_type, long long>::value) {
      const auto &type = PredType::NATIVE_LLONG;
      auto ds_inds     = f.createDataSet(H5std_string("indices"), type, nnz_sp);
      ds_inds.write(inds, type);
    } else if (std::is_same<index_type, unsigned long long>::value) {
      const auto &type = PredType::NATIVE_ULLONG;
      auto ds_inds     = f.createDataSet(H5std_string("indices"), type, nnz_sp);
      ds_inds.write(inds, type);
    } else if (std::is_same<index_type, long>::value) {
      const auto &type = PredType::NATIVE_LONG;
      auto ds_inds     = f.createDataSet(H5std_string("indices"), type, nnz_sp);
      ds_inds.write(inds, type);
    } else if (std::is_same<index_type, unsigned long>::value) {
      const auto &type = PredType::NATIVE_ULONG;
      auto ds_inds     = f.createDataSet(H5std_string("indices"), type, nnz_sp);
      ds_inds.write(inds, type);
    } else if (std::is_same<index_type, int>::value) {
      const auto &type = PredType::NATIVE_INT;
      auto ds_inds     = f.createDataSet(H5std_string("indices"), type, nnz_sp);
      ds_inds.write(inds, type);
    } else if (std::is_same<index_type, unsigned int>::value) {
      const auto &type = PredType::NATIVE_UINT;
      auto ds_inds     = f.createDataSet(H5std_string("indices"), type, nnz_sp);
      ds_inds.write(inds, type);
    } else
      hif_error("unsupported integer type");
    // write values
    const void *vals = (const void *)A.vals().data();
    n                = A.vals().size() * (info[3] == 0u ? 2 : 1);
    DataSpace nnz_sp2(1, &n);
    if (std::is_same<scalar_type, double>::value) {
      const auto &type = PredType::NATIVE_DOUBLE;
      auto ds_vals     = f.createDataSet(H5std_string("vals"), type, nnz_sp2);
      ds_vals.write(vals, type);
    } else if (std::is_same<scalar_type, float>::value) {
      const auto &type = PredType::NATIVE_FLOAT;
      auto ds_vals     = f.createDataSet(H5std_string("vals"), type, nnz_sp2);
      ds_vals.write(vals, type);
    } else {
      const auto &type = PredType::NATIVE_LDOUBLE;
      auto ds_vals     = f.createDataSet(H5std_string("vals"), type, nnz_sp2);
      ds_vals.write(vals, type);
    }
  } catch (const Exception &h5e) {
    hif_error("HDF5 returned error from func:%s with message: %s",
              h5e.getCFuncName(), h5e.getCDetailMsg());
  } catch (...) {
    throw;
  }
#else
  (void)filename;
  (void)A;
  hif_error("write_bin requires HDF5");
#endif
}

/// \brief read data from file to a \ref CRS matrix
/// \tparam IndPtrArray indptr array
/// \tparam IndexArray index array type
/// \tparam ValueArray value array type
/// \param[in] filename file name
/// \param[out] info sparse matrix information
/// \param[out] ind_start indptr array
/// \param[out] indices index array
/// \param[out] vals value array
/// \sa write_bin
/// \note This function requires HDF5
template <class IndPtrArray, class IndexArray, class ValueArray>
inline void read_bin(const char *filename, std::uint64_t info[],
                     IndPtrArray &ind_start, IndexArray &indices,
                     ValueArray &vals) {
#ifdef HIF_HAS_HDF5
  using value_type  = typename ValueArray::value_type;
  using index_type  = typename IndexArray::value_type;
  using indptr_type = typename IndPtrArray::value_type;
  using scalar_type = typename ValueTypeTrait<value_type>::value_type;

#  ifndef H5_NO_NAMESPACE
  using namespace H5;
#  endif

  try {
    Exception::dontPrint();
    H5File f(H5std_string(filename), H5F_ACC_RDONLY);
    f.openDataSet(H5std_string("sparse_info"))
        .read((void *)info, PredType::NATIVE_UINT64);
    if ((std::is_floating_point<value_type>::value && (info[3] == 0u)) ||
        (!std::is_floating_point<value_type>::value && (info[3] == 1u)))
      hif_error("cannot mixed real and complex in read_bin");
    if (info[2])
      ind_start.resize(info[0] + 1);
    else
      ind_start.resize(info[1] + 1);
    auto ds_indptr = f.openDataSet(H5std_string("ind_start"));
    if (sizeof(indptr_type) == ds_indptr.getDataType().getSize())
      ds_indptr.read((void *)ind_start.data(), ds_indptr.getDataType());
    else if (ds_indptr.getDataType().getSize() == 4) {
      Array<int> buf(ind_start.size());
      ds_indptr.read((void *)buf.data(), ds_indptr.getDataType());
      std::copy(buf.cbegin(), buf.cend(), ind_start.begin());
    } else {
      Array<long long> buf(ind_start.size());
      ds_indptr.read((void *)buf.data(), ds_indptr.getDataType());
      std::copy(buf.cbegin(), buf.cend(), ind_start.begin());
    }
    auto ds_inds = f.openDataSet(H5std_string("indices"));
    auto n       = ds_inds.getSpace().getSelectNpoints();
    indices.resize(n);
    if (sizeof(index_type) == ds_inds.getDataType().getSize())
      ds_inds.read((void *)indices.data(), ds_inds.getDataType());
    else if (ds_inds.getDataType().getSize() == 4) {
      Array<int> buf(indices.size());
      ds_inds.read((void *)buf.data(), ds_inds.getDataType());
      std::copy(buf.cbegin(), buf.cend(), indices.begin());
    } else {
      Array<long long> buf(indices.size());
      ds_inds.read((void *)buf.data(), ds_inds.getDataType());
      std::copy(buf.cbegin(), buf.cend(), indices.begin());
    }
    auto ds_vals = f.openDataSet(H5std_string("vals"));
    n            = ds_vals.getSpace().getSelectNpoints();
    if (info[3] == 0u)
      vals.resize(n / 2);  // complex
    else
      vals.resize(n);
    const int fac = info[3] == 0u ? 2 : 1;
    if (sizeof(value_type) == ds_vals.getDataType().getSize() * fac)
      ds_vals.read((void *)vals.data(), ds_vals.getDataType());
    else if (ds_vals.getDataType().getSize() == 4) {
      Array<float> buf(n);
      ds_vals.read((void *)buf.data(), ds_vals.getDataType());
      std::copy_n(buf.data(), n, (scalar_type *)vals.data());
    } else if (ds_vals.getDataType().getSize() == 8) {
      Array<double> buf(n);
      ds_vals.read((void *)buf.data(), ds_vals.getDataType());
      std::copy_n(buf.data(), n, (scalar_type *)vals.data());
    } else {
      Array<long double> buf(n);
      ds_vals.read((void *)buf.data(), ds_vals.getDataType());
      std::copy_n(buf.data(), n, (scalar_type *)vals.data());
    }
  } catch (const Exception &h5e) {
    hif_error("HDF5 returned error from func:%s with message: %s",
              h5e.getCFuncName(), h5e.getCDetailMsg());
  } catch (...) {
    throw;
  }
#else
  (void)filename;
  (void)info;
  (void)ind_start;
  (void)indices;
  (void)vals;
  hif_error("read_bin requires HDF5");
#endif
}

#if 0

/// \brief write to ASCII file
/// \tparam IsRowMajor flag to indicate \ref CRS or \ref CCS
/// \tparam IndexArray index array type, see \ref Array
/// \tparam ValueArray value array type, see \ref Array
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

  if (!ind_start.size()) hif_error("cannot write matrix with empty ind_start");

  decltype(other_size) nrows(IsRowMajor ? ind_start.size() - 1 : other_size),
      ncols(IsRowMajor ? other_size : ind_start.size() - 1),
      nnz(indices.size());

  hif_error_if(vals.size() != nnz, "inconsistent nnz");
  hif_error_if(nnz != decltype(nnz)(ind_start.back() - ind_start.front()),
               "inconsistent nnz");

  std::ofstream f(fname);
  hif_error_if(!f.is_open(), "cannot open file %s.", fname);

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
/// \ingroup itr
/// \sa query_info_bin
inline std::tuple<bool, bool, bool, bool, std::uint64_t, std::uint64_t,
                  std::uint64_t, std::uint64_t>
query_info_ascii(const char *filename) {
  const static char dtypes[5] = {'D', 'S', 'Z', 'C', '\0'};

  std::ifstream f(filename);
  hif_error_if(!f.is_open(), "cannot open file %s.", filename);

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
  if (buf.size() != 3u) hif_error("not a valid native hif ascii file");
  if (buf[0] != 'R' && buf[0] != 'C')
    hif_error("the first char should be either R or C (row or column major)");

  // skip comments
  {
    int i = 0;
    for (; i < 4; ++i)
      if (buf.back() == dtypes[i]) break;
    if (i == 4) hif_error("unknown data type");
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
/// \tparam ValueArray value array type, see \ref Array
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
  hif_error_if(!f.is_open(), "cannot open file %s.", fname);

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
  if (buf.size() != 3u) hif_error("not a valid native hif ascii file");
  if (buf[0] != 'R' && buf[0] != 'C')
    hif_error("the first char should be either R or C (row or column major)");
  if (buf[1] != 'C' && buf[1] != 'F')
    hif_error(
        "the second char should be either C or F (c index or fortran index)");

  // skip comments
  {
    int i = 0;
    for (; i < 4; ++i)
      if (buf.back() == dtypes[i]) break;
    if (i == 4) hif_error("unknown data type");
  }

  is_row = buf.front() == 'R';
  is_c   = buf[1] == 'C';
  d      = buf.back();

  if (is_real) {
    hif_error_if(d == 'Z' || d == 'C',
                 "cannot load complex data to real array");
  } else {
    hif_error_if(d == 'D' || d == 'S',
                 "cannot load real data to complex array");
  }

  // read sizes
  f >> nrows >> ncols >> nnz >> m;

#  ifdef HIF_DEBUG
  hif_info("file %s has size attributes: %zd, %zd, %zd, %zd", fname, nrows,
           ncols, nnz, m);
#  endif

  const std::size_t primary_size = is_row ? nrows + 1 : ncols + 1;
  ind_start.resize(primary_size);
  hif_error_if(ind_start.status() == DATA_UNDEF, "memory allocation failed");
  indices.resize(nnz);
  vals.resize(nnz);
  hif_error_if(indices.status() == DATA_UNDEF || vals.status() == DATA_UNDEF,
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

#endif

/*!
 * @}
 */ // group io

// Matrix Market

namespace internal {

#ifndef DOXYGEN_SHOULD_SKIP_THIS
// helper to handle complex and real

template <class V>
inline typename std::enable_if<std::is_floating_point<V>::value>::type set_real(
    const double r, V &x) {
  x = r;
}

template <class V>
inline typename std::enable_if<!std::is_floating_point<V>::value>::type
set_real(const double r, V &x) {
  // assume V is std::complex
  x.real(r);
}

template <class V>
inline typename std::enable_if<std::is_floating_point<V>::value>::type set_imag(
    const double, V &) {}

template <class V>
inline typename std::enable_if<!std::is_floating_point<V>::value>::type
set_imag(const double i, V &x) {
  // assume V is std::complex
  x.imag(i);
}
#endif  // DOXYGEN_SHOULD_SKIP_THIS

/*!
 * \addtogroup io
 * @{
 */

/// \brief Internal routine to read sparse data
template <int TypeID, bool IsReal, class IndexType, class ValueType>
inline void mm_read_sparse_data(std::FILE *f, const std::size_t nnz,
                                Array<IndexType> &rows, Array<IndexType> &cols,
                                Array<ValueType> &vals) {
  // type IDs
  static constexpr int GENERAL = 0, SYMM = 1, HERM = 2, SK_SYMM = 3;
  static_assert(TypeID == GENERAL || TypeID == SYMM || TypeID == HERM ||
                    TypeID == SK_SYMM,
                "invalid TypeID");

  if (TypeID != GENERAL) {
    // shrink at the end if necessary
    rows.resize(nnz * 2);
    cols.resize(nnz * 2);
    vals.resize(nnz * 2);
  } else {
    rows.resize(nnz);
    cols.resize(nnz);
    vals.resize(nnz);
  }

  std::size_t index(0);
  int         r, c;
  double      vr, vi;
  for (std::size_t i(0); i < nnz; ++i) {
    const int flag = IsReal ? std::fscanf(f, "%d %d %lg", &r, &c, &vr)
                            : std::fscanf(f, "%d %d %lg %lg", &r, &c, &vr, &vi);
    if (flag != 4 - IsReal) {
      std::fclose(f);
      hif_error("insufficient objects scanned from fscanf");
    }
    rows[index] = r - 1;  // to 0-based index
    cols[index] = c - 1;  // to 0-based index
    set_real(vr, vals[index]);
    set_imag(vi, vals[index]);
    ++index;
    if (TypeID == GENERAL || r == c) continue;
    // not general and not along the diagonal
    rows[index] = c - 1;  // to 0-based index
    cols[index] = r - 1;  // to 0-based index
    if (TypeID == SYMM)
      vals[index] = vals[index - 1];
    else if (TypeID == HERM)
      vals[index] = conjugate(vals[index - 1]);
    else
      vals[index] = -vals[index - 1];
    ++index;
  }
  if (index != nnz) {
    rows.resize(index);
    cols.resize(index);
    vals.resize(index);
  }
}

/// \brief Internal routine to read vector data
template <bool IsReal, class ValueType>
inline void mm_read_vector_data(std::FILE *f, const std::size_t n,
                                Array<ValueType> &v) {
  v.resize(n);
  double vr, vi;
  for (std::size_t i(0); i < n; ++i) {
    const int flag = IsReal ? std::fscanf(f, "%lg", &vr)
                            : std::fscanf(f, "%lg %lg", &vr, &vi);
    if (flag != 2 - IsReal) {
      std::fclose(f);
      hif_error("insufficient objects scanned from fscanf");
    }
    set_real(vr, v[i]);
    set_imag(vi, v[i]);
  }
}

/// \brief Internal routine to parse the first line
inline void mm_read_firstline(std::FILE *f, bool &is_sparse, bool &is_real,
                              int &type_id) {
  char line[1025];
  char banner[65];
  char mtx[65];
  char crd[65];
  char data_type[65];
  char storage_scheme[65];

  // read the first line
  if (!std::fgets(line, 1025, f)) hif_error("unable to read the first line");
  if (std::sscanf(line, "%s %s %s %s %s", banner, mtx, crd, data_type,
                  storage_scheme) != 5) {
    std::fclose(f);
    hif_error("insufficient number of tokens in the first line");
  }
  std::string banner_str(banner);
  if (banner_str != std::string("%%MatrixMarket")) {
    std::fclose(f);
    hif_error("incorrect banner");
  }

  std::string mtx_str(mtx);
  // convert to lower
  auto to_lower = [](std::string &ss) {
    std::transform(ss.begin(), ss.end(), ss.begin(),
                   [](unsigned char c) { return std::tolower(c); });
  };
  to_lower(mtx_str);
  if (mtx_str != std::string("matrix")) {
    std::fclose(f);
    hif_error("incorrect mtx");
  }

  std::string crd_str(crd), data_type_str(data_type),
      storage_scheme_str(storage_scheme);
  to_lower(crd_str);
  to_lower(data_type_str);
  to_lower(storage_scheme_str);

  if (crd_str != std::string("coordinate") && crd_str != std::string("array")) {
    std::fclose(f);
    hif_error("unknown crd %s", crd_str.c_str());
  }
  is_sparse = crd_str != std::string("array");

  if (data_type_str != std::string("real") &&
      data_type_str != std::string("complex")) {
    std::fclose(f);
    hif_error("HIFIR unsupported data type %s", data_type_str.c_str());
  }
  is_real = data_type_str != std::string("complex");

  if (storage_scheme_str == std::string("general"))
    type_id = 0;
  else if (storage_scheme_str == std::string("symmetric"))
    type_id = 1;
  else if (storage_scheme_str == std::string("hermitian"))
    type_id = 2;
  else if (storage_scheme_str == std::string("skew-symmetric"))
    type_id = 3;
  else {
    std::fclose(f);
    hif_error("unknown storage_scheme %s", storage_scheme_str.c_str());
  }
}

/// \brief Internal routine to read sizes
inline void mm_read_sparse_size(std::FILE *f, std::size_t &nrows,
                                std::size_t &ncols, std::size_t &nnz) {
  char line[1025];

  nrows = ncols = nnz = 0u;

  /* now continue scanning until you reach the end-of-comments */
  do {
    if (!std::fgets(line, 1025, f)) {
      std::fclose(f);
      hif_error("unable to read file");
    }
  } while (line[0] == '%');

  int m, n, nz;
  if (sscanf(line, "%d %d %d", &m, &n, &nz) == 3) {
    nrows = m;
    ncols = n;
    nnz   = nz;
    return;
  }

  int num_items_read;
  do {
    num_items_read = std::fscanf(f, "%d %d %d", &m, &n, &nz);
    if (num_items_read == EOF) {
      std::fclose(f);
      hif_error("reached EOF!");
    }
  } while (num_items_read != 3);
  nrows = m;
  ncols = n;
  nnz   = nz;
}

inline void mm_read_dense_size(std::FILE *f, std::size_t &nrows,
                               std::size_t &ncols) {
  char line[1025];

  nrows = ncols = 0u;

  /* now continue scanning until you reach the end-of-comments */
  do {
    if (!std::fgets(line, 1025, f)) {
      std::fclose(f);
      hif_error("unable to read file");
    }
  } while (line[0] == '%');

  int m, n;
  if (sscanf(line, "%d %d", &m, &n) == 2) {
    nrows = m;
    ncols = n;
    return;
  }

  int num_items_read;
  do {
    num_items_read = std::fscanf(f, "%d %d", &m, &n);
    if (num_items_read == EOF) {
      std::fclose(f);
      hif_error("reached EOF!");
    }
  } while (num_items_read != 2);
  nrows = m;
  ncols = n;
}

/*!
 * @}
 */ // group io

}  // namespace internal

/*!
 * \addtogroup io
 * @{
 */

/// \brief Read a sparse matrix form a MatrixMarket file
/// \tparam IndexType Index type
/// \tparam ValueType Value type
/// \param[in] filename File name
/// \param[out] nrows Output number of rows
/// \param[out] ncols Output number of columns
/// \param[out] rows Output row indices
/// \param[out] cols Output column indices
/// \param[out] vals Output values
///
/// Regardless what storage does \a filename uses, we always output a general
/// matrix. Also, note that matrix market use coordinate format (aka IJV),
/// thus if you plan to use this routine directly, then you need to convert
/// the coordinate data structures into CRS/CCS if necessary.
template <class IndexType, class ValueType>
inline void read_mm_sparse(const char *filename, std::size_t &nrows,
                           std::size_t &ncols, Array<IndexType> &rows,
                           Array<IndexType> &cols, Array<ValueType> &vals) {
  // open the file
  std::FILE *f = std::fopen(filename, "r");
  hif_error_if(!f, "unable to open matrix market file %s", filename);
  bool is_sparse, is_real;
  int  type_id;
  // parse the first line
  internal::mm_read_firstline(f, is_sparse, is_real, type_id);
  if (!is_sparse) {
    std::fclose(f);
    hif_error("must be sparse matrix");
  }
  if (std::is_floating_point<ValueType>::value ^ is_real) {
    // XOR
    std::fclose(f);
    hif_error("data types did not match between array and file");
  }
  std::size_t nnz;
  internal::mm_read_sparse_size(f, nrows, ncols, nnz);
  // read data
  if (is_real) {
    if (type_id == 0)
      internal::mm_read_sparse_data<0, true>(f, nnz, rows, cols, vals);
    else if (type_id == 1)
      internal::mm_read_sparse_data<1, true>(f, nnz, rows, cols, vals);
    else if (type_id == 2)
      internal::mm_read_sparse_data<2, true>(f, nnz, rows, cols, vals);
    else
      internal::mm_read_sparse_data<3, true>(f, nnz, rows, cols, vals);
  } else {
    if (type_id == 0)
      internal::mm_read_sparse_data<0, false>(f, nnz, rows, cols, vals);
    else if (type_id == 1)
      internal::mm_read_sparse_data<1, false>(f, nnz, rows, cols, vals);
    else if (type_id == 2)
      internal::mm_read_sparse_data<2, false>(f, nnz, rows, cols, vals);
    else
      internal::mm_read_sparse_data<3, false>(f, nnz, rows, cols, vals);
  }
  std::fclose(f);
}

/// \brief Read vector from a MatrixMarket file
/// \tparam ValueType Value type
/// \param[in] filename File name
/// \param[out] v Output vector
/// \note We only read the first column is the file contains multiple columns
/// \sa read_mm_sparse
template <class ValueType>
inline void read_mm_vector(const char *filename, Array<ValueType> &v) {
  // open the file
  std::FILE *f = std::fopen(filename, "r");
  hif_error_if(!f, "unable to open matrix market file %s", filename);
  bool is_sparse, is_real;
  int  type_id;
  // parse the first line
  internal::mm_read_firstline(f, is_sparse, is_real, type_id);
  if (is_sparse) {
    std::fclose(f);
    hif_error("must be dense matrix");
  }
  if (std::is_floating_point<ValueType>::value ^ is_real) {
    // XOR
    std::fclose(f);
    hif_error("data types did not match between array and file");
  }
  std::size_t m, n;
  internal::mm_read_dense_size(f, m, n);

  if (is_real)
    internal::mm_read_vector_data<true>(f, m, v);
  else
    internal::mm_read_vector_data<false>(f, m, v);

  std::fclose(f);
}

/// \brief Write a float-point vector to a MatrixMarket file
/// \sa read_mm_vector
template <class ValueType>
inline void write_mm_vector(const char *filename, const Array<ValueType> &v) {
  // only for floating numbers
  static constexpr bool IS_REAL   = std::is_floating_point<ValueType>::value;
  static const char *   DATA_TYPE = IS_REAL ? "real" : "complex";
  static constexpr int  PREC =
      (IS_REAL && sizeof(ValueType) == sizeof(double)) ||
              (!IS_REAL && sizeof(ValueType) == sizeof(std::complex<double>))
          ? 16
          : 8;

  std::ofstream f(filename);
  hif_error_if(!f.is_open(), "cannot open file %s", filename);
  f << "%%MatrixMarket matrix array " << DATA_TYPE << " general\n";
  f << "% Written by HIFIR " << HIF_GLOBAL_VERSION << '.' << HIF_MAJOR_VERSION
    << '.' << HIF_MINOR_VERSION << '\n';
  f << ' ' << v.size() << " 1\n";
  f.setf(std::ios::scientific);
  f.precision(PREC);
  for (const auto val : v) {
    f << real(val);
    if (!IS_REAL) f << '\t' << imag(val);
    f << '\n';
  }
  f.close();
}

/// \brief Write a sparse matrix
///
/// \tparam IsRowMajor Boolean tag for row-major (CRS) or not (CCS)
/// \tparam IndptrType Index pionter type
/// \tparam IndexType Index type
/// \tparam ValueType Value type
/// \param[in] filename Output file name
/// \param[in] other_size The size of the uncompressed axis, e.g., ncols for CRS
/// \param[in] ind_start Index pionter
/// \param[in] indices Index list
/// \param[in] vals Data values
///
/// For CRS, call this function as write_mm_sparse<true>("test.mm", ncols,
/// rowptr, colind, vals). Similarly, for CCS, call this function as
/// write_mm_sparse<false>("test.mm", nrows, colptr, rowind, vals).
template <bool IsRowMajor, class IndptrType, class IndexType, class ValueType>
inline void write_mm_sparse(const char *filename, const std::size_t other_size,
                            const Array<IndptrType> &ind_start,
                            const Array<IndexType> & indices,
                            const Array<ValueType> & vals) {
  static constexpr bool IS_REAL   = std::is_floating_point<ValueType>::value;
  static const char *   DATA_TYPE = IS_REAL ? "real" : "complex";
  static constexpr int  PREC =
      (IS_REAL && sizeof(ValueType) == sizeof(double)) ||
              (!IS_REAL && sizeof(ValueType) == sizeof(std::complex<double>))
          ? 16
          : 8;

  if (ind_start.size() == 0u) hif_error("cannot write empty matrix");
  std::ofstream f(filename);
  hif_error_if(!f.is_open(), "cannot open file %s", filename);
  f << "%%MatrixMarket matrix coordinate " << DATA_TYPE << " general\n";
  f << "% Written by HIFIR " << HIF_GLOBAL_VERSION << '.' << HIF_MAJOR_VERSION
    << '.' << HIF_MINOR_VERSION << '\n';
  const std::size_t primary_size = ind_start.size() - 1;
  f << ' ' << (IsRowMajor ? primary_size : other_size) << ' '
    << (IsRowMajor ? other_size : primary_size) << ' ' << vals.size() << '\n';
  f.setf(std::ios::scientific);
  f.precision(PREC);
  for (std::size_t i(0); i < primary_size; ++i) {
    for (auto k = ind_start[i]; k < ind_start[i + 1]; ++k) {
      const auto index = indices[k] + 1;  // to fortran index
      const auto val   = vals[k];
      f << (IsRowMajor ? static_cast<IndexType>(i + 1) : index) << '\t'
        << (IsRowMajor ? index : static_cast<IndexType>(i + 1)) << '\t'
        << real(val);
      if (!IS_REAL) f << '\t' << imag(val);
      f << '\n';
    }
  }
  f.close();
}

/*!
 * @}
 */ // group io

}  // namespace hif

#endif  // _HIF_UTILS_IO_HPP
