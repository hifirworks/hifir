//@HEADER
//----------------------------------------------------------------------------
//                Copyright (C) 2019 The PSMILU AUTHORS
//----------------------------------------------------------------------------
//@HEADER

/// \file psmilu_matrix_market.hpp
/// \brief Read data from mm format
/// \authors Qiao,

#ifndef _PSMILU_MATRIXMARKET_HPP
#define _PSMILU_MATRIXMARKET_HPP

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

#ifndef DOXYGEN_SHOULD_SKIP_THIS
#  define __BUF_SIZE__ 512
#endif  // DOXYGEN_SHOULD_SKIP_THIS

#include "psmilu_log.hpp"

namespace psmilu {
namespace internal {
/// \class GeneralLineReader
/// \brief Read general file, i.e. ijv
/// \ingroup util
class GeneralLineReader {
 public:
  GeneralLineReader() = delete;

  /// \brief constructor with streamer
  /// \param[in] s input file streamer
  GeneralLineReader(std::ifstream &s) : _s(&s) {}

  /// \brief parse a line with ijv
  /// \param[out] i row index
  /// \param[out] j column index
  /// \param[out] v value
  inline bool parse_line(std::size_t &i, std::size_t &j, double &v) const {
    _s->getline(_buf, __BUF_SIZE__);
    if (*_s) {
      std::sscanf(_buf, "%zd %zd %lg\n", &i, &j, &v);
      return false;
    }
    return true;
  }

 private:
  mutable std::ifstream *_s;                  ///< input file streamer
  mutable char           _buf[__BUF_SIZE__];  ///< buffer
};

/// \class PatternLineReader
/// \brief read only the patterns, i.e. ij
/// \ingroup util
class PatternLineReader {
 public:
  PatternLineReader() = delete;

  /// \brief constructor with streamer
  /// \param[in] s input file streamer
  PatternLineReader(std::ifstream &s) : _s(&s) {}

  /// \brief parse a line with ij
  /// \param[out] i row index
  /// \param[out] j column index
  inline bool parse_line(std::size_t &i, std::size_t &j, double &) const {
    _s->getline(_buf, __BUF_SIZE__);
    if (*_s) {
      std::sscanf(_buf, "%zd %zd\n", &i, &j);
      return false;
    }
    return true;
  }

 private:
  mutable std::ifstream *_s;                  ///< input file streamer
  mutable char           _buf[__BUF_SIZE__];  ///< buffer
};

/// \brief handy indices and value pair
/// \tparam ValueType value type, e.g. \a double, \a float, etc
/// \tparam IndexType index type, e.g. \a int, \a long, etc
/// \tparam RowMajor if \a true, then CRS is assumed
/// \ingroup util
template <class ValueType, class IndexType, bool RowMajor>
struct IJV {
  /// \brief constructor with ijv triplet
  /// \param[in] i row index
  /// \param[in] j column index
  /// \param[in] v corresponding value
  IJV(const std::size_t i, const std::size_t j, const double v)
      : row(static_cast<IndexType>(i)),
        col(static_cast<IndexType>(j)),
        val(static_cast<ValueType>(v)) {}

  /// \brief less operator for sorting
  /// \param[in] rhs right-hand triplet
  ///
  /// This comparison is implemented so that the resulting array will be
  /// sorted in primary direction and as well as each of the local ranges.
  /// For instance, for CRS matrices, the triplet array will be sorted
  /// first based on row indices, within a local range of same row index, it,
  /// then, will be sorted based on column indices.
  inline bool operator<(const IJV &rhs) const {
    return RowMajor ? row < rhs.row || (row == rhs.row && col < rhs.col)
                    : col < rhs.col || (col == rhs.col && row < rhs.row);
  }
  inline IJV &operator--() {
    --row;
    --col;
    return *this;
  }
  inline bool operator==(const IndexType entry) const {
    return RowMajor ? entry == row : entry == col;
  }

  /// \brief get the index, for CRS, its column index while row for CCS
  inline IndexType idx() const { return RowMajor ? col : row; }
  IndexType        row, col;
  ValueType        val;
};

/// \brief read matrix market kernel, i.e. data lines
/// \tparam ValueArray value array type
/// \tparam IndexArray index array type
/// \tparam OneBased if \a true, then Fortran index is assumed
/// \tparam RowMajor if \a true, then CRS is assumed
/// \tparam LineReader kernel reader type
/// \param[in] nnz total number of nonzeros
/// \param[in] n number of entries in primary direction
/// \param[in] reader kernel reader for ijv triplet
/// \param[out] ind_start starting positions
/// \param[out] indices index array
/// \param[out] vals value array
/// \ingroup util
template <class ValueArray, class IndexArray, bool OneBased, bool RowMajor,
          class LineReader>
void read_mm_kernel(const std::size_t nnz, const std::size_t n,
                    const LineReader &reader, IndexArray &ind_start,
                    IndexArray &indices, ValueArray &vals) {
  typedef typename ValueArray::value_type       value_type;
  typedef typename IndexArray::value_type       index_type;
  typedef IJV<value_type, index_type, RowMajor> ijv_type;

  vals.resize(nnz);
  indices.resize(nnz);
  ind_start.resize(n + 1);
  ind_start.front() = static_cast<index_type>(OneBased);
  // create local buffers
  double      v;
  std::size_t i, j;

  // create work space
  std::vector<ijv_type> ijv;
  ijv.reserve(nnz);

  for (std::size_t k = 0u; k < nnz; ++k) {
    psmilu_error_if(reader.parse_line(i, j, v),
                    "EOF detected without finishing reading %zd nnz", nnz);
    ijv.emplace_back(i, j, v);
  }

  !OneBased ? std::for_each(ijv.begin(), ijv.end(),
                            [](ijv_type &triplet) { --triplet; })
            : (void)0;

  // sort indices
  std::sort(ijv.begin(), ijv.end());
  auto itr = ijv.cbegin();
  j        = 0u;  // reset j
  // use i for outer loop index
  for (i = 0u; i < n; ++i) {
    const std::size_t entry = i + OneBased;
    auto              itr0  = itr;
    for (;; ++itr)
      if (*itr == entry) continue;
    const auto nnz   = itr - itr0;
    ind_start[i + 1] = ind_start[i] + nnz;
    for (; itr0 != itr; ++itr0, ++j) {
      indices[j] = itr0->idx();
      vals[j]    = itr0->val;
    }
  }
}

/// \brief read the file header
/// \param[in,out] s input file streamer
/// \param[out] field tag indicating the file type
/// \param[out] row number of rows
/// \param[out] col number of columns
/// \param[out] nnz total number of nonzeros
/// \warning Only sparse general matrices are supported!
/// \ingroup util
inline void read_mm_header(std::ifstream &s, int &field, std::size_t &row,
                           std::size_t &col, std::size_t &nnz) {
  std::string buf;
  std::getline(s, buf);
  std::istringstream       iss(buf);
  std::vector<std::string> titles(std::istream_iterator<std::string>{iss},
                                  std::istream_iterator<std::string>());
  if (titles.front() != "\%\%MatrixMarket" || titles.size() != 5u)
    psmilu_error("not a valid Matrix Market file format");
  if (titles[1] != "matrix") psmilu_error("only matrix object is supported");
  if (titles[2] != "coordinate")
    psmilu_error("only sparse format is supported");
  if (titles[3] == "complex")
    psmilu_error("complex types are not supported");
  else if (titles[3] == "pattern")
    field = 1;
  else
    field = 0;
  if (titles[4] != "general\n")
    psmilu_error("only general matrices are supported");
  // skip comments
  while (std::getline(s, buf))
    if (buf.front() != '%') break;
  std::sscanf(buf.c_str(), "%zd %zd %zd\n", &row, &col, &nnz);
}
}  // namespace internal

/// \brief top-level reader routine
/// \tparam ValueArray value array type
/// \tparam IndexArray index array type
/// \tparam OneBased if \a true, then Fortran index is assumed
/// \tparam RowMajor if \a true, then CRS is assumed
/// \param[in] filename matrix market file name
/// \param[out] ind_start starting positions
/// \param[out] indices index array
/// \param[out] vals value array
/// \param[out] row number of rows
/// \param[out] col number of columns
/// \ingroup util
template <class ValueArray, class IndexArray, bool OneBased, bool RowMajor>
inline void read_matrix_market(const char *filename, IndexArray &ind_start,
                               IndexArray &indices, ValueArray &vals,
                               std::size_t &row, std::size_t &col) {
  std::size_t   nnz;
  int           field;
  std::ifstream s(filename);
  psmilu_error_if(!s.is_open(), "cannot open file %s", filename);
  internal::read_mm_header(s, field, row, col, nnz);
  if (field == 0) {
    // general case
    internal::GeneralLineReader reader(s);
    internal::read_mm_kernel<ValueArray, IndexArray, OneBased, RowMajor,
                             internal::GeneralLineReader>(
        s, RowMajor ? row : col, reader, ind_start, indices, vals);
  } else if (field == 1) {
    // pattern
    internal::PatternLineReader reader(s);
    internal::read_mm_kernel<ValueArray, IndexArray, OneBased, RowMajor,
                             internal::PatternLineReader>(
        s, RowMajor ? row : col, reader, ind_start, indices, vals);
  }
  s.close();
}

}  // namespace psmilu

#undef __BUF_SIZE__

#endif  // _PSMILU_MATRIXMARKET_HPP
