//@HEADER
//----------------------------------------------------------------------------
//                Copyright (C) 2019 The PSMILU AUTHORS
//----------------------------------------------------------------------------
//@HEADER

/// \file psmilu_DenseMatrix.hpp
/// \brief Last level dense storage
/// \authors Qiao,

#ifndef _PSMILU_DENSEMATRIX_HPP
#define _PSMILU_DENSEMATRIX_HPP

#include <cstddef>
#include <iterator>
#include <type_traits>

#include "psmilu_Array.hpp"

namespace psmilu {

namespace internal {
/// \class StrideIterator
/// \tparam T value type
/// \brief Row-wise iterator
/// \warning This is not efficient, just for the sake of completeness
template <typename T>
class StrideIterator {
 public:
  // NOTE that we cannot use random iterator
  // thus, using this iterator in some STL routines may not be efficient

  /// \name STL iterator requirements
  //@{

  typedef T*                              pointer;            ///< pointer type
  typedef std::bidirectional_iterator_tag iterator_category;  ///< iter tag
  typedef T                               value_type;         ///< value
  typedef T&                              reference;          ///< reference
  typedef std::ptrdiff_t                  difference_type;    ///< difference

  //@}

 private:
  pointer         _ptr;     ///< pointer
  difference_type _stride;  ///< stride size
};
}  // namespace internal

/// \class DenseMatrix
/// \brief Dense storage
/// \tparam ValueType scalar value type, e.g. \a double, \a float, etc
///
/// To be easily compatible with \b LAPACK, we choose to use column major
/// orientation, a.k.a. Fortran index order.
template <class ValueType>
class DenseMatrix {
 public:
  typedef Array<ValueType>                     array_type;     ///< array
  typedef typename array_type::value_type      value_type;     ///< value
  typedef typename array_type::pointer         pointer;        ///< pointer
  typedef typename array_type::reference       reference;      ///< reference
  typedef typename array_type::size_type       size_type;      ///< size
  typedef typename array_type::const_pointer   const_pointer;  ///< constant ptr
  typedef typename array_type::const_reference const_reference;
  ///< const reference
  typedef typename array_type::iterator col_iterator;
  ///< column iterator
  typedef typename array_type::const_iterator const_col_iterator;
  ///< constant column iterator
  typedef internal::StrideIterator<value_type> row_iterator;  ///< row interator
  typedef internal::StrideIterator<const value_type> const_row_iterator;
  ///< constant row iterator
};

}  // namespace psmilu

#endif  // _PSMILU_DENSEMATRIX_HPP
