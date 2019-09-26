///////////////////////////////////////////////////////////////////////////////
//                This file is part of HILUCSI project                       //
///////////////////////////////////////////////////////////////////////////////

/*!
 * \file hilucsi/ds/Array.hpp
 * \brief Array data structure, with functionality of wrapping external data
 * \author Qiao Chen

\verbatim
Copyright (C) 2019 NumGeom Group at Stony Brook University

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
\endverbatim

 */

#ifndef _HILUCSI_DS_ARRAY_HPP
#define _HILUCSI_DS_ARRAY_HPP

#include <algorithm>
#include <cstddef>
#include <new>

#include "hilucsi/utils/log.hpp"

namespace hilucsi {

/*!
 * \addtogroup ds
 * @{
 */

enum : unsigned char {
  DATA_UNDEF = 0,  ///< data type undefined
  DATA_WRAP,       ///< wrapping external data
  DATA_OWN,        ///< data type owned
};

/// \class Array
/// \brief Core data structure used in compressed storage
/// \tparam T value type
///
/// Compared to std::vector, \a Array allows one to wrap external data, which
/// is a necessary feature for this package. In addition, we allow shallow
/// copy instead of deepcopy used in std::vector.
template <class T>
class Array {
 public:
  typedef T             value_type;       ///< value type
  typedef T*            pointer;          ///< corresponding pointer type
  typedef T&            reference;        ///< corresponding lvalue reference
  typedef pointer       iterator;         ///< just use pointer for iterator
  typedef const T*      const_pointer;    ///< pointer to constant
  typedef const T&      const_reference;  ///< constant reference
  typedef const_pointer const_iterator;   ///< constant iterator
  typedef std::size_t   size_type;        ///< size type, use std::size_t

#ifndef DOXYGEN_SHOULD_SKIP_THIS
 private:
  // to enable shallow copy, we need a reference counter
  class _RefCount {
   public:
    _RefCount() : _c(1), _invalid(false) {}
    inline      operator size_type() const { return _c; }
    inline void inc() { ++_c; }
    inline void dec() { --_c; }

   private:
    size_type _c;
    bool      _invalid;

    friend class Array;
    template <class _T>
    friend void steal_array_ownership(Array<_T>&                     arr,
                                      typename Array<_T>::pointer&   data,
                                      typename Array<_T>::size_type& size,
                                      typename Array<_T>::size_type& cap);
  };
#endif  // DOXYGEN_SHOULD_SKIP_THIS

 public:
  /// \brief default constructor
  Array()
      : _data(nullptr),
        _size(0),
        _cap(0),
        _status(DATA_UNDEF),
        _counts(new _RefCount()) {}

  /// \brief constructor with owned data
  /// \param[in] n length of the array
  /// \sa DATA_OWN
  explicit Array(const size_type n)
      : _data(new (std::nothrow) value_type[n]),
        _size(n),
        _cap(n),
        _status(DATA_OWN),
        _counts(new _RefCount()) {
    // handle overflow
    hilucsi_assert(_data, "memory allocation failed");
    if (!_data) {
      _size = _cap = 0u;
      _status      = DATA_UNDEF;
    }
  }

  /// \brief similary to Array(n) but with uniform value initialization
  /// \param[in] n length of the array
  /// \param[in] v init value
  Array(const size_type n, const value_type v) : Array(n) {
    std::fill_n(_data, _size, v);
  }

  /// \brief constructo for copying or wrapping external data
  /// \param[in] n size of array
  /// \param[in] data external data
  /// \param[in] wrap flag to indicate wrapping or copying (optional)
  ///
  /// If \a wrap is \a true, then a wrapper Array will be constructed, as a
  /// result, you cannot \ref resize and \ref reserve data.
  Array(const size_type n, pointer data, bool wrap = false) {
    _counts = new _RefCount();
    if (!wrap) {
      _data = new (std::nothrow) value_type[n];
      // handle overflow
      if (!_data) {
        _size = _cap = 0u;
        _status      = DATA_UNDEF;
      } else {
        std::copy_n(data, n, _data);
        _status = DATA_OWN;
      }
    } else {
      _data   = data;
      _status = DATA_WRAP;
    }
    _size = _cap = n;
  }

  /// \brief constructor to copy a foreign array, useful in type conversion
  /// \tparam V foreign value type
  /// \param other another array
  template <class V>
  explicit Array(const Array<V>& other)
      : _data(new (std::nothrow) value_type[other.size()]),
        _size(other.size()),
        _cap(_size),
        _status(DATA_OWN),
        _counts(new _RefCount()) {
    hilucsi_assert(_data, "memory allocation failed");
    if (!_data) {
      _size = _cap = 0u;
      _status      = DATA_UNDEF;
    } else
      std::copy(other.cbegin(), other.cend(), _data);
  }

  /// \brief destructor, handle shallow copy and external data
  ~Array() {
    // first trigger reference counter to decrement
    // Added checking nullptr so that move can work
    if (_counts) _counts->dec();
    // handle memory deallocation
    if (_data && _status == DATA_OWN && _counts && *_counts == 0u)
      if (!_counts->_invalid) delete[] _data;
    if (_counts && *_counts == 0u) {
      // don't forget to free the counter
      delete _counts;
      _counts = nullptr;
    }
    _data = nullptr;
    _size = _cap = 0;
    _status      = DATA_UNDEF;
  }

  /// \brief constructor for shallow or deep copy
  /// \note the default copy constructor is shallow
  /// \param[in] other another Array
  /// \param[in] clone if \a true, then do deep copy, default is \a false
  Array(const Array& other, bool clone = false) {
    if (!clone) {
      // shallow
      _counts = other._counts;  // reference the counter
      _data   = other._data;
      _size   = other._size;
      _cap    = other._cap;
      _status = other._status;
      // increment the counter
      _counts->inc();
    } else {
      // deep copy
      _counts = new _RefCount();
      _data   = new value_type[other._size];
      _size   = other._size;
      _cap    = _size;
      _status = DATA_OWN;
      std::copy_n(other._data, _size, _data);
    }
  }

#ifndef DOXYGEN_SHOULD_SKIP_THIS
#  define RESET_ARRAY(arr)               \
    do {                                 \
      arr._data = nullptr;               \
      arr._size = arr._cap = 0u;         \
      arr._status          = DATA_UNDEF; \
      arr._counts          = nullptr;    \
    } while (false)
#endif  // DOXYGEN_SHOULD_SKIP_THIS

  /// \brief move constructor (steal)
  /// \param[in] other temp Array
  Array(Array&& other)
      : _data(other._data),
        _size(other._size),
        _cap(other._cap),
        _status(other._status),
        _counts(other._counts) {
    // reset other
    RESET_ARRAY(other);
  }

  /// \brief shallow assignment
  /// \param[in] other rhs Array
  /// \return lvalue reference of \a this
  Array& operator=(const Array& other) {
    if (_data != other._data) {
      // first, deallocate this
      this->~Array();
      _data   = other._data;
      _size   = other._size;
      _cap    = other._cap;
      _status = other._status;
      _counts = other._counts;
      _counts->inc();
    }
    return *this;
  }

  /// \brief move assignment
  /// \param[in] other rhs temp Array
  /// \return lvalue reference of \a this
  Array& operator=(Array&& other) {
    if (_data != other._data) {
      // call destructor
      this->~Array();
      _data   = other._data;
      _size   = other._size;
      _cap    = other._cap;
      _status = other._status;
      _counts = other._counts;
      RESET_ARRAY(other);
    }
    return *this;
  }

#undef RESET_ARRAY

  /// \brief accessing data position i
  /// \param[in] i i-th position
  inline reference operator[](const size_type i) {
    hilucsi_assert(i < _size, "%zd exceeds the size bound %zd", i, _size);
    return _data[i];
  }

  /// \brief accessing data position i
  /// \param[in] i i-th position
  inline const_reference operator[](const size_type i) const {
    hilucsi_assert(i < _size, "%zd exceeds the size bound %zd", i, _size);
    return _data[i];
  }

  // bound checking acessing
  inline reference at(const size_type i) {
    hilucsi_error_if(i >= _size, "%zd exceeds the size bound %zd", i, _size);
    return _data[i];
  }
  inline const_reference at(const size_type i) const {
    hilucsi_error_if(i >= _size, "%zd exceeds the size bound %zd", i, _size);
    return _data[i];
  }

  // utilities mimic STL
  inline pointer         data() { return _data; }
  inline const_pointer   data() const { return _data; }
  inline size_type       size() const { return _size; }
  inline size_type       capacity() const { return _cap; }
  inline bool            empty() const { return _size == 0u; }
  inline unsigned char   status() const { return _status; }
  inline reference       front() { return *_data; }
  inline reference       back() { return _data[_size - 1]; }
  inline const_reference front() const { return *_data; }
  inline const_reference back() const { return _data[_size - 1]; }
  inline void            swap(Array& rhs) {
    std::swap(_data, rhs._data);
    std::swap(_size, rhs._size);
    std::swap(_cap, rhs._cap);
    std::swap(_status, rhs._status);
    std::swap(_counts, rhs._counts);
  }

  // iterators and range loop functionality
  inline iterator       begin() { return _data; }
  inline const_iterator begin() const { return _data; }
  inline iterator       end() { return _data + _size; }
  inline const_iterator end() const { return _data + _size; }
  inline const_iterator cbegin() const { return _data; }
  inline const_iterator cend() const { return _data + _size; }

  // sizes

  /// \brief resize an Array with new size
  /// \param[in] n new size
  /// \param[in] presv if \a true (default), then values will be preserved
  /// \warning This function should not be called for external data
  /// \sa reserve
  inline void resize(const size_type n, bool presv = true) {
    hilucsi_assert(_status != DATA_WRAP, "cannot resize external data");
    // quick return
    if (n <= _cap) {
      _size = n;
      return;
    }
    const size_type cap  = _size ? 1.2 * n : n;  // 20% more
    pointer         data = new (std::nothrow) value_type[cap];
    hilucsi_assert(data, "memory allocation failed");
    if (!data) {
      // call destructor
      this->~Array();
      _counts = new _RefCount();
      return;
    }
    if (presv) std::copy_n(_data, _size, data);
    // now call destructor
    this->~Array();
    // since we own a new database, reset the counter
    _counts = new _RefCount();
    _data   = data;
    _size   = n;
    _cap    = cap;
    _status = DATA_OWN;
  }

  /// \brief reserve space for Array
  /// \param[in] n capacity request
  /// \warning This function should not be called for external data
  /// \sa resize
  inline void reserve(const size_type n) {
    hilucsi_assert(_status != DATA_WRAP,
                   "cannot call reserve for external data");
    if (_cap >= n) return;
    pointer data = new (std::nothrow) value_type[n];
    hilucsi_assert(data, "memory allocation failed");
    if (!data) {
      // call destructor
      this->~Array();
      _counts = new _RefCount();
      return;
    }
    std::copy_n(_data, _size, data);
    const size_type size_bak = _size;
    // now call destructor
    this->~Array();
    // since we own a new database, reset the counter
    _counts = new _RefCount();
    _data   = data;
    _cap    = n;
    _size   = size_bak;
    _status = DATA_OWN;
  }

  /// \brief append a new value
  /// \param[in] v value to be append
  inline void push_back(const value_type v) {
    resize(_size + 1);
    back() = v;
  }

  /// \brief append a new range
  /// \tparam Iter iterator type
  /// \param[in] first starting iterator
  /// \param[in] last ending iterator
  /// \sa push_back_n
  /// \warning The behavior is undefined if \a this and \a last belong to this
  template <class Iter>
  inline void push_back(Iter first, Iter last) {
    // NOTE call distance for generality
    const auto      n     = std::distance(first, last);
    const size_type start = _size;
    resize(_size + n);
    std::copy(first, last, _data + start);
  }

  /// \brief append with leading position and known size
  /// \tparam Iter iterator type
  /// \param[in] first starting position
  /// \param[in] n length to append
  /// \warning The behavior is undefined if \a first belongs to \a this
  template <class Iter>
  inline void push_back_n(Iter first, const size_type n) {
    const size_type start = _size;
    resize(_size + n);
    std::copy_n(first, n, _data + start);
  }

 protected:
  pointer       _data;    ///< data pointer
  size_type     _size;    ///< array size
  size_type     _cap;     ///< array capacity
  unsigned char _status;  ///< array status, see \a enum above

 private:
  _RefCount* _counts;  ///< reference counter

  template <class _T>
  friend void steal_array_ownership(Array<_T>&                     arr,
                                    typename Array<_T>::pointer&   data,
                                    typename Array<_T>::size_type& size,
                                    typename Array<_T>::size_type& cap);
};

/// \brief try steal the array data ownership
/// \tparam T value type
/// \param[in,out] arr input and output array
/// \param[out] data pointer to the internal database
/// \param[out] size array size
/// \param[out] cap array capacity
/// \note If \a arr can't be stole, \a nullptr is assigned to \a data
/// \warning Only works for DATA_OWN status
/// \warning Manually calling \a delete[] may be needed
/// \warning Not designed using in MT
template <class T>
inline void steal_array_ownership(Array<T>&                     arr,
                                  typename Array<T>::pointer&   data,
                                  typename Array<T>::size_type& size,
                                  typename Array<T>::size_type& cap) {
  typedef typename Array<T>::size_type size_type;
  if (arr.status() != DATA_OWN) {
    data = nullptr;
    size = cap = 0u;
    return;
  }
  data        = arr._data;
  size        = arr._size;
  cap         = arr._cap;
  arr._data   = nullptr;
  arr._size   = size_type(0);
  arr._cap    = size_type(0);
  arr._status = DATA_UNDEF;
  // set the counter to invalid
  arr._counts->_invalid = true;
}

/*!
 * @}
 */ // group ds

}  // namespace hilucsi

#endif  // _HILUCSI_DS_ARRAY_HPP
