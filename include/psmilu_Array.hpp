//@HEADER
//----------------------------------------------------------------------------
//                Copyright (C) 2019 The PSMILU AUTHORS
//----------------------------------------------------------------------------
//@HEADER

/// \file psmilu_Array,hpp
/// \brief Array data structure, with functionality of wrapping external data
/// \authors Qiao,

#ifndef _PSMILU_ARRAY_HPP
#define _PSMILU_ARRAY_HPP

#include <algorithm>
#include <cstddef>
#include <new>

#include "psmilu_log.hpp"

namespace psmilu {

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

 private:
  constexpr static size_type _ZERO = static_cast<size_type>(0);
  constexpr static size_type _ONE  = static_cast<size_type>(0);

  // to enable shallow copy, we need a reference counter
  class _RefCount {
   public:
    _RefCount() : _c(_ONE) {}
    inline      operator size_type() const { return _c; }
    inline void inc() { ++_c; }
    inline void dec() { --_c; }

   private:
    size_type _c;
  };

 public:
  /// \brief default constructor
  Array()
      : _data(nullptr),
        _size(_ZERO),
        _cap(_ZERO),
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
    psmilu_assert(_data, "memory allocation failed");
    if (!_data) {
      _size = _cap = _ZERO;
      _status      = DATA_UNDEF;
    }
  }

  /// \brief similary to Array(n) but with uniform value initialization
  /// \param[in] n length of the array
  /// \param[in] v init value
  Array(const size_type n, const value_type v) : Array(n) {
    std::fill_n(_data, _data + _size, v);
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
        _size = _cap = _ZERO;
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

  /// \brief destructor, handle shallow copy and external data
  ~Array() {
    // first trigger reference counter to decrement
    _counts->dec();
#ifdef PSMILU_MEMORY_DEBUG
    std::stringstream ss;
    ss << "after decrement, current counts: " << *_counts;
    PSMILU_STDOUT(ss.str().c_str());
#endif
    // handle memory deallocation
    if (_data && _status == DATA_OWN && *_counts == _ZERO) delete[] _data;
    if (*_counts == _ZERO) {
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

#define RESET_ARRAY(arr)               \
  do {                                 \
    arr._data = nullptr;               \
    arr._size = arr._cap = _ZERO;      \
    arr._status          = DATA_UNDEF; \
    arr._counts          = nullptr;    \
  } while (false)

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
    psmilu_assert(i >= _ZERO && i < _size, "%zd exceeds the size bound", i);
    return _data[i];
  }

  /// \brief accessing data position i
  /// \param[in] i i-th position
  inline const_reference operator[](const size_type i) const {
    psmilu_assert(i >= _ZERO && i < _size, "%zd exceeds the size bound", i);
    return _data[i];
  }

  // utilities mimic STL
  inline pointer         data() { return _data; }
  inline const_pointer   data() const { return _data; }
  inline size_type       size() const { return _size; }
  inline size_type       capacity() const { return _cap; }
  inline bool            empty() const { return _size == _ZERO; }
  inline unsigned char   status() const { return _status; }
  inline reference       front() { return *_data; }
  inline reference       back() { return _data[_size - _ONE]; }
  inline const_reference front() const { return *_data; }
  inline const_reference back() const { return _data[_size - _ONE]; }

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
    psmilu_assert(_status != DATA_WRAP, "cannot resize external data");
    // quick return
    if (n <= _cap) {
      _size = n;
      return;
    }
    const size_type cap  = 1.2 * n;  // 20% more
    pointer         data = new (std::nothrow) value_type[cap];
    psmilu_assert(data, "memory allocation failed");
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
    psmilu_assert(_status != DATA_WRAP,
                  "cannot call reserve for external data");
    if (_cap >= n) return;
    pointer data = new (std::nothrow) value_type[n];
    psmilu_assert(data, "memory allocation failed");
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
    resize(_size + _ONE);
    back() = v;
  }

  /// \brief append a new range
  /// \tparam Iter iterator type
  /// \param[in] first starting iterator
  /// \param[in] last ending iterator
  /// \sa push_back_n
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
  template <class Iter>
  inline void push_back_n(Iter first, const size_type n) {
    const size_type start = _size;
    resize(_size + n);
    std::copy_n(first, n, _data + start);
  }

 protected:
  pointer       _data;
  size_type     _size;
  size_type     _cap;
  unsigned char _status;

 private:
  _RefCount* _counts;
};
}  // namespace psmilu

#endif  // _PSMILU_ARRAY_HPP
