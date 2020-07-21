///////////////////////////////////////////////////////////////////////////////
//                This file is part of HILUCSI project                       //
///////////////////////////////////////////////////////////////////////////////

/*!
 * \file hilucsi/NspFilter.hpp
 * \brief User interface for customizing/using attached null space filters
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

#ifndef _HILUCSI_NSPFILTER_HPP
#define _HILUCSI_NSPFILTER_HPP

#include <cstddef>
#include <functional>
#include <memory>
#include <numeric>

#include "hilucsi/utils/log.hpp"

namespace hilucsi {

/// \class NspFilter
/// \brief filter attached to \ref HILUCSI for filtering null space components
/// \ingroup itr
///
/// Flexible design of null space components filter. The default mode is
/// constant filter, i.e., there is a constant mode in the system. Two ways to
/// customize the filter, 1) override the \ref NspFilter::user_filter function
/// or 2) pass in a callback function.
///
/// For both 1) and 2), the interface is \a void(void*,std::size_t,char) where
/// the first parameter is in/out array whose length is specified by the second
/// parameter. The last parameter indicates the data types, which follow the
/// BLAS conventions, i.e., 'd' for double, 's' for single, etc.
///
/// As to the user call back function, it is wrapped inside \a std::function,
/// which allows easy binding external user data. For more, see \a std::bind.
///
/// In most cases, a (partial) constant mode is enough. If user-customized
/// behaviors are needed, then it will be easy to work with user callback
/// in C++ and override \ref NspFilter::user_filter in other programming
/// environments, e.g., Python.
class NspFilter {
 public:
  using user_cb_type =
      std::function<void(void *, const std::size_t, const char)>;
  ///< user call back type
  enum {
    CONSTANT = 0,  ///< constant null space, can applied on a certain range
    USER_OR,       ///< user override virtual function
    USER_CB,       ///< user callback function
  };

  /// \brief default constructor
  /// \param[in] start (optional) starting position for constant null space
  /// \param[in] end (optional) ending position for constant null space
  explicit NspFilter(const std::size_t start = 0,
                     const std::size_t end   = static_cast<std::size_t>(-1))
      : _type(CONSTANT) {
    _cst_rg[0] = start;
    _cst_rg[1] = end;
  }

  /// \brief constructor with user callback type
  /// \param[in] f external call back
  /// \note allow implicit construction
  NspFilter(const user_cb_type &f) : _type(USER_CB), _user_f(f) {
    _cst_rg[0] = 0;
    _cst_rg[1] = static_cast<std::size_t>(-1);
  }

  /// \brief constructor with rvalue reference of callback type
  /// \param[in] f external call back
  /// \note \a f will become invalid state
  /// \note allow implicit construction
  NspFilter(user_cb_type &&f) : _type(USER_CB), _user_f(std::move(f)) {
    _cst_rg[0] = 0;
    _cst_rg[1] = static_cast<std::size_t>(-1);
  }

  /// \brief default copy
  NspFilter(const NspFilter &) = default;

  /// \brief default move
  NspFilter(NspFilter &&) = default;

  /// \brief virtual destructor
  virtual ~NspFilter() {}

  /// \brief default assignment
  NspFilter &operator=(const NspFilter &) = default;

  /// \brief default move assignment
  NspFilter &operator=(NspFilter &&) = default;

  /// \brief implicit cast to int (type)
  inline operator int() const { return _type; }

  /// \brief specify the range for applying constant null space filter
  /// \param[in] start (optional) starting range default is 0
  /// \param[in] end (optional) ending position, default is max(size_t)
  inline void set_nsp_const(
      const std::size_t start = 0,
      const std::size_t end   = static_cast<std::size_t>(-1)) {
    _cst_rg[0] = start;
    _cst_rg[1] = end;
    _type      = CONSTANT;
  }

  /// \brief set null space callback
  inline void set_nsp_callback(const user_cb_type &f) {
    _user_f = f;
    if (!_user_f) hilucsi_error("empty user callback was not attached!");
    _type = USER_CB;
  }

  /// \brief filter null space components with double precision
  /// \note called via HILUCSI
  inline void filter(double *x, const std::size_t n) const {
    switch (_type) {
      case CONSTANT:
        _const_filter(x, n);
        break;
      case USER_OR:
        user_filter((void *)x, n, 'd');
        break;
      default:
        if (!_user_f) hilucsi_error("user callback was not attached!");
        _user_f((void *)x, n, 'd');
        break;
    }
  }

  /// \brief filter null space components with single precision
  /// \note called via HILUCSI
  inline void filter(float *x, const std::size_t n) const {
    switch (_type) {
      case CONSTANT:
        _const_filter(x, n);
        break;
      case USER_OR:
        user_filter((void *)x, n, 's');
        break;
      default:
        if (!_user_f) hilucsi_error("user callback was not attached!");
        _user_f((void *)x, n, 's');
        break;
    }
  }

  /// \brief filter null space components with long double precision
  /// \note called via HILUCSI
  inline void filter(long double *x, const std::size_t n) const {
    switch (_type) {
      case CONSTANT:
        _const_filter(x, n);
        break;
      case USER_OR:
        user_filter((void *)x, n, 'l');
        break;
      default:
        if (!_user_f) hilucsi_error("user callback was not attached!");
        _user_f((void *)x, n, 'l');
        break;
    }
  }

  /// \brief overridable routine for filtering
  virtual void user_filter(void *, const std::size_t, const char) const {
    hilucsi_error("user routine was not overrided!");
  }

 protected:
  template <class T>
  inline void _const_filter(T *x, const std::size_t n) const {
    const std::size_t start = _cst_rg[0];
    const std::size_t end =
        (_cst_rg[1] == static_cast<std::size_t>(-1) || _cst_rg[1] < _cst_rg[0])
            ? n
            : _cst_rg[1];
    hilucsi_error_if(end < start || end > n,
                     "wrong range (start,end,n)=(%zd,%zd,%zd)", start, end, n);
    if (start == end) return;
    // compute I-one*one^t/n, which is equiv as shifting
    const T shift_val =
        std::accumulate(x + start, x + end, T(0)) / (end - start);
    for (std::size_t i(start); i < end; ++i) x[i] -= shift_val;
  }

 protected:
  int          _type;       ///< type
  std::size_t  _cst_rg[2];  ///< constant range
  user_cb_type _user_f;     ///< user call back
};

/// \typedef NspFilterPtr
/// \brief shared pointer of \ref NspFilter, this should be used in user code
/// \ingroup itr
typedef std::shared_ptr<NspFilter> NspFilterPtr;

/// \brief construct \ref NspFilterPtr
/// \ingroup itr
template <class... Args>
inline NspFilterPtr create_nsp_filter(Args &&... args) {
  return std::make_shared<NspFilter>(args...);
}

}  // namespace hilucsi

#endif  // _HILUCSI_NSPFILTER_HPP