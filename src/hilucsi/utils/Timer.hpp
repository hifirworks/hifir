///////////////////////////////////////////////////////////////////////////////
//                This file is part of HILUCSI project                       //
///////////////////////////////////////////////////////////////////////////////

/*!
 * \file hilucsi/utils/Timer.hpp
 * \brief A simple timer wrapping around C++ chrono
 * \author Qiao Chen
 * \note C++ <chrono> timer should be thread safe

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

#ifndef _HILUCSI_UTILS_TIMER_HPP
#define _HILUCSI_UTILS_TIMER_HPP

#include <chrono>

namespace hilucsi {

/// \enum ClockType
/// \brief clock type used in application
/// \ingroup util
enum ClockType {
  TIMER_SYSTEM_CLOCK,  ///< using system clock
  TIMER_HIRES_CLOCK,   ///< using high resolution clock, may be alias of system
};

/// \enum ClockUnit
/// \brief time period units
/// \ingroup util
enum ClockUnit {
  TIMER_SECONDS = 0,   ///< default second
  TIMER_MILLISECONDS,  ///< milli second
  TIMER_MICROSECONDS,  ///< micro second
  TIMER_NANOSECONDS,   ///< nano second
};

#ifndef DOXYGEN_SHOULD_SKIP_THIS

namespace internal {

// internal traits from HILUCSI to std

template <ClockType Clock>
struct ClockTrait;  // trigger compilation error

template <>
struct ClockTrait<TIMER_SYSTEM_CLOCK> {
  typedef std::chrono::system_clock clock_type;
};

template <>
struct ClockTrait<TIMER_HIRES_CLOCK> {
  typedef std::chrono::high_resolution_clock clock_type;
};

template <ClockType Clock>
struct TimePointTrait {
  typedef typename ClockTrait<Clock>::clock_type::time_point time_point;
};

// unit traits
template <ClockUnit TimeUnit>
struct ClockUnitTrait;  // trigger compilation error

template <>
struct ClockUnitTrait<TIMER_SECONDS> {
  typedef std::ratio<1> unit_type;
};

template <>
struct ClockUnitTrait<TIMER_MILLISECONDS> {
  typedef std::milli unit_type;
};

template <>
struct ClockUnitTrait<TIMER_MICROSECONDS> {
  typedef std::micro unit_type;
};

template <>
struct ClockUnitTrait<TIMER_NANOSECONDS> {
  typedef std::nano unit_type;
};

}  // namespace internal

#endif  // DOXYGEN_SHOULD_SKIP_THIS

/// \class Timer
/// \brief C++11 standard timer implementation
/// \tparam Clock ClockType, default is TIMER_SYSTEM_CLOCK
/// \tparam RepType time period representation type, default is \a double
/// \note Using the timer in a local environment is preferred
/// \ingroup util
template <ClockType Clock = TIMER_SYSTEM_CLOCK, class RepType = double>
class Timer {
 public:
  constexpr static ClockType CLOCK = Clock;  ///< clock type
  typedef typename internal::ClockTrait<CLOCK>::clock_type clock_type;
  ///< std clock type
  typedef typename internal::TimePointTrait<CLOCK>::time_point time_point;
  ///< time point
  typedef RepType default_rep_type;  ///< representation type

  /// \brief start timer
  /// \sa finish
  inline void start() { _start = clock_type::now(); }

  /// \brief finish timer
  /// \sa start
  inline void finish() { _end = clock_type::now(); }

  /// \brief get time report
  /// \tparam _RepType representation type
  /// \tparam Unit report unit type, default is TIMER_SECONDS
  /// \return a time period representation in RepType and Unit
  template <class _RepType, ClockUnit Unit = TIMER_SECONDS>
  inline _RepType time() const {
    using unit_t = typename internal::ClockUnitTrait<Unit>::unit_type;
    std::chrono::duration<RepType, unit_t> period = _end - _start;
    return period.count();
  }

  /// \brief get time report in seconds
  inline default_rep_type time() const { return time<default_rep_type>(); }

  /// \brief get time report in milliseconds
  inline default_rep_type time_milli() const {
    return time<default_rep_type, TIMER_MILLISECONDS>();
  }

  /// \brief get time report in microseconds
  inline default_rep_type time_micro() const {
    return time<default_rep_type, TIMER_MICROSECONDS>();
  }

  /// \brief get time report in nanoseconds
  inline default_rep_type time_nano() const {
    return time<default_rep_type, TIMER_NANOSECONDS>();
  }

 private:
  time_point _start;  ///< start point
  time_point _end;    ///< end point
};

/// \typedef DefaultTimer
/// \brief default timer used in this package
/// \ingroup util
typedef Timer<> DefaultTimer;

}  // namespace hilucsi

#endif  // _HILUCSI_UTILS_TIMER_HPP
