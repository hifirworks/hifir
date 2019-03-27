//@HEADER
//----------------------------------------------------------------------------
//                Copyright (C) 2019 The PSMILU AUTHORS
//----------------------------------------------------------------------------
//@HEADER

/// \file psmilu_Timer.hpp
/// \brief A simple timer wrapping around C++ chrono
/// \authors Qiao,
/// \note C++ <chrono> timer should be thread safe

#ifndef _PSMILU_TIMER_HPP
#define _PSMILU_TIMER_HPP

#include <chrono>

#ifdef _OPENMP
#  include <omp.h>
#endif

namespace psmilu {

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

// internal traits from psmilu to std

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
  inline void start() {
#ifndef _OPENMP
    _start = clock_type::now();
#else
    _start = omp_get_wtime();
#endif
  }

  /// \brief finish timer
  /// \sa start
  inline void finish() {
#ifndef _OPENMP
    _end = clock_type::now();
#else
    _end   = omp_get_wtime();
#endif
  }

  /// \brief get time report
  /// \tparam _RepType representation type
  /// \tparam Unit report unit type, default is TIMER_SECONDS
  /// \return a time period representation in RepType and Unit
  template <class _RepType, ClockUnit Unit = TIMER_SECONDS>
  inline _RepType time() const {
#ifndef _OPENMP
    using unit_t = typename internal::ClockUnitTrait<Unit>::unit_type;
    std::chrono::duration<RepType, unit_t> period = _end - _start;
    return period.count();
#else
    return static_cast<_RepType>(_end - _start);
#endif
  }

  /// \brief get time report in seconds
  inline default_rep_type time() const { return time<default_rep_type>(); }

  /// \brief get time report in milliseconds
  inline default_rep_type time_milli() const {
#ifndef _OPENMP
    return time<default_rep_type, TIMER_MILLISECONDS>();
#else
    constexpr static default_rep_type prefix = default_rep_type(1e3);
    return time<default_rep_type, TIMER_MILLISECONDS>() * prefix;
#endif
  }

  /// \brief get time report in microseconds
  inline default_rep_type time_micro() const {
#ifndef _OPENMP
    return time<default_rep_type, TIMER_MICROSECONDS>();
#else
    constexpr static default_rep_type prefix = default_rep_type(1e6);
    return time<default_rep_type, TIMER_MICROSECONDS>() * prefix;
#endif
  }

  /// \brief get time report in nanoseconds
  inline default_rep_type time_nano() const {
#ifndef _OPENMP
    return time<default_rep_type, TIMER_NANOSECONDS>();
#else
    constexpr static default_rep_type prefix = default_rep_type(1e9);
    return time<default_rep_type, TIMER_NANOSECONDS>() * prefix;
#endif
  }

 private:
#ifdef _OPENMP
  double _start;
  double _end;
#else
  time_point _start;  ///< start point
  time_point _end;    ///< end point
#endif
};

/// \typedef DefaultTimer
/// \brief default timer used in this package
/// \ingroup util
typedef Timer<> DefaultTimer;

}  // namespace psmilu

#endif  // _PSMILU_TIMER_HPP
