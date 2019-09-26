///////////////////////////////////////////////////////////////////////////////
//                This file is part of HILUCSI project                       //
///////////////////////////////////////////////////////////////////////////////

/*!
 * \file hilucsi/utils/log.hpp
 * \brief Logging interface, including info/warning/error
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

#ifndef _HILUCSI_LOG_HPP
#define _HILUCSI_LOG_HPP

#include <cstdarg>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>
#ifdef HILUCSI_THROW
#  include <stdexcept>
#endif

#include "hilucsi/utils/print.hpp"
#include "hilucsi/utils/stacktrace.hpp"

namespace hilucsi {
namespace internal {

#ifndef DOXYGEN_SHOULD_SKIP_THIS
// NOTE for serial version, we use a global static buffer
static std::vector<char> msg_buf;

/// \brief estimate and allocate space for \a msg_buf
/// \param[in] msg message without va args
/// \note We allocate twice as length of \a msg, this should be okay.
/// \ingroup util
inline void alloc_buf(const std::string &msg) {
  // NOTE that we allocate twice more space for va_list
  const auto n = std::max(msg_buf.size(), 2u * msg.size() + 1u);
  if (n > msg_buf.size()) msg_buf.resize(n);
}

// Wrap everything into a single macro
#  define _PARSE_VA(msg)                                                  \
    hilucsi::internal::alloc_buf(msg);                                    \
    va_list aptr;                                                         \
    va_start(aptr, msg);                                                  \
    std::vsnprintf(hilucsi::internal::msg_buf.data(),                     \
                   hilucsi::internal::msg_buf.size(), msg.c_str(), aptr); \
    va_end(aptr)

#endif  // DOXYGEN_SHOULD_SKIP_THIS

}  // namespace internal

/// \brief common information streaming, dump \a msg to \ref HILUCSI_STDOUT
/// \param[in] msg message string
/// \sa hilucsi_info top level macro wrapper
/// \ingroup util
inline void info(std::string msg, ...) {
  _PARSE_VA(msg);
  HILUCSI_STDOUT(internal::msg_buf.data());
}

/// \brief warning information streaming, dump to \ref HILUCSI_STDERR
/// \param[in] prefix prefix message, omitted if passed as \a nullptr
/// \param[in] file filename
/// \param[in] func function name, i.e. \a __func__
/// \param[in] line line number, i.e. \a __LINE__
/// \param[in] msg message string
/// \sa hilucsi_warning, hilucsi_warning_if
/// \ingroup util
inline void warning(const char *prefix, const char *file, const char *func,
                    const unsigned line, std::string msg, ...) {
  const bool print_pre = prefix;
  _PARSE_VA(msg);
  std::stringstream ss;
#ifndef HILUCSI_LOG_PLAIN_PREFIX
  ss << "\033[1;33mWARNING!\033[0m ";
#else
  ss << "WARNING! ";
#endif  // HILUCSI_LOG_PLAIN_PREFIX
  if (print_pre) ss << prefix << ", ";
  ss << "function " << func << ", at " << file << ':' << line
     << "\nmessage: " << internal::msg_buf.data();
  HILUCSI_STDERR(ss.str().c_str());
}

/// \brief set/get warning flag
/// \param[in] flag warning flag
inline bool warn_flag(const int flag = -1) {
  static bool warn = true;
  if (flag < 0) return warn;
  warn = flag;
  return warn;
}

/// \brief error information streaming, dump to \ref HILUCSI_STDERR
/// \param[in] prefix prefix message, omitted if passed as \a nullptr
/// \param[in] file filename, i.e. __FILE__
/// \param[in] func function name, i.e. \a __func__
/// \param[in] line line number, i.e. \a __LINE__
/// \param[in] msg message string
/// \sa hilucsi_error, hilucsi_error_if
/// \ingroup util
inline void error(const char *prefix, const char *file, const char *func,
                  const unsigned line, std::string msg, ...) {
  const bool print_pre = prefix;
  _PARSE_VA(msg);
  std::stringstream ss;
#ifndef HILUCSI_LOG_PLAIN_PREFIX
  ss << "\033[1;31mERROR!\033[0m ";
#else
  ss << "ERROR! ";
#endif  // HILUCSI_LOG_PLAIN_PREFIX
  if (print_pre) ss << prefix << ", ";
  ss << "function " << func << ", at " << file << ':' << line
     << "\nmessage: " << internal::msg_buf.data() << "\n\n";
  LOAD_STACKTRACE(ss, 63);
#ifdef HILUCSI_THROW
  throw std::runtime_error(ss.str());
#else
  HILUCSI_STDERR(ss.str().c_str());
  std::abort();
#endif
}
}  // namespace hilucsi

/// \def hilucsi_info(__msgs)
/// \brief general message streaming macro wrapper
/// \sa hilucsi::info
/// \ingroup util
#define hilucsi_info(__msgs...) ::hilucsi::info(__msgs)

#ifndef DOXYGEN_SHOULD_SKIP_THIS

#  ifdef __GNUC__
#    define __HILUCSI_FUNC__ __FUNCTION__
#  else
#    define __HILUCSI_FUNC__ __func__
#  endif

// strip out the prefix of the file
#  define __HILUCSI_FILE__ \
    (std::strrchr(__FILE__, '/') ? std::strrchr(__FILE__, '/') + 1 : __FILE__)

#endif  // DOXYGEN_SHOULD_SKIP_THIS

/// \def hilucsi_warning(__msgs)
/// \brief print warning message
/// \sa hilucsi::warning
/// \ingroup util
#define hilucsi_warning(__msgs...)                                    \
  do {                                                                \
    if (::hilucsi::warn_flag())                                       \
      ::hilucsi::warning(nullptr, __HILUCSI_FILE__, __HILUCSI_FUNC__, \
                         __LINE__, __msgs);                           \
  } while (false)

/// \def hilucsi_warning_if(__cond, __msgs)
/// \brief conditionally print message, __cond will be shown as \a prefix
/// \sa hilucsi::warning
/// \ingroup util
#define hilucsi_warning_if(__cond, __msgs...)                               \
  do {                                                                      \
    if (::hilucsi::warn_flag() && (__cond))                                 \
      ::hilucsi::warning("condition " #__cond " alerted", __HILUCSI_FILE__, \
                         __HILUCSI_FUNC__, __LINE__, __msgs);               \
  } while (false)

/// \def hilucsi_error(__msgs)
/// \brief print warning message and abort
/// \sa hilucsi::error
/// \ingroup util
#define hilucsi_error(__msgs...)                                          \
  ::hilucsi::error(nullptr, __HILUCSI_FILE__, __HILUCSI_FUNC__, __LINE__, \
                   __msgs)

/// \def hilucsi_error_if(__cond, __msgs)
/// \brief conditionally print warning message and abort
/// \sa hilucsi::error
/// \ingroup util
#define hilucsi_error_if(__cond, __msgs...)                        \
  if (__cond)                                                      \
  ::hilucsi::error("invalid condition " #__cond, __HILUCSI_FILE__, \
                   __HILUCSI_FUNC__, __LINE__, __msgs)

/// \def hilucsi_assert(__cond, __msgs)
/// \brief internal debugging assertion
/// \ingroup util

/// \def hilucsi_debug_code(__code)
/// \brief code will only be translated on debug builds
/// \ingroup util

#ifndef NDEBUG
#  define hilucsi_assert(__cond, __msgs...)                            \
    if (!(__cond))                                                     \
    ::hilucsi::error("condition " #__cond " failed", __HILUCSI_FILE__, \
                     __HILUCSI_FUNC__, __LINE__, __msgs)
#  define hilucsi_debug_code(__code) __code
#else
#  define hilucsi_assert(__cond, __msgs...)
#  define hilucsi_debug_code(__code)
#endif

#undef _PARSE_VA

#endif  // _HILUCSI_LOG_HPP
