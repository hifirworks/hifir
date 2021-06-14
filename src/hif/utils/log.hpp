///////////////////////////////////////////////////////////////////////////////
//                  This file is part of the HIFIR library                   //
///////////////////////////////////////////////////////////////////////////////

/*!
 * \file hif/utils/log.hpp
 * \brief Logging interface, including info/warning/error
 * \author Qiao Chen

\verbatim
Copyright (C) 2021 NumGeom Group at Stony Brook University

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

#ifndef _HIF_LOG_HPP
#define _HIF_LOG_HPP

#include <cstdarg>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>
#ifdef HIF_THROW
#  include <stdexcept>
#endif

#include "hif/utils/print.hpp"
#include "hif/utils/stacktrace.hpp"

namespace hif {
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
#  define _PARSE_VA(msg)                                              \
    hif::internal::alloc_buf(msg);                                    \
    va_list aptr;                                                     \
    va_start(aptr, msg);                                              \
    std::vsnprintf(hif::internal::msg_buf.data(),                     \
                   hif::internal::msg_buf.size(), msg.c_str(), aptr); \
    va_end(aptr)

#endif  // DOXYGEN_SHOULD_SKIP_THIS

}  // namespace internal

/// \brief common information streaming, dump \a msg to \ref HIF_STDOUT
/// \param[in] msg message string
/// \sa hif_info top level macro wrapper
/// \ingroup util
inline void info(std::string msg, ...) {
  _PARSE_VA(msg);
  HIF_STDOUT(internal::msg_buf.data());
}

/// \brief warning information streaming, dump to \ref HIF_STDERR
/// \param[in] prefix prefix message, omitted if passed as \a nullptr
/// \param[in] file filename
/// \param[in] func function name, i.e. \a __func__
/// \param[in] line line number, i.e. \a __LINE__
/// \param[in] msg message string
/// \sa hif_warning, hif_warning_if
/// \ingroup util
inline void warning(const char *prefix, const char *file, const char *func,
                    const unsigned line, std::string msg, ...) {
  const bool print_pre = prefix;
  _PARSE_VA(msg);
  std::stringstream ss;
#ifndef HIF_LOG_PLAIN_PREFIX
  ss << "\033[1;33mWARNING!\033[0m ";
#else
  ss << "WARNING! ";
#endif  // HIF_LOG_PLAIN_PREFIX
  if (print_pre) ss << prefix << ", ";
  ss << "function " << func << ", at " << file << ':' << line
     << "\nmessage: " << internal::msg_buf.data();
  HIF_STDERR(ss.str().c_str());
}

/// \brief set/get warning flag
/// \param[in] flag warning flag
inline bool warn_flag(const int flag = -1) {
  static bool warn = true;
  if (flag < 0) return warn;
  warn = flag;
  return warn;
}

/// \brief error information streaming, dump to \ref HIF_STDERR
/// \param[in] prefix prefix message, omitted if passed as \a nullptr
/// \param[in] file filename, i.e. __FILE__
/// \param[in] func function name, i.e. \a __func__
/// \param[in] line line number, i.e. \a __LINE__
/// \param[in] msg message string
/// \sa hif_error, hif_error_if
/// \ingroup util
inline void error(const char *prefix, const char *file, const char *func,
                  const unsigned line, std::string msg, ...) {
  const bool print_pre = prefix;
  _PARSE_VA(msg);
  std::stringstream ss;
#ifndef HIF_LOG_PLAIN_PREFIX
  ss << "\033[1;31mERROR!\033[0m ";
#else
  ss << "ERROR! ";
#endif  // HIF_LOG_PLAIN_PREFIX
  if (print_pre) ss << prefix << ", ";
  ss << "function " << func << ", at " << file << ':' << line
     << "\nmessage: " << internal::msg_buf.data() << "\n\n";
  LOAD_STACKTRACE(ss, 63);
#ifdef HIF_THROW
  throw std::runtime_error(ss.str());
#else
  HIF_STDERR(ss.str().c_str());
  std::abort();
#endif
}
}  // namespace hif

/// \def hif_info(__msgs)
/// \brief general message streaming macro wrapper
/// \sa hif::info
/// \ingroup util
#define hif_info(__msgs...) ::hif::info(__msgs)

#ifndef DOXYGEN_SHOULD_SKIP_THIS

#  ifdef __GNUC__
#    define __HIF_FUNC__ __FUNCTION__
#  else
#    define __HIF_FUNC__ __func__
#  endif

// strip out the prefix of the file
#  define __HIF_FILE__ \
    (std::strrchr(__FILE__, '/') ? std::strrchr(__FILE__, '/') + 1 : __FILE__)

#endif  // DOXYGEN_SHOULD_SKIP_THIS

/// \def hif_warning(__msgs)
/// \brief print warning message
/// \sa hif::warning
/// \ingroup util
#define hif_warning(__msgs...)                                               \
  do {                                                                       \
    if (::hif::warn_flag())                                                  \
      ::hif::warning(nullptr, __HIF_FILE__, __HIF_FUNC__, __LINE__, __msgs); \
  } while (false)

/// \def hif_warning_if(__cond, __msgs)
/// \brief conditionally print message, __cond will be shown as \a prefix
/// \sa hif::warning
/// \ingroup util
#define hif_warning_if(__cond, __msgs...)                           \
  do {                                                              \
    if (::hif::warn_flag() && (__cond))                             \
      ::hif::warning("condition " #__cond " alerted", __HIF_FILE__, \
                     __HIF_FUNC__, __LINE__, __msgs);               \
  } while (false)

/// \def hif_error(__msgs)
/// \brief print warning message and abort
/// \sa hif::error
/// \ingroup util
#define hif_error(__msgs...) \
  ::hif::error(nullptr, __HIF_FILE__, __HIF_FUNC__, __LINE__, __msgs)

/// \def hif_error_if(__cond, __msgs)
/// \brief conditionally print warning message and abort
/// \sa hif::error
/// \ingroup util
#define hif_error_if(__cond, __msgs...)                                  \
  if (__cond)                                                            \
  ::hif::error("invalid condition " #__cond, __HIF_FILE__, __HIF_FUNC__, \
               __LINE__, __msgs)

/// \def hif_assert(__cond, __msgs)
/// \brief internal debugging assertion
/// \ingroup util

/// \def hif_debug_code(__code)
/// \brief code will only be translated on debug builds
/// \ingroup util

#ifndef NDEBUG
#  define hif_assert(__cond, __msgs...)                                      \
    if (!(__cond))                                                           \
    ::hif::error("condition " #__cond " failed", __HIF_FILE__, __HIF_FUNC__, \
                 __LINE__, __msgs)
#  define hif_debug_code(__code) __code
#else
#  define hif_assert(__cond, __msgs...)
#  define hif_debug_code(__code)
#endif

#undef _PARSE_VA

#endif  // _HIF_LOG_HPP
