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

// Wrap everything into a single macro
#  define _PARSE_VA(__buf, msg)                                         \
    do {                                                                \
      va_list aptr1;                                                    \
      va_start(aptr1, msg);                                             \
      va_list aptr2;                                                    \
      va_copy(aptr2, aptr1);                                            \
      __buf.resize(1 + std::vsnprintf(nullptr, 0, msg.c_str(), aptr1)); \
      va_end(aptr1);                                                    \
      std::vsnprintf(__buf.data(), __buf.size(), msg.c_str(), aptr2);   \
      va_end(aptr2);                                                    \
    } while (false)

#endif  // DOXYGEN_SHOULD_SKIP_THIS

}  // namespace internal

/// \brief common information streaming, dump \a msg to \ref HIF_STDOUT
/// \param[in] msg message string
/// \sa hif_info top level macro wrapper
/// \ingroup util
inline void info(std::string msg, ...) {
  std::vector<char> msg_buf;
  _PARSE_VA(msg_buf, msg);
  HIF_STDOUT(msg_buf.data());
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
  std::vector<char> msg_buf;
  const bool        print_pre = prefix;
  _PARSE_VA(msg_buf, msg);
  std::stringstream ss;
#ifndef HIF_LOG_PLAIN_PREFIX
  ss << "\033[1;33mWARNING!\033[0m ";
#else
  ss << "WARNING! ";
#endif  // HIF_LOG_PLAIN_PREFIX
  if (print_pre) ss << prefix << ", ";
  ss << "function " << func << ", at " << file << ':' << line
     << "\nmessage: " << msg_buf.data();
  HIF_STDERR(ss.str().c_str());
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
  std::vector<char> msg_buf;
  const bool        print_pre = prefix;
  _PARSE_VA(msg_buf, msg);
  std::stringstream ss;
#ifndef HIF_LOG_PLAIN_PREFIX
  ss << "\033[1;31mERROR!\033[0m ";
#else
  ss << "ERROR! ";
#endif  // HIF_LOG_PLAIN_PREFIX
  if (print_pre) ss << prefix << ", ";
  ss << "function " << func << ", at " << file << ':' << line
     << "\nmessage: " << msg_buf.data() << "\n\n";
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
#define hif_warning(__msgs...) \
  ::hif::warning(nullptr, __HIF_FILE__, __HIF_FUNC__, __LINE__, __msgs);

/// \def hif_warning_if(__cond, __msgs)
/// \brief conditionally print message, __cond will be shown as \a prefix
/// \sa hif::warning
/// \ingroup util
#define hif_warning_if(__cond, __msgs...)                                     \
  ::hif::warning("condition " #__cond " alerted", __HIF_FILE__, __HIF_FUNC__, \
                 __LINE__, __msgs);

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

#if defined HIF_DEBUG
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
