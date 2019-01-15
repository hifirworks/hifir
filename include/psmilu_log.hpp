//@HEADER
//----------------------------------------------------------------------------
//                Copyright (C) 2019 The PSMILU AUTHORS
//----------------------------------------------------------------------------
//@HEADER

/// \file psmilu_log.hpp
/// \brief Logging interface, including info/warning/error
/// \authors Qiao,

#ifndef _PSMILU_LOG_HPP
#define _PSMILU_LOG_HPP

#include <cstdarg>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>
#ifdef PSMILU_THROW
#  include <stdexcept>
#endif

#include "psmilu_print.hpp"
#include "psmilu_stacktrace.hpp"

namespace psmilu {
namespace internal {

// NOTE for serial version, we use a global static buffer
static std::vector<char> msg_buf;

/// \brief estimate and allocate space for \a msg_buf
/// \param[in] msg message without va args
/// \note We allocate twice as length of \a msg, this should be okay.
inline void alloc_buf(const std::string &msg) {
  // NOTE that we allocate twice more space for va_list
  const auto n = std::max(msg_buf.size(), 2u * msg.size() + 1u);
  if (n > msg_buf.size()) msg_buf.resize(n);
}

// Wrap everything into a single macro
#define _PARSE_VA(msg)                                                 \
  psmilu::internal::alloc_buf(msg);                                    \
  va_list aptr;                                                        \
  va_start(aptr, msg);                                                 \
  std::vsnprintf(psmilu::internal::msg_buf.data(),                     \
                 psmilu::internal::msg_buf.size(), msg.c_str(), aptr); \
  va_end(aptr)

}  // namespace internal

/// \brief common information streaming, dump \a msg to \ref PSMILU_STDOUT
/// \param[in] msg message string
/// \sa psmilu_info top level macro wrapper
inline void info(std::string msg, ...) {
  _PARSE_VA(msg);
  PSMILU_STDOUT(internal::msg_buf.data());
}

/// \brief warning information streaming, dump to \ref PSMILU_STDERR
/// \param[in] prefix prefix message, omitted if passed as \a nullptr
/// \param[in] file filename
/// \param[in] func function name, i.e. \a __func__
/// \param[in] line line number, i.e. \a __LINE__
/// \param[in] msg message string
/// \sa psmilu_warning, psmilu_warning_if
inline void warning(const char *prefix, const char *file, const char *func,
                    const unsigned line, std::string msg, ...) {
  const bool print_pre = prefix;
  _PARSE_VA(msg);
  std::stringstream ss;
  ss << "\033[1;33mWARNING!\033[0m ";
  if (print_pre) ss << prefix << ", ";
  ss << "function " << func << ", at " << file << ':' << line
     << "\nmessage: " << internal::msg_buf.data();
  PSMILU_STDERR(ss.str().c_str());
}

/// \brief error information streaming, dump to \ref PSMILU_STDERR
/// \param[in] prefix prefix message, omitted if passed as \a nullptr
/// \param[in] file filename, i.e. __FILE__
/// \param[in] func function name, i.e. \a __func__
/// \param[in] line line number, i.e. \a __LINE__
/// \param[in] msg message string
/// \sa psmilu_error, psmilu_error_if
inline void error(const char *prefix, const char *file, const char *func,
                  const unsigned line, std::string msg, ...) {
  const bool print_pre = prefix;
  _PARSE_VA(msg);
  std::stringstream ss;
  ss << "\033[1;31mERROR!\033[0m ";
  if (print_pre) ss << prefix << ", ";
  ss << "function " << func << ", at " << file << ':' << line
     << "\nmessage: " << internal::msg_buf.data() << "\n\n";
  LOAD_STACKTRACE(ss, 63);
#ifdef PSMILU_THROW
  throw std::runtime_error(ss.str());
#else
  PSMILU_STDERR(ss.str().c_str());
  std::abort();
#endif
}
}  // namespace psmilu

/// \def psmilu_info(__msgs)
/// \brief general message streaming macro wrapper
/// \sa psmilu::info
#define psmilu_info(__msgs...) ::psmilu::info(__msgs)

#ifdef __GNUC__
#  define __PSMILU_FUNC__ __FUNCTION__
#else
#  define __PSMILU_FUNC__ __func__
#endif

// strip out the prefix of the file
#define __PSMILU_FILE__ \
  (std::strrchr(__FILE__, '/') ? std::strrchr(__FILE__, '/') + 1 : __FILE__)

/// \def psmilu_warning(__msgs)
/// \brief print warning message
/// \sa psmilu::warning
#define psmilu_warning(__msgs...) \
  ::psmilu::warning(nullptr, __PSMILU_FILE__, __PSMILU_FUNC__, __LINE__, __msgs)

/// \def psmilu_warning_if(__cond, __msgs)
/// \brief conditionally print message, __cond will be shown as \a prefix
/// \sa psmilu::warning
#define psmilu_warning_if(__cond, __msgs...)                          \
  if ((__cond))                                                       \
  ::psmilu::warning("condition " #__cond " alerted", __PSMILU_FILE__, \
                    __PSMILU_FUNC__, __LINE__, __msgs)

/// \def psmilu_error(__msgs)
/// \brief print warning message and abort
/// \sa psmilu::error
#define psmilu_error(__msgs...) \
  ::psmilu::error(nullptr, __PSMILU_FILE__, __PSMILU_FUNC__, __LINE__, __msgs)

/// \def psmilu_error_if(__msgs)
/// \brief conditionally print warning message and abort
/// \sa psmilu::error
#define psmilu_error_if(__cond, __msgs...)                       \
  if (__cond)                                                    \
  ::psmilu::error("invalid condition " #__cond, __PSMILU_FILE__, \
                  __PSMILU_FUNC__, __LINE__, __msgs)

#ifndef NDEBUG
#  define psmilu_assert(__cond, __msgs...)                           \
    if (!(__cond))                                                   \
    ::psmilu::error("condition " #__cond " failed", __PSMILU_FILE__, \
                    __PSMILU_FUNC__, __LINE__, __msgs)
#  define psmilu_debug_code(__code) __code
#else
#  define psmilu_assert(__cond, __msgs...)
#  define psmilu_debug_code(__code)
#endif

#undef _PARSE_VA

#endif  // _PSMILU_LOG_HPP
