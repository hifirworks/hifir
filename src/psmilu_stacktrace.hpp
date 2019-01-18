//@HEADER
//----------------------------------------------------------------------------
//                Copyright (C) 2019 The PSMILU AUTHORS
//----------------------------------------------------------------------------
//@HEADER

/// \file psmilu_stacktrace.hpp
/// \brief Print stacktrace on error aborting
/// \authors Qiao,
/// \note Only works on Linux

#ifndef _PSMILU_STACKTRACE_HPP
#define _PSMILU_STACKTRACE_HPP

// see https://panthema.net/2008/0901-stacktrace-demangled/

/// \def LOAD_STACKTRACE(__ss, __limit)
/// \brief Load stack trace on error aborts.
/// \ingroup util
///
/// Currently, this macro only works on *nix systems. It will load C++ symbols
/// nicely to \a ostream \a __ss, thus can be later transferred to, e.g.,
/// PSMILU_STDERR

#ifdef __linux__
#  include <cxxabi.h>
#  include <execinfo.h>
#  define LOAD_STACKTRACE(__ss, __limit)                                     \
    do {                                                                     \
      __ss << "stack trace:\n";                                              \
      void *    addrs[__limit + 1];                                          \
      const int addr_len = backtrace(addrs, sizeof(addrs) / sizeof(void *)); \
      if (addr_len == 0) {                                                   \
        __ss << " <empty, possibly corrupt>\n";                              \
        return;                                                              \
      }                                                                      \
      char **     symbol_list    = backtrace_symbols(addrs, addr_len);       \
      std::size_t func_name_size = 512u;                                     \
      char *      func_name      = (char *)std::malloc(func_name_size);      \
      for (int i = 1; i < addr_len; ++i) {                                   \
        char *begin_name(nullptr), *begin_offset(nullptr),                   \
            *end_offset(nullptr);                                            \
        for (char *p = symbol_list[i]; *p; ++p) {                            \
          if (*p == '(')                                                     \
            begin_name = p;                                                  \
          else if (*p == '+')                                                \
            begin_offset = p;                                                \
          else if (*p == ')' && begin_offset) {                              \
            end_offset = p;                                                  \
            break;                                                           \
          }                                                                  \
        }                                                                    \
        if (begin_name && begin_offset && end_offset &&                      \
            begin_name < begin_offset) {                                     \
          *begin_name++   = '\0';                                            \
          *begin_offset++ = '\0';                                            \
          *end_offset     = '\0';                                            \
          int   status;                                                      \
          char *ret = abi::__cxa_demangle(begin_name, func_name,             \
                                          &func_name_size, &status);         \
          if (status == 0) {                                                 \
            func_name = ret;                                                 \
            __ss << '[' << i << "] " << symbol_list[i] << ':' << func_name   \
                 << '+' << begin_offset << '\n';                             \
          } else                                                             \
            __ss << '[' << i << "] " << symbol_list[i] << ':' << begin_name  \
                 << "()+" << begin_offset << '\n';                           \
        } else                                                               \
          __ss << '[' << i << "] " << symbol_list[i] << '\n';                \
      }                                                                      \
      std::free(func_name);                                                  \
      std::free(symbol_list);                                                \
    } while (false)
#else
#  define LOAD_STACKTRACE(__ss, __limit)                         \
    do {                                                         \
      __ss << "stack trace is not available on the platform.\n"; \
    } while (false)
#endif  // __linux__

#endif  // _PSMILU_STACKTRACE_HPP
