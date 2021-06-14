///////////////////////////////////////////////////////////////////////////////
//                  This file is part of the HIFIR library                   //
///////////////////////////////////////////////////////////////////////////////

/*!
 * \file hif/utils/stacktrace.hpp
 * \brief Print stacktrace on error aborting
 * \author Qiao Chen
 * \note Only works on Unix/Linux

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

#ifndef _HIF_UTILS_STACKTRACE_HPP
#define _HIF_UTILS_STACKTRACE_HPP

// see https://panthema.net/2008/0901-stacktrace-demangled/

/// \def LOAD_STACKTRACE(__ss, __limit)
/// \brief Load stack trace on error aborts.
/// \ingroup util
///
/// Currently, this macro only works on *nix systems. It will load C++ symbols
/// nicely to \a ostream \a __ss, thus can be later transferred to, e.g.,
/// HIF_STDERR

#if defined(__GNUC__) && defined(__unix__)
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
#endif  // __GNUC__

#endif  // _HIF_UTILS_STACKTRACE_HPP
