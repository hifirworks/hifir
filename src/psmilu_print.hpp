//@HEADER
//----------------------------------------------------------------------------
//                Copyright (C) 2019 The PSMILU AUTHORS
//----------------------------------------------------------------------------
//@HEADER

/// \file psmilu_print.hpp
/// \brief Macros for "stdout" and "stderr" message printers
/// \authors Qiao,
///
/// This file contains two simple hooks for dumping messages to "stdout" and
/// "stderr". By default, we use \a stdout and \a stderr, but one can change
/// this by pre-defining two macros. Notice that the input string has type
/// of C-string (const char *), and lacks a newline for single line msgs.
///
/// One of the use can be dumping messages to another language's streamers,
/// e.g. \a print in Python or similar routine(s) in MEX.

#ifndef _PSMILU_PRINT_HPP
#define _PSMILU_PRINT_HPP

/// \def PSMILU_STDOUT(__msg_wo_nl)
/// \brief dump message string to "stdout"
/// \ingroup util
#ifndef PSMILU_STDOUT
#  include <iostream>
#  define PSMILU_STDOUT(__msg_wo_nl) std::cout << __msg_wo_nl << '\n'
#endif  // PSMILU_STDOUT

/// \def PSMILU_STDERR(__msg_wo_nl)
/// \brief dump message string to "stderr"
/// \ingroup util
#ifndef PSMILU_STDERR
#  include <iostream>
#  define PSMILU_STDERR(__msg_wo_nl) std::cerr << __msg_wo_nl << '\n'
#endif  // PSMILU_STDERR

#endif  // _PSMILU_PRINT_HPP
