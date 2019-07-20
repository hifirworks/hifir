//@HEADER
//----------------------------------------------------------------------------
//                Copyright (C) 2019 The HILUCSI AUTHORS
//----------------------------------------------------------------------------
//@HEADER

/// \file hilucsi/utils/print.hpp
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

#ifndef _HILUCSI_UTILS_PRINT_HPP
#define _HILUCSI_UTILS_PRINT_HPP

/// \def HILUCSI_STDOUT(__msg_wo_nl)
/// \brief dump message string to "stdout"
/// \ingroup macros
#ifndef HILUCSI_STDOUT
#  include <iostream>
#  define HILUCSI_STDOUT(__msg_wo_nl) std::cout << __msg_wo_nl << '\n'
#endif  // HILUCSI_STDOUT

/// \def HILUCSI_STDERR(__msg_wo_nl)
/// \brief dump message string to "stderr"
/// \ingroup macros
#ifndef HILUCSI_STDERR
#  include <iostream>
#  define HILUCSI_STDERR(__msg_wo_nl) std::cerr << __msg_wo_nl << '\n'
#endif  // HILUCSI_STDERR

#endif  // _HILUCSI_UTILS_PRINT_HPP