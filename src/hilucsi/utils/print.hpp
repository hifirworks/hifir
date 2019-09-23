///////////////////////////////////////////////////////////////////////////////
//                This file is part of HILUCSI project                       //
///////////////////////////////////////////////////////////////////////////////

/*!
 * \file hilucsi/utils/print.hpp
 * \brief Macros for "stdout" and "stderr" message printers
 * \authors Qiao,
 *
 * This file contains two simple hooks for dumping messages to "stdout" and
 * "stderr". By default, we use \a stdout and \a stderr, but one can change
 * this by pre-defining two macros. Notice that the input string has type
 * of C-string (const char *), and lacks a newline for single line msgs.
 *
 * One of the use can be dumping messages to another language's streamers,
 * e.g. \a print in Python or similar routine(s) in MEX.

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
