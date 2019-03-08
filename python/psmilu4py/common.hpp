//@HEADER
//----------------------------------------------------------------------------
//                Copyright (C) 2019 The PSMILU AUTHORS
//----------------------------------------------------------------------------
//@HEADER

// Authors:
//  Qiao,

#ifndef _PSMILU_PYTHON_COMMON_HPP
#define _PSMILU_PYTHON_COMMON_HPP

// NOTE we need to ensure C++ code throws exceptions
#ifndef PSMILU_THROW
#  define PSMILU_THROW
#endif  // PSMILU_THROW

// we need to make sure that the stdout and stderr are not pre-defined
#ifdef PSMILU_STDOUT
#  undef PSMILU_STDOUT
#endif  // PSMILU_STDOUT
#ifdef PSMILU_STDERR
#  undef PSMILU_STDERR
#endif  // PSMILU_STDOUT

#endif  // _PSMILU_PYTHON_COMMON_HPP
