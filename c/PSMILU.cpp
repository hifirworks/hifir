//@HEADER
//----------------------------------------------------------------------------
//                Copyright (C) 2019 The PSMILU AUTHORS
//----------------------------------------------------------------------------
//@HEADER

// source implementation

// ensure we throw
#ifndef PSMILU_THROW
#  define PSMILU_THROW
#endif

#include <string>

#include "PSMILU.h"

// local error flag
static int error_flag;

// message buffer
static std::string msg_buf;

namespace {

inline void set_error_msg(const std::string &msg) { ::msg_buf = msg; }
inline void clean_error() { ::error_flag = 0; }
inline void set_error() { ::error_flag = 1; }
inline void set_error(const std::string &msg) {
  set_error();
  set_error_msg(msg);
}
}  // namespace

extern "C" {

// main version interface
const char *psmilu_get_version(void) {
  return std::to_string(PSMILU_VERSION).c_str();
}
// compatible for fortran
const char *psmilu_get_version_(void) { return psmilu_get_version(); }
const char *psmilu_get_version__(void) { return psmilu_get_version(); }
const char *PSMILU_GET_VERSION(void) { return psmilu_get_version(); }
const char *PSMILU_GET_VERSION_(void) { return psmilu_get_version(); }
const char *PSMILU_GET_VERSION__(void) { return psmilu_get_version(); }

// main error check interface
int psmilu_error_check(void) { return error_flag; }
// compatible for fortran
int psmilu_error_check_(void) { return psmilu_error_check(); }
int psmilu_error_check__(void) { return psmilu_error_check(); }
int PSMILU_ERROR_CHECK(void) { return psmilu_error_check(); }
int PSMILU_ERROR_CHECK_(void) { return psmilu_error_check(); }
int PSMILU_ERROR_CHECK__(void) { return psmilu_error_check(); }

// main error message interface
const char *psmilu_error_msg(void) { return msg_buf.c_str(); }
// compatible for fortran
const char *psmilu_error_msg_(void) { return psmilu_error_msg(); }
const char *psmilu_error_msg__(void) { return psmilu_error_msg(); }
const char *PSMILU_ERROR_MSG(void) { return psmilu_error_msg(); }
const char *PSMILU_ERROR_MSG_(void) { return psmilu_error_msg(); }
const char *PSMILU_ERROR_MSG__(void) { return psmilu_error_msg(); }
}
