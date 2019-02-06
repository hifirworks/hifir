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

const char *psmilu_version() { return std::to_string(PSMILU_VERSION).c_str(); }

int psmilu_error_check(void) { return error_flag; }

const char *psmilu_error_msg(void) { return msg_buf.c_str(); }
}
