//@HEADER
//----------------------------------------------------------------------------
//                Copyright (C) 2019 The PSMILU AUTHORS
//----------------------------------------------------------------------------
//@HEADER

#pragma once

#ifndef PSMILU_THROW
#  define PSMILU_THROW
#endif

#ifdef NDEBUG
#  undef NDEBUG
#endif

#ifndef PSMILU_MEMORY_DEBUG
#  define PSMILU_MEMORY_DEBUG
#endif

#include "psmilu_log.hpp"
