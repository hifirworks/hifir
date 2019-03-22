//@HEADER
//----------------------------------------------------------------------------
//                Copyright (C) 2019 The PSMILU AUTHORS
//----------------------------------------------------------------------------
//@HEADER

/// \file psmilu_MT/utils.hpp
/// \brief Utilities for multi-threading kernels
/// \authors Qiao,

#ifndef _PSMILU_MT_UTILS_HPP
#define _PSMILU_MT_UTILS_HPP

#include <cstddef>

#include "psmilu_Array.hpp"
#include "psmilu_log.hpp"

namespace psmilu {

/*!
 * \addtogroup mt
 * @{
 */

/// \typedef UniformPart
/// \brief a compact chunk of data that belongs to a thread
///
/// Given an array of size \a n, say \a V, a thread owns data from
/// \a istart,istart+1,...,istart+len-1
typedef struct {
  std::size_t istart;  ///< index start
  std::size_t len;     ///< local length
} UniformPart;

/// \typedef UniformParts
/// \brief list of partitions
typedef Array<UniformPart> UniformParts;

/// \brief make partitions
/// \param[in] length global array length
/// \param[in] threads number of threads
inline UniformParts make_uni_parts(const std::size_t length,
                                   const int         threads) {
  psmilu_error_if(threads < 1, "invalid thread counts %d", threads);
  UniformParts parts(threads);
  psmilu_error_if(parts.status() == DATA_UNDEF, "memory allocation failed");
  const std::size_t len0(length / threads);
  const int offsets(length - threads * len0), offset_start(threads - offsets);
  int       i(0);
  for (; i < offset_start; ++i) parts[i] = {len0 * i, len0};
  for (; i < threads; ++i) {
    parts[i].istart = parts[i - 1].istart + parts[i - 1].len;
    parts[i].len    = len0 + 1;
  }
  return parts;
}

/// \def PSMILU_FOR_PAR
/// \brief wrapper for handy for loop
#define PSMILU_FOR_PAR(__i, __part, __leading)                   \
  for (std::size_t __i = __part.istart + __leading,              \
                   _n_ = __part.istart + __part.len + __leading; \
       __i < _n_; ++i)

/*!
 * @}
 */ // group mt
}  // namespace psmilu

#endif  // _PSMILU_MT_UTILS_HPP
