//@HEADER
//----------------------------------------------------------------------------
//                Copyright (C) 2019 The HILUCSI AUTHORS
//----------------------------------------------------------------------------
//@HEADER

/// \file hilucsi/utils/mt.hpp
/// \brief Multithreading helpers
/// \authors Qiao,

#ifndef _HILUCSI_UTILS_MT_HPP
#define _HILUCSI_UTILS_MT_HPP

#ifdef _OPENMP
#  include <omp.h>
#  include <cstdlib>
#endif

#include <cstddef>
#include <limits>
#include <utility>

namespace hilucsi {
namespace mt {

/*!
 * \addtogroup mt
 * @{
 */

/// \brief get number of threads
/// \param[in] threads user threads or tag
///
/// If \a threads is the INT_MAX, then query the system threads without
/// initializing (if haven't). If \a threads is not positive, then set HILUCSI
/// threads to be system threads, else set it to be user threads, and then
/// return it. Thread-safe only when \a threads is INT_MAX
inline int get_nthreads(const int threads = std::numeric_limits<int>::max()) {
#ifdef _OPENMP
  static int _nthreads = 0;
  if (threads == std::numeric_limits<int>::max()) return _nthreads;
  if (threads <= 0) {
    const char *env_threads = std::getenv("HILUCSI_NUM_THREADS");
    if (env_threads) {
      _nthreads = std::atoi(env_threads);
    } else {
      env_threads = std::getenv("OMP_NUM_THREADS");
      if (env_threads)
        _nthreads = std::atoi(env_threads);
      else
        _nthreads = omp_get_max_threads();
    }
  } else
    _nthreads = threads;
  return _nthreads;
#else
  (void)threads;
  return 1;
#endif
}

/// \brief get "my" thread id
inline int get_thread() {
#ifdef _OPENMP
  return omp_get_thread_num();
#else
  return 0;
#endif
}

/// \brief uniform partition
inline std::pair<std::size_t, std::size_t> uniform_partition(
    const std::size_t n, const int threads, const int thread) {
  const std::size_t B = n / threads, offsets = n - B * threads;
  std::size_t       start_idx_ = B * thread, len_ = B;
  if (offsets) {
    const int shift = threads - offsets;
    if (thread >= shift) {
      ++len_;
      start_idx_ += thread - shift;
    }
  }
  return std::make_pair(start_idx_, len_ + start_idx_);
}

/*!
 * @}
 */

}  // namespace mt
}  // namespace hilucsi

#endif  // _HILUCSI_UTILS_MT_HPP