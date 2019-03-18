//@HEADER
//----------------------------------------------------------------------------
//                Copyright (C) 2019 The PSMILU AUTHORS
//----------------------------------------------------------------------------
//@HEADER

// This is a template implementation of multi-threading GMRES with right
// preconditioner for benchmark purpose.

#pragma once

#include <array>
#include <cstdlib>

#include "GMRES.hpp"

#ifdef _OPENMP
#  include <omp.h>
#endif

namespace psmilu {
namespace bench {

namespace internal {

class IdentityPrec_MT : public IdentityPrec {
  using base = IdentityPrec;

 public:
  template <class Vector>
  inline void solve_mt(const Vector &b, const std::size_t start,
                       const std::size_t len, Vector &x) const {
    std::copy_n(b.cbegin() + start, len, x.begin() + start);
  }
};

}  // namespace internal

using internal::IdentityPrec_MT;

#ifdef _OPENMP

typedef struct {
  std::size_t istart;
  std::size_t len;
} Partition;

template <class ValueType, int CacheLine = 64>
class SharedBuf {
  constexpr static int N = CacheLine / sizeof(ValueType);

  using _buf_t = std::array<ValueType, N>;
  std::vector<_buf_t> _buf;

 public:
  SharedBuf() = default;
  explicit SharedBuf(const int n) : _buf(n) {}
  inline void             resize(const int n) { _buf.resize(n); }
  inline ValueType &      operator[](const int i) { return _buf[i][0]; }
  inline const ValueType &operator[](const int i) const { return _buf[i][0]; }
  inline ValueType        sum() const {
    ValueType v(0);
    for (const auto &buf : _buf) v += buf[0];
    return v;
  }
};

#  define MT_COPY(__itr_I, __itr_O, __part)            \
    std::copy_n((__itr_I) + __part.istart, __part.len, \
                (__itr_O) + __part.istart)
#  define MT_INNER(__itr1, __itr2, __part, __v0)              \
    std::inner_product((__itr1) + __part.istart,              \
                       (__itr1) + __part.istart + __part.len, \
                       (__itr2) + __part.istart, __v0)

inline std::vector<Partition> make_part(const std::size_t global_len,
                                        const int         parts) {
  std::vector<Partition> p(parts);
  const std::size_t      len0(global_len / parts);  // floor
  const int offsets(global_len - parts * len0), offset_start(parts - offsets);
  for (int i = 0; i < offset_start; ++i) p[i] = {len0 * i, len0};
  for (int j = offset_start; j < parts; ++j) {
    p[j].istart = p[j - 1].istart + p[j - 1].len;
    p[j].len    = len0 + 1;
  }
  return p;
}

#  define FOR_PAR(__i, __part)                                              \
    for (std::size_t __i = __part.istart, _n_ = __part.istart + __part.len; \
         __i < _n_; ++__i)

template <int Digits, class Operator, class RightPrec, class Array,
          class Scalar, class Shared_Buf, class CoutStreamer,
          class CerrStreamer>
inline std::tuple<int, std::size_t, double> gmres_mt_kernel(
    const int restart, const std::size_t maxit, const Scalar rtol,
    const std::vector<Partition> &parts, const Operator &A, const RightPrec &M,
    const Array &b, Array &x, Array &_y, Array &_R, Array &Q, Array &Z,
    Array &_J, Array &v, Array &w, Array &w2, Array &resids, Shared_Buf &buf,
    const CoutStreamer &Cout, const CerrStreamer &Cerr, const bool with_guess) {
  using internal::conj;
  using std::size_t;
  using value_t                 = typename Array::value_type;
  const static Scalar _stag_eps = std::pow(Scalar(10), -Scalar(Digits));

  const size_t n = b.size();
  // quick return if invalid parameters
  if (n != x.size()) return std::make_tuple(GMRES_INVALID_FUNC_PARS, 0ul, 0.0);
  if (rtol <= 0 || restart <= 0)
    return std::make_tuple(GMRES_INVALID_PARS, 0ul, 0.0);
  if (M.empty()) {
    Cerr("\033[1;33mWARNING!\033[0m For MT version, the M cannot be empty!.\n");
    return std::make_tuple(GMRES_INVALID_PARS, 0ul, 0.0);
  }

  const size_t max_outer_iters = (size_t)std::ceil((Scalar)maxit / restart);

  Array &_w = w2;

  // shared returns
  int    g_flag(GMRES_SUCCESS);
  size_t g_iter(0);

  const int threads = parts.size();

  const double time_start = omp_get_wtime();

#  pragma omp parallel num_threads(threads) default(shared)
  {
    const int   my_id = omp_get_thread_num();
    const auto &part  = parts[my_id];

    // var for residuals
    Scalar resid(1), resid_prev;
    int    flag(GMRES_SUCCESS);
    size_t iter(0);

    buf[my_id] = MT_INNER(b.cbegin(), b.cbegin(), part, Scalar(0));
#  pragma omp barrier
    const auto beta0 = std::sqrt(buf.sum());

    // loop begins
    for (size_t it_outer(0); it_outer < max_outer_iters; ++it_outer) {
#  pragma omp master
      do {
        Cout("Enter outer iteration %zd.\n", it_outer + 1);
      } while (false);  // master
      if (it_outer || with_guess) {
#  pragma omp master
        do {
          Cerr("\033[1;33mWARNING!\033[0m Couldn\'t solve with %d restarts.\n",
               restart);
        } while (false);  // master
        A.mv_nt(x, part.istart, part.len, v);
        FOR_PAR(i, part) v[i] = b[i] - v[i];
      } else {
        MT_COPY(b.cbegin(), v.begin(), part);
      }
      buf[my_id] = MT_INNER(v.cbegin(), v.cbegin(), part, Scalar(0));
#  pragma omp barrier
#  pragma omp master
      do {
        const auto beta2 = buf.sum();
        const auto beta  = std::sqrt(beta2);
        _y[0]            = beta;
      } while (false);  // master
#  pragma omp barrier
      do {
        const auto inv_beta   = 1. / _y[0];
        FOR_PAR(i, part) Q[i] = v[i] * inv_beta;
      } while (false);
      size_t j(0);
      auto   R_itr = _R.begin();
      for (;;) {
        const auto jn = j * n;
        MT_COPY(Q.cbegin() + jn, v.begin(), part);
#  pragma omp barrier
        M.solve_mt(v, part.istart, part.len, w);
        MT_COPY(w.cbegin(), Z.begin() + jn, part);
#  pragma omp barrier
        A.mv_nt(w, part.istart, part.len, v);
        for (size_t k(0); k <= j; ++k) {
          auto itr   = Q.cbegin() + k * n;
          buf[my_id] = MT_INNER(v.cbegin(), itr, part, Scalar(0));
#  pragma omp barrier
          const auto tmp = buf.sum();
          FOR_PAR(i, part) v[i] -= tmp * itr[i];
#  pragma omp master
          _w[k] = tmp;  // master
#  pragma omp barrier   // Gosh!
        }
        buf[my_id] = MT_INNER(v.cbegin(), v.cbegin(), part, Scalar(0));
#  pragma omp barrier
        const auto v_norm2 = buf.sum(), v_norm = std::sqrt(v_norm2);
        if (j + 1 < (size_t)restart) {
          auto       itr          = Q.begin() + jn + n;
          const auto inv_norm     = 1. / v_norm;
          FOR_PAR(i, part) itr[i] = inv_norm * v[i];
        }
#  pragma omp master
        do {
          auto J1 = _J.begin(), J2 = J1 + restart;
          for (size_t colJ(0); colJ + 1u <= j; ++colJ) {
            const auto tmp = _w[colJ];
            _w[colJ]     = conj(J1[colJ]) * tmp + conj(J2[colJ]) * _w[colJ + 1];
            _w[colJ + 1] = -J2[colJ] * tmp + J1[colJ] * _w[colJ + 1];
          }
          const auto rho = std::sqrt(_w[j] * _w[j] + v_norm2);
          J1[j]          = _w[j] / rho;
          J2[j]          = v_norm / rho;
          _y[j + 1]      = -J2[j] * _y[j];
          _y[j]          = conj(J1[j]) * _y[j];
          _w[j]          = rho;
          R_itr          = std::copy_n(_w.cbegin(), j + 1, R_itr);
        } while (false);  // master
#  pragma omp barrier
        resid_prev = resid;
        resid      = std::abs(_y[j + 1]) / beta0;
        if (resid >= resid_prev * (1 - _stag_eps)) {
#  pragma omp master
          do {
            Cerr(
                "\033[1;33mWARNING!\033[0m Stagnated detected at iteration "
                "%zd.\n",
                iter);
          } while (false);  // master
          flag = GMRES_STAGNATED;
          break;
        } else if (iter >= maxit) {
#  pragma omp master
          do {
            Cerr(
                "\033[1;33mWARNING!\033[0m Reached maxit iteration limit "
                "%zd.\n",
                maxit);
          } while (false);  // master
          flag = GMRES_DIVERGED;
          break;
        }
        ++iter;
#  pragma omp master
        do {
          Cout("  At iteration %zd, relative residual is %g.\n", iter, resid);
          resids.push_back(resid);
        } while (false);  // master
        if (resid < rtol || j + 1u >= (size_t)restart) break;
        ++j;
#  pragma omp barrier
      }  // inf loop
// backsolve
#  pragma omp master
      do {
        for (int k = j; k > -1; --k) {
          --R_itr;
          _y[k] /= *R_itr;
          const auto tmp = _y[k];
          for (int i = k - 1; i > -1; --i) {
            --R_itr;
            _y[i] -= tmp * *R_itr;
          }
        }
      } while (false);  // master
#  pragma omp barrier
      for (size_t i(0); i <= j; ++i) {
        const auto tmp   = _y[i];
        auto       Z_itr = Z.cbegin() + i * n;
        FOR_PAR(k, part) x[k] += tmp * Z_itr[k];
      }
      if (resid < rtol || flag != GMRES_SUCCESS) break;
#  pragma omp barrier
    }
#  pragma omp master
    do {
      g_flag = flag;
      g_iter = iter;
    } while (false);  // master
  }                   // parallel

  const double sec = omp_get_wtime() - time_start;

  return std::make_tuple(g_flag, g_iter, sec);
}

inline int determine_num_threads() {
  const char *env_threads = std::getenv("PSMILU_NUM_THREADS");
  if (env_threads) return std::atoi(env_threads);
  env_threads = std::getenv("OMP_NUM_THREADS");
  if (env_threads) return std::atoi(env_threads);
  return omp_get_max_threads();
}

#endif

template <class Operator>
inline std::size_t primary_size(const Operator &A) {
  return A.nrows();  // CRS, overload this helper function for other cases
}

template <class ValueType = double, class PrecType = IdentityPrec_MT,
          class ArrayType = std::vector<ValueType>>
class GMRES_MT : public GMRES<ValueType, PrecType, ArrayType> {
  using _base = GMRES<ValueType, PrecType, ArrayType>;

 protected:
  constexpr static int _D = _base::_D;
  using _base::_solve_kernel;

 public:
  using value_type  = typename _base::value_type;
  using size_type   = typename _base::size_type;
  using array_type  = typename _base::array_type;
  using scalar_type = typename _base::scalar_type;
  using _base::maxit;
  using _base::restart;
  using _base::rtol;

  GMRES_MT() = default;
  GMRES_MT(const scalar_type rel_tol, int restart_ = 0, int max_iters = 0)
      : _base(rel_tol, restart_, max_iters) {}

  template <class Operator, class VectorType>
  std::tuple<int, size_type, double> solve_with_guess(
      const Operator &A, const VectorType &b, VectorType &x0,
      const bool verbose = true, int threads = 0,
      const bool __guess_flag_internal = true) const {
#ifdef _OPENMP
    bool is_mt = threads >= 0;
    if (is_mt) {
      if (!threads) threads = determine_num_threads();
      is_mt = threads > 1;
    }
    if (!is_mt)
#else
    (void)threads;
    (void)__guess_flag_internal;
#endif
    {
      return _base::solve_with_guess(A, b, x0, verbose);
    }
#ifdef _OPENMP
    if (verbose) std::printf("\nGMRES is running with %d threads\n\n", threads);
    const size_type cur_size = primary_size(A);
    const bool      repart =
        cur_size != _cached_size || _parts.size() != (size_type)threads;
    _cached_size = cur_size;
    if (repart) {
      _parts = make_part(_cached_size, threads);
      _base::_ensure_data_capacities(_cached_size);
      _ww.resize(restart + 1);
      _buf.resize(threads);
    }
    if (verbose) {
      std::printf(
          "The problem has %zd unknowns which have been partitioned as:\n\n",
          _cached_size);
      for (int i = 0; i < threads; ++i)
        std::printf(" thread %d, istart %zd, len %zd.\n", i, _parts[i].istart,
                    _parts[i].len);
    }
    // omp_set_num_threads(threads);
    return verbose
               ? gmres_mt_kernel<_D>(restart, maxit, rtol, _parts, A, _base::M,
                                     b, x0, _y, _R, _Q, _Z, _J, _v, _w, _ww,
                                     _resids, _buf, internal::Cout,
                                     internal::Cerr, __guess_flag_internal)
               : gmres_mt_kernel<_D>(restart, maxit, rtol, _parts, A, _base::M,
                                     b, x0, _y, _R, _Q, _Z, _J, _v, _w, _ww,
                                     _resids, _buf, internal::Dummy_streamer,
                                     internal::Dummy_streamer,
                                     __guess_flag_internal);
#endif
  }

  template <class Operator, class VectorType>
  std::tuple<int, size_type, double> solve(const Operator &  A,
                                           const VectorType &b, VectorType &x,
                                           const bool verbose = true,
                                           int        threads = 0) const {
    std::fill(x.begin(), x.end(), value_type());
    return solve_with_guess(A, b, x, verbose, threads, false);
  }

#ifdef _OPENMP
 protected:
  using _base::_J;
  using _base::_Q;
  using _base::_R;
  using _base::_resids;
  using _base::_v;
  using _base::_w;
  using _base::_y;
  using _base::_Z;
  mutable std::vector<Partition> _parts;
  mutable size_type              _cached_size = 0;
  mutable array_type             _ww;
  mutable SharedBuf<ValueType>   _buf;
#endif
};

}  // namespace bench
}  // namespace psmilu
