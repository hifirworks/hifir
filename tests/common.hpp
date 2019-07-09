//@HEADER
//----------------------------------------------------------------------------
//                Copyright (C) 2019 The HILUCSI AUTHORS
//----------------------------------------------------------------------------
//@HEADER

// Unit testing utilities
// Author(s):
//    Qiao,

#pragma once

#include <algorithm>
#include <ctime>
#include <numeric>
#include <random>
#include <type_traits>
#include <vector>

#ifndef HILUCSI_THROW
#  define HILUCSI_THROW
#endif

#ifdef NDEBUG
#  undef NDEBUG
#endif

#ifndef HILUCSI_UNIT_TESTING
#  define HILUCSI_UNIT_TESTING
#endif

#include "hilucsi/utils/log.hpp"

//----------------------
// dense section
//----------------------

template <class T>
using matrix = std::vector<std::vector<T>>;

template <typename T>
static matrix<T> create_mat(const int nrows, const int ncols) {
  return matrix<T>(nrows, std::vector<T>(ncols, T()));
}

// convert crs to dense
template <class CS>
static typename std::enable_if<CS::ROW_MAJOR,
                               matrix<typename CS::value_type>>::type
convert2dense(const CS &cs) {
  typedef typename CS::value_type v_t;
  typedef typename CS::size_type  i_t;
  const auto c_idx = [](const i_t i) -> i_t { return i - CS::ONE_BASED; };
  auto       mat(create_mat<v_t>(cs.nrows(), cs.ncols()));
  for (i_t i = 0u; i < cs.nrows(); ++i) {
    auto col_itr = cs.col_ind_cbegin(i);
    for (auto val_itr = cs.val_cbegin(i), val_end = cs.val_cend(i);
         val_itr != val_end; ++col_itr, ++val_itr)
      mat[i].at(c_idx(*col_itr)) = *val_itr;
  }
  return mat;
}

// convert ccs to dense
template <class CS>
static typename std::enable_if<!CS::ROW_MAJOR,
                               matrix<typename CS::value_type>>::type
convert2dense(const CS &cs) {
  typedef typename CS::value_type v_t;
  typedef typename CS::size_type  i_t;
  const auto c_idx = [](const i_t i) -> i_t { return i - CS::ONE_BASED; };
  auto       mat(create_mat<v_t>(cs.nrows(), cs.ncols()));
  for (i_t j = 0u; j < cs.ncols(); ++j) {
    auto row_itr = cs.row_ind_cbegin(j);
    for (auto val_itr = cs.val_cbegin(j), val_end = cs.val_cend(j);
         val_itr != val_end; ++row_itr, ++val_itr)
      mat.at(c_idx(*row_itr))[j] = *val_itr;
  }
  return mat;
}

#define COMPARE_MATS(mat1, mat2)                                       \
  do {                                                                 \
    const auto n = std::min(mat1.size(), mat2.size());                 \
    const auto m = std::min(mat1.front().size(), mat2.front().size()); \
    for (decltype(mat1.size()) i = 0u; i < n; ++i)                     \
      for (decltype(i) j = 0u; j < m; ++j)                             \
        ASSERT_EQ(mat1[i][j], mat2[i][j]) << i << ',' << j << '\n';    \
  } while (false)

#define COMPARE_MATS_BLOCK_TOL(mat1, mat2, m, tol)        \
  do {                                                    \
    using int_t = std::remove_const<decltype(m)>::type;   \
    for (int_t i = 0; i < m; ++i)                         \
      for (int_t j = 0; j < m; ++j)                       \
        ASSERT_LE(std::abs(mat1[i][j] - mat2[i][j]), tol) \
            << i << ',' << j << '\n';                     \
  } while (false)

#define COMPARE_MATS_BLOCK(mat1, mat2, m) \
  COMPARE_MATS_BLOCK_TOL(mat1, mat2, m, 0)

#define COMPARE_MATS_TOL(mat1, mat2, tol)                              \
  do {                                                                 \
    const auto n = std::min(mat1.size(), mat2.size());                 \
    const auto m = std::min(mat1.front().size(), mat2.front().size()); \
    for (decltype(mat1.size()) i = 0u; i < n; ++i)                     \
      for (decltype(i) j = 0u; j < m; ++j)                             \
        ASSERT_LE(std::abs(mat1[i][j] - mat2[i][j]), tol)              \
            << i << ',' << j << '\n';                                  \
  } while (false)

template <class T>
static void interchange_dense_rows(matrix<T> &mat, const int i, const int j) {
  mat.at(i).swap(mat.at(j));
}

template <class T>
static void interchange_dense_cols(matrix<T> &mat, const int i, const int j) {
#define s_w_a_p(__x, __y) \
  temp = __x;             \
  __x  = __y;             \
  __y  = temp
  T temp;
  for (auto k = 0u; k < mat.size(); ++k) {
    s_w_a_p(mat[k].at(i), mat[k].at(j));
  }
#undef s_w_a_p
}

template <class T>
static void load_dense_col(const int col, const matrix<T> &mat,
                           std::vector<T> &cv) {
  for (auto i = 0u; i < mat.size(); ++i) cv[i] = mat[i].at(col);
}

template <class T>
static void load_dense_row(const int row, const matrix<T> &mat,
                           std::vector<T> &rv) {
  const auto &temp = mat.at(row);
  rv               = temp;
}

template <class T, class DiagArray>
static void scale_dense_left(matrix<T> &mat, const DiagArray &s) {
  for (auto i = 0u; i < mat.size(); ++i) {
    const auto d   = s[i];
    auto &     row = mat[i];
    for (auto &v : row) v *= d;
  }
}

template <class T, class DiagArray>
static void scale_dense_right(matrix<T> &mat, const DiagArray &t) {
  const auto n = std::min(mat.front().size(), t.size());
  for (auto i = 0u; i < mat.size(); ++i) {
    auto &row = mat[i];
    for (auto j = 0u; j < n; ++j) row[j] *= t[j];
  }
}

template <class T>
static matrix<T> dense_mm(const matrix<T> &A, const matrix<T> &B) {
  const int nrows = A.size();
  const int ncols = B.front().size();
  const int K     = B.size();
  auto      C     = create_mat<T>(nrows, ncols);  // all initialized to zeros
  for (int i = 0; i < nrows; ++i) {
    auto &C_i = C[i];
    for (int k = 0; k < K; ++k) {
      const T A_ik = A[i][k];
      for (int j = 0; j < ncols; ++j) C_i[j] += A_ik * B[k][j];
    }
  }
  return C;
}

template <class T, class V>
static std::vector<T> dense_mv(const matrix<T> &A, const V &x,
                               bool tranA = false) {
  if (!tranA) {
    const int      nrows = A.size();
    const int      ncols = x.size();
    std::vector<T> y(nrows, T());
    for (int i = 0; i < nrows; ++i) {
      const auto &row = A[i];
      for (int j = 0; j < ncols; ++j) y[i] += row[j] * x[j];
    }
    return y;
  }
  const int      ncols = A.size();
  const int      nrows = A.front().size();
  std::vector<T> y(nrows, T());
  for (int j = 0; j < ncols; ++j) {
    const auto &col = A[j];
    const T     xj  = x[j];
    for (int i = 0; i < nrows; ++i) y[i] += col[i] * xj;
  }
  return y;
}

template <class T>
static matrix<T> extract_leading_block(const matrix<T> &A, const int n) {
  // assume n is no larger than min(size(A))!
  auto mat = create_mat<T>(n, n);
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j) mat[i][j] = A[i][j];
  return mat;
}

template <class T>
static std::vector<T> extract_diag(const matrix<T> &A) {
  const int      n = std::min(A.size(), A.front().size());
  std::vector<T> diag(n);
  for (int i = 0; i < n; ++i) diag[i] = A[i][i];
  return diag;
}

template <class T>
static std::vector<T> extract_erase_diag(matrix<T> &A) {
  const int      n    = std::min(A.size(), A.front().size());
  const static T zero = T();
  std::vector<T> diag(n);
  for (int i = 0; i < n; ++i) {
    diag[i] = A[i][i];
    A[i][i] = zero;
  }
  return diag;
}

template <class T>
static void add_diag(const std::vector<T> &diag, matrix<T> &A) {
  const int n = std::min(std::min(A.size(), A.front().size()), diag.size());
  for (int i = 0; i < n; ++i) A[i][i] += diag[i];
}

// no permutation
template <class T, class Diag>
static matrix<T> compute_dense_Schur_c(const matrix<T> &A, const matrix<T> &L,
                                       const matrix<T> &U, const Diag &d,
                                       const int m) {
  const int n    = A.size() - m;
  matrix<T> temp = create_mat<T>(n, n);
  if (n) {
    // load A
    for (int i = 0; i < n; ++i)
      for (int j = 0; j < n; ++j) temp[i][j] = A[i + m][j + m];

    for (int i = 0; i < n; ++i) {
      const auto &l_i    = L[i + m];
      auto &      temp_i = temp[i];
      for (int k = 0; k < m; ++k) {
        const T     ld  = l_i[k] * d[k];
        const auto &u_k = U[k];
        for (int j = 0; j < n; ++j) temp_i[j] -= ld * u_k[j + m];
      }
    }
  }
  return temp;
}

template <class T>
static void forward_sub_unit_diag(const matrix<T> &L, matrix<T> &X) {
  const int n = L.size();
  const int m = X.front().size();
  for (int k = 0; k < m; ++k)
    for (int i = 1; i < n; ++i) {
      T sum(0);
      for (int j = 0; j < i; ++j) sum += L[i][j] * X[j][k];
      X[i][k] = X[i][k] - sum;
    }
}

template <class T>
static void backward_sub_unit_diag(const matrix<T> &U, matrix<T> &X) {
  const int n = U.size();
  const int m = X.front().size();
  for (int k = 0; k < m; ++k)
    for (int i = n - 2; i > -1; --i) {
      T sum(0);
      for (int j = i + 1; j < n; ++j) sum += U[i][j] * X[j][k];
      X[i][k] = X[i][k] - sum;
    }
}

template <class T, class DiagType>
static matrix<T> compute_dense_Schur_h_t_e(const matrix<T> &A, const int m,
                                           const matrix<T> &L,
                                           const matrix<T> &U,
                                           const DiagType & d,
                                           const matrix<T> &L_E) {
  const int n = A.size() - m;
  if (!n) return matrix<T>();
  auto B = create_mat<T>(m, m);
  for (int i = 0; i < m; ++i) std::copy_n(A[i].cbegin(), m, B[i].begin());
  forward_sub_unit_diag(L, B);
  for (int i = 0; i < m; ++i) {
    const auto  dd = d[i];
    auto &      b  = B[i];
    const auto &u  = U[i];
    for (int j = 0; j < m; ++j) b[j] -= dd * u[j];
  }
  return dense_mm(L_E, B);
}

template <class T>
static matrix<T> compute_dense_Schur_h_t_f(const matrix<T> &U,
                                           const matrix<T> &U_F) {
  auto temp = matrix<T>(U_F);
  backward_sub_unit_diag(U, temp);
  return temp;
}

template <class T, class DiagType>
static matrix<T> compute_dense_Schur_h(const matrix<T> &A, const int m,
                                       const matrix<T> &L, const matrix<T> &U,
                                       const DiagType &d, const matrix<T> &L_E,
                                       const matrix<T> &U_F) {
  return dense_mm(compute_dense_Schur_h_t_e(A, m, L, U, d, L_E),
                  compute_dense_Schur_h_t_f(U, U_F));
}

template <class GenericDense>
static matrix<typename GenericDense::value_type> from_gen_dense(
    const GenericDense &m) {
  using v_t = typename GenericDense::value_type;
  auto mat  = create_mat<v_t>(m.nrows(), m.ncols());
  for (auto i = 0u; i < m.nrows(); ++i)
    for (auto j = 0u; j < m.ncols(); ++j) mat[i][j] = m(i, j);
  return mat;
}

//----------------------
// aug ds section
//----------------------

template <class AugCrs>
static void load_aug_crs_col(
    const int col, const AugCrs &aug_crs,
    std::vector<typename AugCrs::crs_type::value_type> &cv) {
  typedef typename AugCrs::crs_type::value_type v_t;
  // fill the buffer
  std::fill(cv.begin(), cv.end(), v_t());
  auto id = aug_crs.start_col_id(col);
  for (;;) {
    if (aug_crs.is_nil(id)) break;
    cv.at(aug_crs.row_idx(id)) = aug_crs.val_from_col_id(id);
    id                         = aug_crs.next_col_id(id);
  }
}

template <class AugCcs>
static void load_aug_ccs_row(const int row, const AugCcs &aug_ccs,
                             std::vector<typename AugCcs::value_type> &rv) {
  typedef typename AugCcs::value_type v_t;
  // fill the buffer
  std::fill(rv.begin(), rv.end(), v_t());
  auto id = aug_ccs.start_row_id(row);
  for (;;) {
    if (aug_ccs.is_nil(id)) break;
    rv.at(aug_ccs.col_idx(id)) = aug_ccs.val_from_row_id(id);
    id                         = aug_ccs.next_row_id(id);
  }
}

//--------------------------
// random number generators
//--------------------------

template <typename T>
class RandGen {
  constexpr static bool _IS_INT = std::is_integral<T>::value;
  typedef typename std::conditional<
      _IS_INT, typename std::uniform_int_distribution<T>,
      typename std::uniform_real_distribution<T>>::type _dist_t;
  mutable std::mt19937_64                               _eng;
  mutable _dist_t                                       _d;

 public:
  typedef T value_type;
  RandGen(T low = T(), T hi = _IS_INT ? std::numeric_limits<T>::max() : (T)1)
      : _eng(std::time(0)), _d(low, hi) {}
  RandGen(const RandGen &) = delete;
  RandGen &operator=(const RandGen &) = delete;
  RandGen(RandGen &&)                 = default;

  inline T operator()() const { return _d(_eng); }
};

typedef RandGen<int>    RandIntGen;
typedef RandGen<double> RandRealGen;

// generate random sparse matrices
template <class CsType>
static typename std::enable_if<CsType::ROW_MAJOR, CsType>::type gen_rand_sparse(
    const int m, const int n, const bool ensure_diag = true) {
  using index_type                               = typename CsType::index_type;
  using value_type                               = typename CsType::value_type;
  constexpr static bool                ONE_BASED = CsType::ONE_BASED;
  const RandGen<index_type>            i_rand(0, n - 1);
  const RandGen<value_type>            v_rand;
  CsType                               A(m, n);
  std::vector<std::vector<index_type>> col_inds(m);
  std::vector<bool>                    tags(n, false);
  typename CsType::size_type           N(0);
  for (int i = 0; i < m; ++i) {
    auto &    col_ind = col_inds[i];
    const int nnz     = i_rand() + 1;
    col_ind.reserve(nnz + ensure_diag);
    index_type counts = 0, guard = 0;
    for (;;) {
      if (counts == nnz || guard > 2 * n) break;
      const index_type idx = i_rand();
      ++guard;
      if (tags[idx]) continue;
      ++counts;
      col_ind.push_back(idx);
      tags[idx] = true;
    }
    if (ensure_diag && i < n &&
        std::find(col_ind.cbegin(), col_ind.cend(), i) == col_ind.cend())
      col_ind.push_back(i);
    std::sort(col_ind.begin(), col_ind.end());
    N += col_ind.size();
    for (const auto j : col_ind) tags[j] = false;
  }
  A.reserve(N);
  std::vector<value_type> buf(n);
  A.begin_assemble_rows();
  for (int i = 0; i < m; ++i) {
    auto &col_ind = col_inds[i];
    for (auto &j : col_ind) {
      buf[j] = v_rand();
      j += ONE_BASED;
    }
    A.push_back_row(i, col_ind.cbegin(), col_ind.cend(), buf);
  }
  A.end_assemble_rows();
  return A;
}

template <class CsType>
static typename std::enable_if<!CsType::ROW_MAJOR, CsType>::type
gen_rand_sparse(const int m, const int n, const bool ensure_diag = true) {
  using index_type                               = typename CsType::index_type;
  using value_type                               = typename CsType::value_type;
  constexpr static bool                ONE_BASED = CsType::ONE_BASED;
  const RandGen<index_type>            i_rand(0, m - 1);
  const RandGen<value_type>            v_rand;
  CsType                               A(m, n);
  std::vector<std::vector<index_type>> row_inds(n);
  std::vector<bool>                    tags(m, false);
  typename CsType::size_type           N(0);
  for (int i = 0; i < n; ++i) {
    auto &    row_ind = row_inds[i];
    const int nnz     = i_rand() + 1;
    row_ind.reserve(nnz + ensure_diag);
    index_type counts = 0, guard = 0;
    for (;;) {
      if (counts == nnz || guard > 2 * m) break;
      const index_type idx = i_rand();
      ++guard;
      if (tags[idx]) continue;
      ++counts;
      row_ind.push_back(idx);
      tags[idx] = true;
    }
    if (ensure_diag && i < m &&
        std::find(row_ind.cbegin(), row_ind.cend(), i) == row_ind.cend())
      row_ind.push_back(i);
    std::sort(row_ind.begin(), row_ind.end());
    N += row_ind.size();
    for (const auto j : row_ind) tags[j] = false;
  }
  A.reserve(N);
  std::vector<value_type> buf(m);
  A.begin_assemble_cols();
  for (int i = 0; i < n; ++i) {
    auto &row_ind = row_inds[i];
    for (auto &j : row_ind) {
      buf[j] = v_rand();
      j += ONE_BASED;
    }
    A.push_back_col(i, row_ind.cbegin(), row_ind.cend(), buf);
  }
  A.end_assemble_cols();
  return A;
}

template <class Vector>
static void fill_ran_vec(Vector &v, typename Vector::value_type low = 0,
                         typename Vector::value_type hi = 0) {
  using v_t                      = typename Vector::value_type;
  const bool         use_default = low == hi;
  const RandGen<v_t> r(use_default ? RandGen<v_t>() : RandGen<v_t>(low, hi));
  for (auto &vv : v) vv = r();
}

template <class Vector>
static Vector gen_ran_vec(const typename Vector::size_type n,
                          typename Vector::value_type      low = 0,
                          typename Vector::value_type      hi  = 0) {
  Vector v(n);
  fill_ran_vec(v, low, hi);
  return v;
}

template <class CssType>
static CssType gen_rand_strict_lower_sparse(
    const typename CssType::size_type n) {
  using index_type                = typename CssType::index_type;
  using value_type                = typename CssType::value_type;
  constexpr static bool ONE_BASED = CssType::ONE_BASED;
  static_assert(!CssType::ROW_MAJOR, "must be CCS type");
  CssType                              A(n, n);
  std::vector<value_type>              buf(n);
  const RandGen<value_type>            v_rand;
  const RandGen<index_type>            nnz_rand(1, n);
  std::vector<std::vector<index_type>> row_inds(n);
  std::vector<bool>                    tags(n, false);
  const int                            N = n - 1;
  int                                  M = 0;
  for (int i = 0; i < N; ++i) {
    auto &                    row_ind = row_inds[i];
    const RandGen<index_type> i_rand(i + 1, N);
    const int                 nnz    = (nnz_rand() % (N - i)) + 1;
    index_type                counts = 0, guard = 0;
    for (;;) {
      if (counts == nnz || guard > 2 * (N - i)) break;
      const index_type idx = i_rand();
      ++guard;
      if (tags[idx]) continue;
      ++counts;
      row_ind.push_back(idx);
      tags[idx] = true;
    }
    std::sort(row_ind.begin(), row_ind.end());
    M += row_ind.size();
    for (const auto j : row_ind) tags[j] = false;
  }
  A.reserve(M);
  A.begin_assemble_cols();
  for (int i = 0; i < (index_type)n; ++i) {
    auto &row_ind = row_inds[i];
    for (auto &j : row_ind) {
      buf[j] = v_rand();
      j += ONE_BASED;
    }
    A.push_back_col(i, row_ind.cbegin(), row_ind.cend(), buf);
  }
  A.end_assemble_cols();
  return A;
}

template <class CrsType>
static CrsType gen_rand_strict_upper_sparse(
    const typename CrsType::size_type n) {
  static_assert(CrsType::ROW_MAJOR, "must be CRS type");
  auto    B = gen_rand_strict_lower_sparse<typename CrsType::other_type>(n);
  CrsType A(n, n);
  A.row_start() = std::move(B.col_start());
  A.col_ind()   = std::move(B.row_ind());
  A.vals()      = std::move(B.vals());
  return A;
}
