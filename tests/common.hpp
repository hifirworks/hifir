//@HEADER
//----------------------------------------------------------------------------
//                Copyright (C) 2019 The PSMILU AUTHORS
//----------------------------------------------------------------------------
//@HEADER

#pragma once

#include <algorithm>
#include <ctime>
#include <numeric>
#include <random>
#include <type_traits>
#include <vector>

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

#define COMPARE_MATS(mat1, mat2)                                              \
  do {                                                                        \
    ASSERT_EQ(mat1.size(), mat2.size());                                      \
    ASSERT_EQ(mat1.front().size(), mat2.front().size());                      \
    const auto n = mat1.size(), m = mat1.front().size();                      \
    for (decltype(mat1.size()) i = 0u; i < n; ++i)                            \
      for (decltype(i) j = 0u; j < m; ++j) ASSERT_EQ(mat1[i][j], mat2[i][j]); \
  } while (false)

#define COMPARE_MATS_TOL(mat1, mat2, tol)                  \
  do {                                                     \
    ASSERT_EQ(mat1.size(), mat2.size());                   \
    ASSERT_EQ(mat1.front().size(), mat2.front().size());   \
    const auto n = mat1.size(), m = mat1.front().size();   \
    for (decltype(mat1.size()) i = 0u; i < n; ++i)         \
      for (decltype(i) j = 0u; j < m; ++j)                 \
        ASSERT_LE(std::abs(mat1[i][j] - mat2[i][j]), tol); \
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

  inline T operator()() const { return _d(_eng); }
};

typedef RandGen<int>    RandIntGen;
typedef RandGen<double> RandRealGen;
