///////////////////////////////////////////////////////////////////////////////
//                  This file is part of the HIFIR library                   //
///////////////////////////////////////////////////////////////////////////////

#include "common.hpp"
// line break to avoid sorting
#include "hif/ds/CompressedStorage.hpp"

#include <gtest/gtest.h>

static const RandIntGen i_rand(5, 30);

TEST(MM_IO, crs_real) {
  const std::size_t m(i_rand()), n(i_rand());
  using matrix_t = hif::CRS<double, int>;
  const auto A1  = gen_rand_sparse<matrix_t>(m, n);
  A1.write_mm("test_io_mm_crs_real.mm");
  const auto A2 = matrix_t::from_mm("test_io_mm_crs_real.mm");
  ASSERT_EQ(A2.nrows(), m);
  ASSERT_EQ(A2.ncols(), n);
  ASSERT_EQ(A2.nnz(), A1.nnz());
  for (std::size_t i = 0; i < m + 1; ++i)
    ASSERT_EQ(A2.row_start()[i], A1.row_start()[i]);
  for (std::size_t i = 0; i < A1.nnz(); ++i) {
    ASSERT_EQ(A2.col_ind()[i], A1.col_ind()[i]);
    ASSERT_NEAR(A2.vals()[i], A1.vals()[i], 1e-15);
  }
}

TEST(MM_IO, crs_complex) {
  const std::size_t m(i_rand()), n(i_rand());
  using matrix_t = hif::CRS<std::complex<double>, int>;
  matrix_t A1;
  do {
    using _matrix_t = hif::CRS<double, int>;
    const auto _A1  = gen_rand_sparse<_matrix_t>(m, n);
    A1.resize(m, n);
    const auto img = gen_ran_vec<_matrix_t::array_type>(_A1.nnz());
    A1.row_start() = _A1.row_start();
    A1.col_ind()   = _A1.col_ind();
    A1.vals().resize(_A1.nnz());
    for (std::size_t i(0); i < _A1.nnz(); ++i) {
      A1.vals()[i].real(_A1.vals()[i]);
      A1.vals()[i].imag(img[i]);
    }
  } while (false);

  A1.write_mm("test_io_mm_crs_complex.mm");
  const auto A2 = matrix_t::from_mm("test_io_mm_crs_complex.mm");
  ASSERT_EQ(A2.nrows(), m);
  ASSERT_EQ(A2.ncols(), n);
  ASSERT_EQ(A2.nnz(), A1.nnz());
  for (std::size_t i = 0; i < m + 1; ++i)
    ASSERT_EQ(A2.row_start()[i], A1.row_start()[i]);
  for (std::size_t i = 0; i < A1.nnz(); ++i) {
    ASSERT_EQ(A2.col_ind()[i], A1.col_ind()[i]);
    ASSERT_NEAR(A2.vals()[i].real(), A1.vals()[i].real(), 1e-15);
    ASSERT_NEAR(A2.vals()[i].imag(), A1.vals()[i].imag(), 1e-15);
  }
}

TEST(MM_IO, ccs_real) {
  const std::size_t m(i_rand()), n(i_rand());
  using matrix_t = hif::CCS<double, int>;
  const auto A1  = gen_rand_sparse<matrix_t>(m, n);
  A1.write_mm("test_io_mm_ccs_real.mm");
  const auto A2 = matrix_t::from_mm("test_io_mm_ccs_real.mm");
  ASSERT_EQ(A2.nrows(), m);
  ASSERT_EQ(A2.ncols(), n);
  ASSERT_EQ(A2.nnz(), A1.nnz());
  for (std::size_t i = 0; i < n + 1; ++i)
    ASSERT_EQ(A2.col_start()[i], A1.col_start()[i]);
  for (std::size_t i = 0; i < A1.nnz(); ++i) {
    ASSERT_EQ(A2.row_ind()[i], A1.row_ind()[i]);
    ASSERT_NEAR(A2.vals()[i], A1.vals()[i], 1e-15);
  }
}

TEST(MM_IO, ccs_complex) {
  const std::size_t m(i_rand()), n(i_rand());
  using matrix_t = hif::CCS<std::complex<double>, int>;
  matrix_t A1;
  do {
    using _matrix_t = hif::CCS<double, int>;
    const auto _A1  = gen_rand_sparse<_matrix_t>(m, n);
    A1.resize(m, n);
    const auto img = gen_ran_vec<_matrix_t::array_type>(_A1.nnz());
    A1.col_start() = _A1.col_start();
    A1.row_ind()   = _A1.row_ind();
    A1.vals().resize(_A1.nnz());
    for (std::size_t i(0); i < _A1.nnz(); ++i) {
      A1.vals()[i].real(_A1.vals()[i]);
      A1.vals()[i].imag(img[i]);
    }
  } while (false);

  A1.write_mm("test_io_mm_ccs_complex.mm");
  const auto A2 = matrix_t::from_mm("test_io_mm_ccs_complex.mm");
  ASSERT_EQ(A2.nrows(), m);
  ASSERT_EQ(A2.ncols(), n);
  ASSERT_EQ(A2.nnz(), A1.nnz());
  for (std::size_t i = 0; i < n + 1; ++i)
    ASSERT_EQ(A2.col_start()[i], A1.col_start()[i]);
  for (std::size_t i = 0; i < A1.nnz(); ++i) {
    ASSERT_EQ(A2.row_ind()[i], A1.row_ind()[i]);
    ASSERT_NEAR(A2.vals()[i].real(), A1.vals()[i].real(), 1e-15);
    ASSERT_NEAR(A2.vals()[i].imag(), A1.vals()[i].imag(), 1e-15);
  }
}
