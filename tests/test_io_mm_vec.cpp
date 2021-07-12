///////////////////////////////////////////////////////////////////////////////
//                  This file is part of the HIFIR library                   //
///////////////////////////////////////////////////////////////////////////////

#include "common.hpp"
// line break to avoid sorting
#include "hif/utils/io.hpp"

#include <gtest/gtest.h>

static const RandIntGen i_rand(5, 30);

TEST(MM_VEC_IO, real) {
  const std::size_t n(i_rand());
  using array_t = hif::Array<double>;
  const auto v1 = gen_ran_vec<array_t>(n);
  v1.write_mm("test_io_mm_vec_real.mm");
  const auto v2 = array_t::from_mm("test_io_mm_vec_real.mm");
  ASSERT_EQ(v2.size(), n);
  for (std::size_t i(0); i < n; ++i) ASSERT_NEAR(v1[i], v2[i], 1e-15);
}

TEST(MM_VEC_IO, complex) {
  const std::size_t n(i_rand());
  using _array_t = hif::Array<double>;
  const auto _v1 = gen_ran_vec<_array_t>(n), _v2 = gen_ran_vec<_array_t>(n);
  using array_t = hif::Array<std::complex<double>>;
  array_t v1(n);
  for (std::size_t i = 0; i < n; ++i) {
    v1[i].real(_v1[i]);
    v1[i].imag(_v2[i]);
  }
  v1.write_mm("test_io_mm_vec_complex.mm");
  const auto v2 = array_t::from_mm("test_io_mm_vec_complex.mm");
  ASSERT_EQ(v2.size(), n);
  for (std::size_t i(0); i < n; ++i) {
    ASSERT_NEAR(v1[i].real(), v2[i].real(), 1e-15);
    ASSERT_NEAR(v1[i].imag(), v2[i].imag(), 1e-15);
  }
}
