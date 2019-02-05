//@HEADER
//----------------------------------------------------------------------------
//                Copyright (C) 2019 The PSMILU AUTHORS
//----------------------------------------------------------------------------
//@HEADER

/// \file psmilu_small_scale/LUP.hpp
/// \brief Small scale solver with dense LU with partial pivoting
/// \authors Qiao,

#ifndef _PSMILU_SMALLSCALE_LUP_HPP
#define _PSMILU_SMALLSCALE_LUP_HPP

#include <algorithm>

#include "SmallScaleBase.hpp"
#include "psmilu_Array.hpp"
#include "psmilu_lapack/lapack.hpp"

namespace psmilu {

template <class ValueType>
class LUP : public internal::SmallScaleBase<ValueType> {
  using _base = internal::SmallScaleBase<ValueType>;

 public:
  typedef ValueType                 value_type;     ///< value type
  typedef Lapack<value_type>        lapack_kernel;  ///< lapack backend
  typedef typename _base::size_type size_type;      ///< size type

  /// \brief get the solver type
  inline static const char *method() { return "LUP"; }

  /// \brief default constructor
  LUP() = default;

  /// \brief perform LU with partial pivoting
  inline void factorize() {
    psmilu_error_if(_mat.empty(), "matrix is still empty!");
    psmilu_error_if(!_base::is_squared(), "the matrix must be squared!");
    _ipiv.resize(_mat.nrows());
    const auto info = lapack_kernel::getrf(_mat, _ipiv);
    if (info < 0)
      psmilu_error("GETRF returned negative info!");
    else if (info > 0) {
      psmilu_warning(
          "GETRF returned positive info, U(%zd,%zd) is exactly zero! "
          "Consider using QRCP for small scale solver!",
          (size_type)info, (size_type)info);
      _rank = info - 1;
    } else
      _rank = _mat.ncols();
  }

  /// \brief solve \f$\boldsymbol{LUx}=\boldsymbol{Px}\f$
  /// \param[in,out] x input rhs, output solution
  inline void solve(Array<value_type> &x) const {
    psmilu_error_if(
        _mat.empty() || _ipiv.empty(),
        "either the matrix is not set or the factorization has not yet done!");
    psmilu_error_if(x.size() != _mat.nrows(),
                    "unmatched sizes between system and rhs");
    if (_base::full_rank()) {
      if (lapack_kernel::getrs(_mat, _ipiv, x) < 0)
        psmilu_error("GETRS returned negative info!");
    } else {
      // Not full rank
      const size_type n = _ipiv.size();
      for (size_type i = 0u; i < n; ++i)
        if (static_cast<size_type>(_ipiv[i] - 1) != i)
          std::swap(x[i], x[_ipiv[i] - 1]);
      // solve partially U
      lapack_kernel::trsv('U', 'N', 'N', _rank, _mat.data(), _mat.nrows(),
                          x.data(), 1);
      // solve L
      lapack_kernel::trsv('L', 'N', 'U', _mat, x);
    }
  }

 protected:
  using _base::_mat;               ///< matrix
  using _base::_rank;              ///< rank
  Array<psmilu_lapack_int> _ipiv;  ///< row pivoting array
};

}  // namespace psmilu

#endif  // _PSMILU_SMALLSCALE_LUP_HPP
