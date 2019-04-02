//@HEADER
//----------------------------------------------------------------------------
//                Copyright (C) 2019 The PSMILU AUTHORS
//----------------------------------------------------------------------------
//@HEADER

/// \file psmilu_blockJacobi/Builder.hpp
/// \brief Interface(s) for building Jacobi blocks
/// \authors Qiao,

#ifndef _PSMILU_BLOCKJACOBI_BUILDER_HPP
#define _PSMILU_BLOCKJACOBI_BUILDER_HPP

#include <algorithm>

#include "block_buillder.hpp"
#include "psmilu_Builder.hpp"
#ifndef PSMILU_DISABLE_METIS
#  include "GraphPart.hpp"
#endif  // PSMILU_DISABLE_METIS

namespace psmilu {
namespace bjacobi {

namespace internal {
// duplicate the rest template pars to avoid implementing another trait...
/// \class BJBase
/// \class Child children class
/// \tparam ValueType numerical value type, e.g. \a double
/// \tparam IndexType index type, e.g. \a int
/// \tparam OneBased if \a false (default), then assume C index system
/// \tparam SSSType default is LU with partial pivoting
template <class Child, class ValueType, class IndexType, bool OneBased,
          SmallScaleType SSSType>
class BJBase {
  /// \brief cast to const child
  inline const Child &_this() const {
    return static_cast<const Child &>(*this);
  }
  /// \brief cast to child
  inline Child &_this() { return static_cast<Child &>(*this); }

 public:
  using psmilu_type = PSMILU<ValueType, IndexType, OneBased, SSSType>;
  ///< \ref PSMILU type
  using size_type  = typename psmilu_type::size_type;   ///< size
  using array_type = typename psmilu_type::array_type;  ///< value array

  /// \brief default constructor
  /// \note We enforce _ns_ptr having at least size 1
  BJBase() : _ps(), _ns_ptr(1) { _ns_ptr.front() = 0; }

  /// \brief check number of blocks
  inline size_type blocks() const { return _ns_ptr.size() - 1; }

  /// \brief check emptyness
  inline bool empty() const { return blocks() == 0u; }

  /// \brief check size
  inline size_type size() const { return _ns_ptr.back(); }

  /// \brief check the size of a specific block
  inline size_type size(const size_type block) const {
    psmilu_error_if(block >= blocks(),
                    "%zd exceeds the total number of blocks %zd", block,
                    blocks());
    return _ns_ptr[block + 1] - _ns_ptr[block];
  }

  template <class CsType>
  inline void partition(const CsType &A, const int parts) {
    psmilu_error_if(parts < 1, "non-positive partition is not allowed");
    _ns_ptr.resize(parts + 1);
    psmilu_error_if(_ns_ptr.status() == DATA_UNDEF, "memory allocation failed");
    _ns_ptr.front() = 0;
    _ps.resize(parts);
    psmilu_error_if(_ps.status() == DATA_UNDEF, "memory allocation failed");
    // call kernel
    _this()._partition(A, parts);
  }

  template <class CsType>
  inline void factorize(const CsType & A, const size_type = 0,
                        const Options &opts  = get_default_options(),
                        const bool     check = true) {
    if (empty()) partition(A, 1);  // default do one partition
    if (psmilu_verbose(INFO, opts)) {
      if (!::psmilu::internal::introduced) {
        psmilu_info(::psmilu::internal::intro, PSMILU_GLOBAL_VERSION,
                    PSMILU_MAJOR_VERSION, PSMILU_MINOR_VERSION, __TIME__,
                    __DATE__);
        ::psmilu::internal::introduced = true;
      }
    }
    DefaultTimer t;
    for (size_type i(0); i < blocks(); ++i) {
      t.start();
      // call kernel to extract block
      const auto B = _this()._extract_block(A, i);
      t.finish();
      if (psmilu_verbose(INFO, opts)) {
        psmilu_info(
            "\nEntered block %zd and took %gs to extract the leading block",
            i + 1u, t.time());
        psmilu_info(
            "The leading block\'s size is %zd(%zd) and nnz is %zd(%zd).",
            B.nrows(), A.nrows(), B.nnz(), A.nnz());
      }
      _ps[i].factorize(B, 0, opts, check);
    }
  }

  inline void solve(const array_type &b, array_type &x) const {
    psmilu_error_if(empty(), "Block of Precs is empty");
    psmilu_error_if(size() != b.size(), "size unmatched");
    psmilu_error_if(b.size() != x.size(), "size unmatched");
    // call kernel to solve
    _this()._solve(b, x);
  }

 protected:
  Array<psmilu_type> _ps;      ///< list of \ref PSMILU
  Array<size_type>   _ns_ptr;  ///< start positions of blocks
};
}  // namespace internal

/// \class BlockJacobiPSMILU
/// \tparam ValueType numerical value type, e.g. \a double
/// \tparam IndexType index type, e.g. \a int
/// \tparam OneBased if \a false (default), then assume C index system
/// \tparam SSSType default is LU with partial pivoting
template <class ValueType, class IndexType, bool OneBased = false,
          SmallScaleType SSSType = SMALLSCALE_LUP>
class BlockJacobiPSMILU
    : public internal::BJBase<
          BlockJacobiPSMILU<ValueType, IndexType, OneBased, SSSType>, ValueType,
          IndexType, OneBased, SSSType> {
  using _base = internal::BJBase<
      BlockJacobiPSMILU<ValueType, IndexType, OneBased, SSSType>, ValueType,
      IndexType, OneBased, SSSType>;
  ///< base wrapper
  friend _base;

 public:
  using size_type  = typename _base::size_type;
  using array_type = typename _base::array_type;

  BlockJacobiPSMILU() = default;

 protected:
  template <class CsType>
  inline void _partition(const CsType &A, const int parts) {
    const size_type n = A.nrows();
    const size_type len(n / parts);
    const int       offsets(n - parts * len), offset_start(parts - offsets);
    int             i(0);
    for (; i < offset_start; ++i) _ns_ptr[i + 1] = _ns_ptr[i] + len;
    for (; i < parts; ++i) _ns_ptr[i + 1] = _ns_ptr[i] + len + 1;
  }

  template <class CsType>
  inline CsType _extract_block(const CsType &A, const size_type i) const {
    return simple_block_build(A, _ns_ptr[i], _ns_ptr[i + 1]);
  }

  inline void _solve(const array_type &b, array_type &x) const {
    constexpr static bool WRAP = true;
    using value_type           = typename array_type::value_type;
    const value_type *B(b.data());
    value_type *      X(x.data());

    for (size_type i(0); i < _base::blocks(); ++i) {
      const array_type bb(_ns_ptr[i + 1] - _ns_ptr[i],
                          const_cast<value_type *>(B + _ns_ptr[i]), WRAP);
      array_type       xx(bb.size(), X + _ns_ptr[i], WRAP);
      _ps[i].solve(bb, xx);
    }
  }

 protected:
  using _base::_ns_ptr;
  using _base::_ps;
};

/// \typedef C_BlockJacobiPSMILU
/// \tparam ValueType numerical value type, e.g. \a double
/// \tparam IndexType index type, e.g. \a int
template <class ValueType, class IndexType>
using C_BlockJacobiPSMILU = BlockJacobiPSMILU<ValueType, IndexType>;

/// \typedef F_BlockJacobiPSMILU
/// \tparam ValueType numerical value type, e.g. \a double
/// \tparam IndexType index type, e.g. \a int
template <class ValueType, class IndexType>
using F_BlockJacobiPSMILU = BlockJacobiPSMILU<ValueType, IndexType, true>;

/// \typedef C_Default_BlockJacobiPSMILU
typedef C_BlockJacobiPSMILU<double, int> C_Default_BlockJacobiPSMILU;

/// \typedef F_Default_BlockJacobiPSMILU
typedef F_BlockJacobiPSMILU<double, int> F_Default_BlockJacobiPSMILU;

#ifndef PSMILU_DISABLE_METIS

/// \class GraphBlockJacobiPSMILU
/// \tparam ValueType numerical value type, e.g. \a double
/// \tparam IndexType index type, e.g. \a int
/// \tparam OneBased if \a false (default), then assume C index system
/// \tparam SSSType default is LU with partial pivoting
template <class ValueType, class IndexType, bool OneBased = false,
          SmallScaleType SSSType = SMALLSCALE_LUP>
class GraphBlockJacobiPSMILU
    : public internal::BJBase<
          GraphBlockJacobiPSMILU<ValueType, IndexType, OneBased, SSSType>,
          ValueType, IndexType, OneBased, SSSType> {
  using _base = internal::BJBase<
      GraphBlockJacobiPSMILU<ValueType, IndexType, OneBased, SSSType>,
      ValueType, IndexType, OneBased, SSSType>;
  ///< base wrapper
  friend _base;

 public:
  using size_type  = typename _base::size_type;   ///< size
  using array_type = typename _base::array_type;  ///< value array
  constexpr static bool WITH_METIS = true;        ///< flag for using metis

  GraphBlockJacobiPSMILU() : _base(), _G(), _solve_buf() {}

  inline const GraphPart<IndexType> &G() const { return _G; }

 protected:
  template <class CsType>
  inline void _partition(const CsType &A, const int parts) {
    _G.create_parts(A, parts);
    // copy to _ns_ptr
    std::copy(_G.part_start().cbegin(), _G.part_start().cend(),
              _ns_ptr.begin());
  }

  template <class CsType>
  inline CsType _extract_block(const CsType &A, const size_type i) const {
    // TODO should we store this buffer, for dynamic problems, we can reuse them
    // to sync the values of each block
    array_type tmp;
    return _G.extract_block(A, i, tmp);
  }

  inline void _solve(const array_type &b, array_type &x) const {
    constexpr static bool WRAP = true;

    psmilu_error_if(
        b.size() != _G.size(),
        "mismatching size between graph partition and input vectors");
    const auto &P = _G.P()(), &PT = _G.P().inv();
    _solve_buf.resize(b.size());
    for (size_type i(0); i < _base::blocks(); ++i) {
      // get start idx
      const auto      start = _ns_ptr[i];
      const auto      end   = _ns_ptr[i + 1];
      const size_type n     = end - start;
      // permutate b to x
      for (size_type j(start); j < end; ++j) x[j] = b[P[j]];
      const array_type bb(n, x.data() + start, WRAP);
      array_type       xx(n, _solve_buf.data() + start, WRAP);
      _ps[i].solve(bb, xx);
    }
    // permutate back
    for (size_type j(0); j < b.size(); ++j) x[j] = _solve_buf[PT[j]];
  }

 protected:
  using _base::_ns_ptr;
  using _base::_ps;
  GraphPart<IndexType> _G;          ///< graph
  mutable array_type   _solve_buf;  ///< buffer for solving stage
};

#else

#  warning "METIS is disabled; the GraphBlockJacobiPSMILU is just an alias!"

template <class ValueType, class IndexType, bool OneBased = false,
          SmallScaleType SSSType = SMALLSCALE_LUP>
class GraphBlockJacobiPSMILU
    : public BlockJacobiPSMILU<ValueType, IndexType, OneBased, SSSType> {
 public:
  constexpr static bool WITH_METIS = false;
};

#endif  // PSMILU_DISABLE_METIS

/// \typedef C_GraphBlockJacobiPSMILU
/// \tparam ValueType numerical value type, e.g. \a double
/// \tparam IndexType index type, e.g. \a int
template <class ValueType, class IndexType>
using C_GraphBlockJacobiPSMILU = GraphBlockJacobiPSMILU<ValueType, IndexType>;

/// \typedef F_GraphBlockJacobiPSMILU
/// \tparam ValueType numerical value type, e.g. \a double
/// \tparam IndexType index type, e.g. \a int
template <class ValueType, class IndexType>
using F_GraphBlockJacobiPSMILU =
    GraphBlockJacobiPSMILU<ValueType, IndexType, true>;

/// \typedef C_Default_GraphBlockJacobiPSMILU
typedef C_GraphBlockJacobiPSMILU<double, int> C_Default_GraphBlockJacobiPSMILU;

/// \typedef F_Default_GraphBlockJacobiPSMILU
typedef F_GraphBlockJacobiPSMILU<double, int> F_Default_GraphBlockJacobiPSMILU;

}  // namespace bjacobi
}  // namespace psmilu

#endif  // _PSMILU_BLOCKJACOBI_BUILDER_HPP
