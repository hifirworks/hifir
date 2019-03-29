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

#include "block_buillder.hpp"
#include "psmilu_Builder.hpp"

namespace psmilu {
namespace bjacobi {

template <class ValueType, class IndexType, bool OneBased = false,
          SmallScaleType SSSType = SMALLSCALE_LUP>
class BlockJacobiPSMILU {
 public:
  using psmilu_type = PSMILU<ValueType, IndexType, OneBased, SSSType>;
  using size_type   = typename psmilu_type::size_type;
  using array_type  = typename psmilu_type::array_type;

  BlockJacobiPSMILU() : _ps(), _ns_ptr(1) { _ns_ptr.front() = 0; }
  explicit BlockJacobiPSMILU(const size_type n) : _ps(n), _ns_ptr(n + 1) {
    psmilu_error_if(n == 0u, "cannot be empty parts");
    _ns_ptr.front() = 0;
  }

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

  inline void trivial_partition(const int parts, const size_type n) {
    psmilu_error_if(parts < 1, "non-positive partition is not allowed");
    _ns_ptr.resize(parts + 1);
    psmilu_error_if(_ns_ptr.status() == DATA_UNDEF, "memory allocation failed");
    _ns_ptr.front() = 0;
    _ps.resize(parts);
    psmilu_error_if(_ps.status() == DATA_UNDEF, "memory allocation failed");
    const size_type len(n / parts);
    const int       offsets(n - parts * len), offset_start(parts - offsets);
    int             i(0);
    for (; i < offset_start; ++i) _ns_ptr[i + 1] = _ns_ptr[i] + len;
    for (; i < parts; ++i) _ns_ptr[i + 1] = _ns_ptr[i] + len + 1;
  }

  template <class CsType>
  inline void factorize(const CsType & A, const size_type = 0,
                        const Options &opts  = get_default_options(),
                        const bool     check = true) {
    if (empty()) trivial_partition(1, A.nrows());
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
      const auto B = simple_block_build(A, _ns_ptr[i], _ns_ptr[i + 1]);
      t.finish();
      if (psmilu_verbose(INFO, opts)) {
        psmilu_info(
            "\nEntered block %zd and took %gs to extract the leading block",
            i + 1u, t.time());
        psmilu_info("The leading block\'s size is %zd and nnz is %zd.",
                    B.nrows(), B.nnz());
      }
      _ps[i].factorize(B, 0, opts, check);
    }
  }

  inline void solve(const array_type &b, array_type &x) const {
    constexpr static bool WRAP = true;
    using value_type           = typename array_type::value_type;

    psmilu_error_if(empty(), "Block of Precs is empty");
    psmilu_error_if(size() != b.size(), "size unmatched");
    psmilu_error_if(b.size() != x.size(), "size unmatched");
    const value_type *B(b.data());
    value_type *      X(x.data());

    for (size_type b(0); b < blocks(); ++b) {
      const array_type bb(_ns_ptr[b + 1] - _ns_ptr[b],
                          const_cast<value_type *>(B + _ns_ptr[b]), WRAP);
      array_type       xx(bb.size(), X + _ns_ptr[b], WRAP);
      _ps[b].solve(bb, xx);
    }
  }

 protected:
  Array<psmilu_type> _ps;
  Array<size_type>   _ns_ptr;
};

}  // namespace bjacobi
}  // namespace psmilu

#endif  // _PSMILU_BLOCKJACOBI_BUILDER_HPP
