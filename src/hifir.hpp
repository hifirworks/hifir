///////////////////////////////////////////////////////////////////////////////
//                  This file is part of the HIFIR library                   //
///////////////////////////////////////////////////////////////////////////////

/*!
 * \file hifir.hpp
 * \brief top-level user interface
 * \author Qiao Chen

\verbatim
Copyright (C) 2021 NumGeom Group at Stony Brook University

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
\endverbatim

 */

#ifndef _HIFIR_HPP
#define _HIFIR_HPP

#ifdef _MKL_H_
#  define HIF_HAS_MKL 1
#else
#  define HIF_HAS_MKL 0
#endif  // MKL
// #if defined(_MKL_SPBLAS_H_) || HIF_HAS_MKL
// #  define HIF_HAS_SPARSE_MKL 1
// #else
// #  define HIF_HAS_SPARSE_MKL 0
// #endif

#include "hif/macros.hpp"

#include "hif/builder.hpp"

namespace hif {

/// \brief get the version string representation during runtime
/// \return string representation of version
/// \ingroup itr
inline std::string version() {
  using std::to_string;
  return to_string(HIF_GLOBAL_VERSION) + "." +
         to_string(HIF_MAJOR_VERSION) + "." +
         to_string(HIF_MINOR_VERSION);
}

}  // namespace hif

#endif  // _HIFIR_HPP
