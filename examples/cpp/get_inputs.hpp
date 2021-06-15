//===========================================================================//
//                  This file is part of the HIFIR library                   //
//===========================================================================//

// common interface for getting input A, b and leading dimension
// authors: Qiao,
//
// Copyright (C) 2020 NumGeom Group at Stony Brook University
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as published
// by the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

#pragma once

#include <fstream>
#include <iostream>
#include <string>
#include <tuple>

template <class MatrixType, class ArrayType>
inline std::tuple<MatrixType, ArrayType> get_inputs(std::string dir,
                                                    const bool  rhs_a1) {
  if (dir.back() != '/') dir += "/";
  std::string A_file = dir + "A.psmilu", b_file = dir + "b.txt";
  if (!std::ifstream(A_file).good()) A_file = dir + "A.hif";
  typename ArrayType::size_type m(0);
  MatrixType                    A = MatrixType::from_bin(A_file.c_str(), &m);
  (void)m;  // not used
  ArrayType b(A.nrows());
  if (!rhs_a1) {
    std::ifstream f(b_file.c_str());
    if (!f.is_open()) {
      std::cerr << "cannot open file " << b_file << std::endl;
      std::exit(1);
    }
    for (auto &v : b) f >> v;
    f.close();
  } else
    A.multiply_nt(ArrayType(b.size(), 1.0), b);
  return std::make_tuple(A, b);
}
