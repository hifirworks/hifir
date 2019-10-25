###############################################################################
#                 This file is part of HILUCSI project                        #
###############################################################################

# Copyright (C) 2019 NumGeom Group at Stony Brook University
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

cmake_minimum_required(VERSION 3.5)

# prevent in-source build
if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
    message(FATAL_ERROR "In-source build is not allowed, please `mkdir build; cd build; cmake ..`")
endif()

# copy to current building directory
file(COPY ${CMAKE_SOURCE_DIR}/src/hilucsi DESTINATION ${CMAKE_BINARY_DIR})
file(COPY ${CMAKE_SOURCE_DIR}/src/HILUCSI.hpp DESTINATION ${CMAKE_BINARY_DIR})
set(HILUCSI_BUILD_DIR ${CMAKE_BINARY_DIR})

# version file
file(READ ${HILUCSI_BUILD_DIR}/hilucsi/version.h HILUCI_VERSION_FILE)
# global
string(REGEX MATCH "#define HILUCSI_GLOBAL_VERSION ([0-9]*)" _HILUCSI_GLOBAL_VERSION ${HILUCI_VERSION_FILE})
# trim global version
set(HILUCSI_GLOBAL_VERSION ${CMAKE_MATCH_1})
# major
string(REGEX MATCH "#define HILUCSI_MAJOR_VERSION ([0-9]*)" _HILUCSI_MAJOR_VERSION ${HILUCI_VERSION_FILE})
set(HILUCSI_MAJOR_VERSION ${CMAKE_MATCH_1})
# minor
string(REGEX MATCH "#define HILUCSI_MINOR_VERSION ([0-9]*)" _HILUCSI_MINOR_VERSION ${HILUCI_VERSION_FILE})
set(HILUCSI_MINOR_VERSION ${CMAKE_MATCH_1})
set(HILUCSI_VERSION ${HILUCSI_GLOBAL_VERSION}.${HILUCSI_MAJOR_VERSION}.${HILUCSI_MINOR_VERSION})

# define project
project(hilucsi LANGUAGES CXX VERSION ${HILUCSI_VERSION})
add_library(hilucsi INTERFACE)

# make sure we have C++11
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-std=c++11" HILUCSI_MUST_HAVE_CPP11)
if (NOT HILUCSI_MUST_HAVE_CPP11)
    message(FATAL_ERROR "Must have C++11 support")
endif ()

# add include directory
target_include_directories(hilucsi INTERFACE ${CMAKE_INSTALL_PREFIX}/include)

# OpenMP
find_package(OpenMP)
if (OPENMP_FOUND)
    target_compile_options(hilucsi INTERFACE ${OpenMP_CXX_FLAGS})
    target_link_libraries(hilucsi INTERFACE ${OpenMP_CXX_FLAGS})
endif ()

# TODO: shall we configure LAPACK?

# module path
set(HILUCSI_MOD_PATH ${CMAKE_SOURCE_DIR}/cmake)

##########
# options
##########

set(HILUCSI_THROW OFF CACHE BOOL "Throw exceptions instead of call abort")
if (HILUCSI_THROW)
    set(HILUCSI_THROW_CODE "#define HILUCSI_THROW")
else ()
    set(HILUCSI_THROW_CODE "// #define HILUCSI_THROW")
endif ()

set(HILUCSI_LOG_PLAIN_PREFIX OFF CACHE BOOL "Without log ASCII color prefix")
if (HILUCSI_LOG_PLAIN_PREFIX)
    set(HILUCSI_LOG_PLAIN_PREFIX_CODE "#define HILUCSI_LOG_PLAIN_PREFIX")
else ()
    set(HILUCSI_LOG_PLAIN_PREFIX_CODE "// #define HILUCSI_LOG_PLAIN_PREFIX")
endif ()

set(HILUCSI_NO_DROP_LE_UF OFF CACHE BOOL "Disable dropping on L_E and U_F")
if (HILUCSI_NO_DROP_LE_UF)
    set(HILUCSI_NO_DROP_LE_UF_CODE "#define HILUCSI_NO_DROP_LE_UF")
else ()
    set(HILUCSI_NO_DROP_LE_UF_CODE "// #define HILUCSI_NO_DROP_LE_UF")
endif ()

set(HILUCSI_DISABLE_SPACE_DROP OFF CACHE BOOL "Disable scalability dropping")
if (HILUCSI_DISABLE_SPACE_DROP)
    set(HILUCSI_DISABLE_SPACE_DROP_CODE "#define HILUCSI_DISABLE_SPACE_DROP")
else ()
    set(HILUCSI_DISABLE_SPACE_DROP_CODE "// #define HILUCSI_DISABLE_SPACE_DROP")
endif ()

set(HILUCSI_RESERVE_FAC 5 CACHE STRING "Space ratio (wrt input) reserved for L and U for each level")
set(HILUCSI_LASTLEVEL_DENSE_SIZE 1500 CACHE STRING "Static size for switching to dense level")
set(HILUCSI_LASTLEVEL_SPARSE_SIZE 15000 CACHE STRING "Static size for switching to complete sparse level")
set(HILUCSI_MIN_LOCAL_SIZE_PERCTG 85 CACHE STRING "Average percentage for local row/column sizes used in space dropping")
set(HILUCSI_DENSE_MODE 1 CACHE STRING "Dense solve mode LU (1, default) or QRCP (2)")

configure_file(${HILUCSI_MOD_PATH}/macros.hpp.in ${HILUCSI_BUILD_DIR}/hilucsi/macros.hpp)

# installation
include(CMakePackageConfigHelpers)
install(DIRECTORY ${HILUCSI_BUILD_DIR}/hilucsi DESTINATION include)
install(FILES ${HILUCSI_BUILD_DIR}/HILUCSI.hpp DESTINATION include)
install(TARGETS hilucsi EXPORT hilucsiTargets)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/cmake/hilucsiConfigVersion.cmake
    VERSION ${HILUCSI_VERSION}
    COMPATIBILITY SameMajorVersion)
export(EXPORT hilucsiTargets
    FILE ${CMAKE_CURRENT_BINARY_DIR}/cmake/hilucsiTargets.cmake
    NAMESPACE hilucsi::
)
export(PACKAGE hilucsi)
install(EXPORT hilucsiTargets
    FILE hilucsiTargets.cmake
    NAMESPACE hilucsi::
    DESTINATION share/hilucsi/cmake
)
configure_file(
    ${HILUCSI_MOD_PATH}/hilucsiConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/cmake/hilucsiConfig.cmake
    COPYONLY
)
install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/cmake DESTINATION share/hilucsi)
