###############################################################################
#                 This file is part of HIF project                        #
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
file(COPY ${CMAKE_SOURCE_DIR}/src/hif DESTINATION ${CMAKE_BINARY_DIR})
file(COPY ${CMAKE_SOURCE_DIR}/src/HIF.hpp DESTINATION ${CMAKE_BINARY_DIR})
set(HIF_BUILD_DIR ${CMAKE_BINARY_DIR})

# version file
file(READ ${HIF_BUILD_DIR}/hif/version.h HILUCI_VERSION_FILE)
# global
string(REGEX MATCH "#define HIF_GLOBAL_VERSION ([0-9]*)" _HIF_GLOBAL_VERSION ${HILUCI_VERSION_FILE})
# trim global version
set(HIF_GLOBAL_VERSION ${CMAKE_MATCH_1})
# major
string(REGEX MATCH "#define HIF_MAJOR_VERSION ([0-9]*)" _HIF_MAJOR_VERSION ${HILUCI_VERSION_FILE})
set(HIF_MAJOR_VERSION ${CMAKE_MATCH_1})
# minor
string(REGEX MATCH "#define HIF_MINOR_VERSION ([0-9]*)" _HIF_MINOR_VERSION ${HILUCI_VERSION_FILE})
set(HIF_MINOR_VERSION ${CMAKE_MATCH_1})
set(HIF_VERSION ${HIF_GLOBAL_VERSION}.${HIF_MAJOR_VERSION}.${HIF_MINOR_VERSION})

# define project
project(hif LANGUAGES CXX VERSION ${HIF_VERSION})
add_library(hif INTERFACE)

# make sure we have C++11
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-std=c++11" HIF_MUST_HAVE_CPP11)
if (NOT HIF_MUST_HAVE_CPP11)
    message(FATAL_ERROR "Must have C++11 support")
endif ()

# add include directory
target_include_directories(hif INTERFACE ${CMAKE_INSTALL_PREFIX}/include)

# OpenMP
find_package(OpenMP)
if (OPENMP_FOUND)
    target_compile_options(hif INTERFACE ${OpenMP_CXX_FLAGS})
    target_link_libraries(hif INTERFACE ${OpenMP_CXX_FLAGS})
endif ()

# TODO: shall we configure LAPACK?

# module path
set(HIF_MOD_PATH ${CMAKE_SOURCE_DIR}/cmake)

##########
# options
##########

set(HIF_THROW OFF CACHE BOOL "Throw exceptions instead of call abort")
if (HIF_THROW)
    set(HIF_THROW_CODE "#define HIF_THROW")
else ()
    set(HIF_THROW_CODE "// #define HIF_THROW")
endif ()

set(HIF_LOG_PLAIN_PREFIX OFF CACHE BOOL "Without log ASCII color prefix")
if (HIF_LOG_PLAIN_PREFIX)
    set(HIF_LOG_PLAIN_PREFIX_CODE "#define HIF_LOG_PLAIN_PREFIX")
else ()
    set(HIF_LOG_PLAIN_PREFIX_CODE "// #define HIF_LOG_PLAIN_PREFIX")
endif ()

set(HIF_NO_DROP_LE_UF OFF CACHE BOOL "Disable dropping on L_E and U_F")
if (HIF_NO_DROP_LE_UF)
    set(HIF_NO_DROP_LE_UF_CODE "#define HIF_NO_DROP_LE_UF")
else ()
    set(HIF_NO_DROP_LE_UF_CODE "// #define HIF_NO_DROP_LE_UF")
endif ()

set(HIF_DISABLE_SPACE_DROP OFF CACHE BOOL "Disable scalability dropping")
if (HIF_DISABLE_SPACE_DROP)
    set(HIF_DISABLE_SPACE_DROP_CODE "#define HIF_DISABLE_SPACE_DROP")
else ()
    set(HIF_DISABLE_SPACE_DROP_CODE "// #define HIF_DISABLE_SPACE_DROP")
endif ()

set(HIF_RESERVE_FAC 5 CACHE STRING "Space ratio (wrt input) reserved for L and U for each level")
set(HIF_LASTLEVEL_DENSE_SIZE 1500 CACHE STRING "Static size for switching to dense level")
set(HIF_LASTLEVEL_SPARSE_SIZE 15000 CACHE STRING "Static size for switching to complete sparse level")
set(HIF_MIN_LOCAL_SIZE_PERCTG 85 CACHE STRING "Average percentage for local row/column sizes used in space dropping")
set(HIF_DENSE_MODE 1 CACHE STRING "Dense solve mode QRCP (1, default) or LU (!=1)")

configure_file(${HIF_MOD_PATH}/macros.hpp.in ${HIF_BUILD_DIR}/hif/macros.hpp)

# installation
include(CMakePackageConfigHelpers)
install(DIRECTORY ${HIF_BUILD_DIR}/hif DESTINATION include)
install(FILES ${HIF_BUILD_DIR}/HIF.hpp DESTINATION include)
install(TARGETS hif EXPORT hifTargets)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/cmake/hifConfigVersion.cmake
    VERSION ${HIF_VERSION}
    COMPATIBILITY SameMajorVersion)
export(EXPORT hifTargets
    FILE ${CMAKE_CURRENT_BINARY_DIR}/cmake/hifTargets.cmake
    NAMESPACE hif::
)
export(PACKAGE hif)
install(EXPORT hifTargets
    FILE hifTargets.cmake
    NAMESPACE hif::
    DESTINATION share/hif/cmake
)
configure_file(
    ${HIF_MOD_PATH}/hifConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/cmake/hifConfig.cmake
    COPYONLY
)
install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/cmake DESTINATION share/hif)
