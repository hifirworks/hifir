ACLOCAL_AMFLAGS=-I m4

lib_LTLIBRARIES = libhifir.la libhifir_i64.la
AM_CXXFLAGS = -g -O3 -march=native

# source file
include_HEADERS = libhifir.h

# 32bit integer
libhifir_la_SOURCES = libhifir.cpp libhifir.h
libhifir_la_CPPFLAGS = -DNDEBUG -I../src
libhifir_la_LDFLAGS = -version-info @SHARED_VERSION_INFO@

# 64bit integer
libhifir_i64_la_SOURCES = libhifir.cpp libhifir.h
libhifir_i64_la_CPPFLAGS = -DNDEBUG -I../src -DLIBHIFIR_INT_SIZE=64
libhifir_i64_la_LDFLAGS = -version-info @SHARED_VERSION_INFO@
