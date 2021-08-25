AC_PREREQ([2.61])
AC_INIT([libhifir], [0.1.0], [], [], [https://github.com/hifirworks/hifir])
AC_CONFIG_AUX_DIR(aux-dist)
AM_INIT_AUTOMAKE([foreign 1.13])

# Shared-library version number for libtool, which indicates api compatibility,
# and is not the same as the "public" version number.
# see https://autotools.io/libtool/version.html
# also see https://www.sourceware.org/autobook/autobook/autobook_61.html
SHARED_VERSION_INFO="1:0:0" # CURRENT:REVISION:AGE
# first interface (1), revision=0 first implementation, age=0
AC_SUBST(SHARED_VERSION_INFO)
LT_INIT

AC_PROG_CXX
AC_LANG_CPLUSPLUS

AX_CXX_COMPILE_STDCXX([11], [noext], [mandatory])
CXXFLAGS = -std=c++11

AC_CONFIG_MACRO_DIR([m4])

# Check for OpenMP libraries

AC_ARG_WITH(openmp, [AS_HELP_STRING([--with-openmp],[enable OpenMP parallelization])], with_omp=$withval, with_omp=no)
if test "x$with_omp" = xyes; then
   AC_DEFINE([USE_OPENMP], [1], [Define to use OpenMP threading.])
   AX_OPENMP([], [AC_MSG_ERROR([Could not find OpenMP flag; configure with --without-openmp])])
   CXXFLAGS="$CXXFLAGS $OPENMP_CXXFLAGS"
fi

# BLAS and LAPACK
AX_BLAS([], [AC_MSG_ERROR([BLAS was not found!])])
AX_LAPACK([], [AC_MSG_ERROR([LAPACK was not found!])])

LIBS="$LAPACK_LIBS $BLAS_LIBS $LIBS $FLIBS"

AC_CONFIG_FILES([Makefile])
AC_OUTPUT
