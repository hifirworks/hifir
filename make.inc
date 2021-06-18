#-----------------------------------------------------------------------------#
#              This file is part of the HIFIR library                         #
#-----------------------------------------------------------------------------#

CXX = g++
LIBHIFIR_CXXFLAGS = $(CXXFLAGS) -g -fPIC -fopenmp -Wall -Wno-comment -I../src -I.
LIBHIFIR_OPT_FLAGS = -O3 -ffast-math -DNDEBUG
LIBHIFIR_DBG_FLAGS = -O2
LIBHIFIR_LIBS = $(LIBS) -fopenmp -pthread
ifeq ($(LIBHIFIR_LAPACK_LIB),)
	LIBHIFIR_LIBS += -llapack -lblas
else
	LIBHIFIR_LIBS += $(LIBHIFIR_LAPACK_LIB)
endif
LIBHIFIR_LDFLAGS = $(LDFLAGS)
