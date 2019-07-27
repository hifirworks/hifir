# ----------------------------------------------------------------------------
#                Copyright (C) 2019 The HILUCSI AUTHORS
# ----------------------------------------------------------------------------

# This is a makefile for Ubuntu, assuming MKL is installed

CXX = g++
CXXFLAGS =
DEMO_FLAGS = -g -m64 -O3 -march=native -std=c++11
ifeq ($(DEBUG),)
	DEMO_FLAGS += -DNDEBUG
endif
MKL_ROOT = /opt/intel/mkl
MKL_LIB_ROOT = ${MKL_ROOT}/lib/intel64
MKL_LIBS = -lmkl_intel_lp64 -lmkl_sequential -lmkl_core
LDFLAGS =
LDMKL = -Wl,-rpath,${MKL_LIB_ROOT}
LIBS =
HILUCSI_ROOT = ../../src
VPATH = ${HILUCSI_ROOT}
DEMO_CASES = $(shell \ls *.cpp)
DEMOS = $(DEMO_CASES:%.cpp=%.exe)

.PHONY: clean
.PRECIOUS: $(DEMOS)

all: $(DEMOS)

%.exe: %.cpp HILUCSI.hpp
	$(CXX) $(DEMO_FLAGS) $(CXXFLAGS) -I$(HILUCSI_ROOT) $< -o $@ \
		$(LIBS) -L${MKL_LIB_ROOT} -Wl,--no-as-needed ${MKL_LIBS} \
		-fopenmp -pthread -lgfortran -lm $(LDMKL) $(LDFLAGS)

clean:
	rm -f $(DEMOS)
