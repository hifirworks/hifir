# ----------------------------------------------------------------------------
#                Copyright (C) 2021 NumGeom Group at Stony Brook University
# ----------------------------------------------------------------------------

# This is a makefile for Ubuntu, assuming MKL is installed

HIF_CXX = $(CXX)
ifeq ($(HIF_CXX),)
	HIF_CXX = g++
endif
HIF_CXXFLAGS = $(CXXFLAGS)
DEMO_FLAGS = -g -m64 -O3 -march=native -std=c++11
ifeq ($(DEBUG),)
	DEMO_FLAGS += -DNDEBUG
endif
HIF_LDFLAGS = $(LDFLAGS)
HIF_LAPACK_LIB = -llapack -lblas
HIF_LIBS = $(LIBS) $(HIF_LAPACK_LIB) -lm
HIF_ROOT = ../../src
VPATH = ${HIF_ROOT}
DEMO_CASES = $(shell \ls *.cpp)
DEMOS = $(DEMO_CASES:%.cpp=%.exe)

.PHONY: clean
.PRECIOUS: $(DEMOS)

all: $(DEMOS)

%.exe: %.cpp HIF.hpp
	$(CXX) $(DEMO_FLAGS) $(HIF_CXXFLAGS) -I$(HIF_ROOT) $< -o $@ \
		$(HIF_LIBS) $(HIF_LDFLAGS)

clean:
	rm -f $(DEMOS)
