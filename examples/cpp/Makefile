# ----------------------------------------------------------------------------
#                Copyright (C) 2019 The HILUCSI AUTHORS
# ----------------------------------------------------------------------------

# This is a makefile for Ubuntu, assuming MKL is installed

HILUCSI_CXX = $(CXX)
ifeq ($(HILUCSI_CXX),)
	HILUCSI_CXX = g++
endif
HILUCSI_CXXFLAGS = $(CXXFLAGS)
DEMO_FLAGS = -g -m64 -O3 -march=native -std=c++11
ifeq ($(DEBUG),)
	DEMO_FLAGS += -DNDEBUG
endif
HILUCSI_LDFLAGS = $(LDFLAGS)
HILUCSI_LAPACK_LIB = -llapack -lblas
HILUCSI_LIBS = $(LIBS) $(HILUCSI_LAPACK_LIB) -lm
HILUCSI_ROOT = ../../src
VPATH = ${HILUCSI_ROOT}
DEMO_CASES = $(shell \ls *.cpp)
DEMOS = $(DEMO_CASES:%.cpp=%.exe)

.PHONY: clean
.PRECIOUS: $(DEMOS)

all: $(DEMOS)

%.exe: %.cpp HILUCSI.hpp
	$(CXX) $(DEMO_FLAGS) $(HILUCSI_CXXFLAGS) -I$(HILUCSI_ROOT) $< -o $@ \
		$(HILUCSI_LIBS) $(HILUCSI_LDFLAGS)

clean:
	rm -f $(DEMOS)
