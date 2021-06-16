#-----------------------------------------------------------------------------#
#              This file is part of the HIFIR library                         #
#-----------------------------------------------------------------------------#

HIF_CXX = $(CXX)
ifeq ($(HIF_CXX),)
	HIF_CXX = g++
endif
HIF_CXXFLAGS = $(CXXFLAGS)
HIF_CXXFLAGS += -g -O3 -ffast-math -std=c++11 -march=native
HIF_LIBS = $(LIBS)
ifeq ($(HIF_LAPACK_LIB),)
	HIF_LIBS += -llapack -lblas
else
	HIF_LIBS += $(HIF_LAPACK_LIB)
endif
HIF_LDFLAGS = $(LDFLAGS)

.PHONY: clean
.PRECIOUS: demo

demo: demo.cpp
	$(HIF_CXX) $(HIF_CXXFLAGS) -I../src $< -o $@ $(HIF_LIBS) $(HIF_LDFLAGS)

clean:
	rm -f demo
