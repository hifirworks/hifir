CXX = g++
CXXFLAGS = -DNDEBUG -g -rdynamic -march=native -O3 -std=c++11

MC64_ROOT = hsl_mc64-2.3.1
PSMILU_INCLUDE = ../src

_INCS = -I$(PSMILU_INCLUDE) -I$(MC64_ROOT)/include
LIBS = $(MC64_ROOT)/lib/libhsl_mc64.a
LAPACKBLAS_LIB = -lopenblas
LDFLAGS =

.PHONY: driver clean

driver: GMRES.hpp driver.cpp
	$(CXX) $(CXXFLAGS) $(_INCS) driver.cpp -o $@ $(LIBS) $(LAPACKBLAS_LIB) -pthread -lgfortran -lm $(LDFLAGS)

mt_scaling_analysis: GMRES.hpp GMRES_MT.hpp mt_scaling_analysis.cpp
	$(CXX) $(CXXFLAGS) -I$(PSMILU_INCLUDE) mt_scaling_analysis.cpp -o $@ -fopenmp

clean:
	rm -f driver mt_scaling_analysis
