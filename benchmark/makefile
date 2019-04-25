CXX = g++
DEBUG = -DNDEBUG -fno-omit-frame-pointer -fsanitize=address
CXXFLAGS = -g $(DEBUG) -rdynamic -march=native -O3 -std=c++11

MC64_ROOT = hsl_mc64-2.3.1
PSMILU_INCLUDE = ../src

_INCS = -I$(PSMILU_INCLUDE) -I$(MC64_ROOT)/include
LIBS = $(MC64_ROOT)/lib/libhsl_mc64.a
LAPACKBLAS_LIB = -lopenblas
LDFLAGS =
MACROS =

.PHONY: driver clean

driver: GMRES.hpp driver.cpp
	$(CXX) $(CXXFLAGS) $(_INCS) driver.cpp -o $@ $(LIBS) $(LAPACKBLAS_LIB) -pthread -lgfortran -lm $(LDFLAGS)

driver_bjcb: GMRES.hpp driver_bjcb.cpp
	$(CXX) $(CXXFLAGS) $(_INCS) driver_bjcb.cpp -o $@ $(LIBS) $(LAPACKBLAS_LIB) -pthread -lgfortran -lm $(LDFLAGS) -lmetis

mt_driver: GMRES_MT.hpp mt_driver.cpp
	$(CXX) $(CXXFLAGS) $(_INCS) mt_driver.cpp -o $@ $(LIBS) $(LAPACKBLAS_LIB) -pthread -lgfortran -lm $(LDFLAGS) -fopenmp

mt_driver_dbg: GMRES_MT.hpp mt_driver.cpp
	$(CXX) $(CXXFLAGS) $(_INCS) mt_driver.cpp -o $@ $(LIBS) $(LAPACKBLAS_LIB) -pthread -lgfortran -lm $(LDFLAGS) -fopenmp

mt_L: mt_L.cpp sch_solve.hpp
	$(CXX) $(CXXFLAGS) $(_INCS) mt_L.cpp -o $@ $(LIBS) $(LAPACKBLAS_LIB) -pthread -lgfortran -lm $(LDFLAGS) -fopenmp

bjcb: GMRES.hpp bjcb.cpp
	$(CXX) $(CXXFLAGS) $(_INCS) bjcb.cpp -o $@ $(LIBS) $(LAPACKBLAS_LIB) -pthread -lgfortran -lm $(LDFLAGS) -lmetis

bj_solve: BJacobi.hpp bj_solve.cpp
	$(CXX) $(CXXFLAGS) $(_INCS) bj_solve.cpp -o $@ $(LIBS) $(LAPACKBLAS_LIB) -pthread -lgfortran -lm $(LDFLAGS) -lmetis

do_diag: do_diag.cpp
	$(CXX) $(CXXFLAGS) $(_INCS) do_diag.cpp -o $@ $(LIBS) $(LAPACKBLAS_LIB) -pthread -lgfortran -lm $(LDFLAGS)

driver_pre: FGMRES.hpp driver_pre.cpp
	$(CXX) $(CXXFLAGS) $(MACROS) $(_INCS) driver_pre.cpp -o $@ $(LIBS) $(LAPACKBLAS_LIB) -pthread -lgfortran -lm $(LDFLAGS)

driver_jacobi: FGMRES.hpp driver_jacobi.cpp
	$(CXX) $(CXXFLAGS) $(MACROS) $(_INCS) driver_jacobi.cpp -o $@ $(LIBS) $(LAPACKBLAS_LIB) -pthread -lgfortran -lm $(LDFLAGS)

driver_cjacobi: FGMRES.hpp driver_cjacobi.cpp
	$(CXX) $(CXXFLAGS) $(MACROS) $(_INCS) driver_cjacobi.cpp -o $@ $(LIBS) $(LAPACKBLAS_LIB) -pthread -lgfortran -lm $(LDFLAGS)

driver_defer: FGMRES.hpp driver_defer.cpp
	$(CXX) $(CXXFLAGS) $(MACROS) $(_INCS) driver_defer.cpp -o $@ $(LIBS) $(LAPACKBLAS_LIB) -pthread -lgfortran -lm $(LDFLAGS)

driver_pvt: FGMRES.hpp driver_pvt.cpp
	$(CXX) $(CXXFLAGS) $(MACROS) $(_INCS) driver_pvt.cpp -o $@ $(LIBS) $(LAPACKBLAS_LIB) -pthread -lgfortran -lm $(LDFLAGS)

clean:
	rm -f driver mt_driver mt_L mt_driver_dbg driver_bjcb bjcb bj_solve driver_pvt driver_pre driver_jacobi driver_cjacobi driver_defer
