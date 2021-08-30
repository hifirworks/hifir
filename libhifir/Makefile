# Commands
CXX = c++
AR = ar
ARFLAGS = cr
RANLIB = ranlib

# Options
USE_OPENMP = 1
PREFIX = /usr/local

SO_MAJOR = 0
SO_IMPL_AGE = 2.0
SO_SUFFIX = ${SO_MAJOR}.${SO_IMPL_AGE}

HIFIR_HOME = ../src
CPPFLAGS = -I${HIFIR_HOME}
ifeq (${USE_OPENMP},1)
	OPENMP_CXXFLAGS = -fopenmp
endif
OPTFLAGS = -O3 -ffast-math -DNDEBUG
LIBHIFIR_CXXFLAGS = -fPIC ${OPENMP_CXXFLAGS} ${OPTFLAGS} ${CXXFLAGS}
ifeq (${LAPACK_LIBS},)
	LAPACK_LIBS = -llapack -lblas
endif

libs: lib/libhifir_i32.a lib/libhifir_i64.a lib/libhifir_i32.so lib/libhifir_i64.so

.SUFFIXES: .o .a .so

lib/%.a: %.o
	mkdir -p lib
	${AR} ${ARFLAGS} $@ $<
	${RANLIB} $@

lib/%.so: %.o
	mkdir -p lib
	${CXX} -shared -Wl,-soname,${@}.${SO_MAJOR} -o ${@}.${SO_SUFFIX} $< ${LAPACK_LIBS} ${OPENMP_CXXFLAGS} ${LDFLAGS}
	ln -sf $*.so.${SO_SUFFIX} ${@}.${SO_MAJOR}
	ln -sf $*.so.${SO_MAJOR} $@

libhifir_i32.o: src/libhifir.cpp include/libhifir.h
	${CXX} -Iinclude -c ${LIBHIFIR_CXXFLAGS} ${CPPFLAGS} $< -o $@

libhifir_i64.o: src/libhifir.cpp include/libhifir.h
	${CXX} -Iinclude -c ${LIBHIFIR_CXXFLAGS} ${CPPFLAGS} -DLIBHIFIR_INT_SIZE=64 $< -o $@

install: libs
	@mkdir -p ${PREFIX}/include
	@mkdir -p ${PREFIX}/lib
	cp -d lib/*.so* ${PREFIX}/lib
	cp -d lib/*.a ${PREFIX}/lib
	cp -d include/libhifir.h ${PREFIX}/include

.PHONY: install clean test libs

test: libs
	make -C tests PREFIX=$(PWD)
	(cd tests; ./test_real.exe)
	(cd tests; ./test_complex.exe)

clean:
	rm -f lib/*.* *.o
