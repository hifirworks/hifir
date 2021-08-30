# Commands
CXX = c++
AR = ar
ARFLAGS = cr
RANLIB = ranlib

# Options
USE_OPENMP = 1
PREFIX =
INSTALL_PREFIX = $(or $(PREFIX),/usr/local)

HIFIR_HOME = ../src
SO_MAJOR = $(shell grep -i "define hif_global" ${HIFIR_HOME}/hif/version.h|awk '{print $$3}')
SO_MINOR = $(shell grep -i "define hif_major" ${HIFIR_HOME}/hif/version.h|awk '{print $$3}')
SO_PATCH = $(shell grep -i "define hif_minor" ${HIFIR_HOME}/hif/version.h|awk '{print $$3}')
SO_IMPL_AGE = ${SO_MINOR}.${SO_PATCH}
SO_SUFFIX = ${SO_MAJOR}.${SO_IMPL_AGE}
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
	${CXX} -shared -Wl,-soname,$*.so.${SO_MAJOR} -o ${@}.${SO_SUFFIX} $< ${LAPACK_LIBS} ${OPENMP_CXXFLAGS} ${LDFLAGS}
	ln -sf $*.so.${SO_SUFFIX} ${@}.${SO_MAJOR}
	ln -sf $*.so.${SO_MAJOR} $@

libhifir_i32.o: src/libhifir.cpp include/libhifir.h
	${CXX} -Iinclude -c ${LIBHIFIR_CXXFLAGS} ${CPPFLAGS} $< -o $@

libhifir_i64.o: src/libhifir.cpp include/libhifir.h
	${CXX} -Iinclude -c ${LIBHIFIR_CXXFLAGS} ${CPPFLAGS} -DLIBHIFIR_INT_SIZE=64 $< -o $@

install: libs
	@mkdir -p ${INSTALL_PREFIX}/include
	@mkdir -p ${INSTALL_PREFIX}/lib
	cp -d lib/*.so* ${INSTALL_PREFIX}/lib
	cp -d lib/*.a ${INSTALL_PREFIX}/lib
	cp -d include/libhifir.h ${INSTALL_PREFIX}/include

.PHONY: install clean test libs

test: libs
	make -C tests PREFIX=$(or $(PREFIX),$(PWD))
	(cd tests; ./test_real.exe)
	(cd tests; ./test_complex.exe)

clean:
	rm -f lib/*.* *.o
	make -C tests clean
