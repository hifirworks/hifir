# Commands
CXX = c++
AR = ar
ARFLAGS = cr
RANLIB = ranlib

# Options
USE_OPENMP = 1
PREFIX = /usr/local

SO_MAJOR = 1
SO_IMPL_AGE = 0.0
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

lib: libhifir_i32.a libhifir_i32.so libhifir_i64.a libhifir_i64.so

libhifir_i32.a: libhifir_i32.o
	${AR} ${ARFLAGS} $@ $<
	${RANLIB} $@

libhifir_i32.so: libhifir_i32.o
	${CXX} -shared -Wl,-soname,${@}.${SO_MAJOR} -o ${@}.${SO_SUFFIX} $< ${LAPACK_LIBS} ${OPENMP_CXXFLAGS} ${LDFLAGS}
	ln -sf ${@}.${SO_SUFFIX} ${@}.${SO_MAJOR}
	ln -sf ${@}.${SO_MAJOR} $@

libhifir_i32.o: libhifir.cpp libhifir.h
	${CXX} -c ${LIBHIFIR_CXXFLAGS} ${CPPFLAGS} $< -o $@

libhifir_i64.a: libhifir_i64.o
	${AR} ${ARFLAGS} $@ $<
	${RANLIB} $@

libhifir_i64.so: libhifir_i64.o
	${CXX} -shared -Wl,-soname,${@}.${SO_MAJOR} -o ${@}.${SO_SUFFIX} $< ${LAPACK_LIBS} ${OPENMP_CXXFLAGS} ${LDFLAGS}
	ln -sf ${@}.${SO_SUFFIX} ${@}.${SO_MAJOR}
	ln -sf ${@}.${SO_MAJOR} $@

libhifir_i64.o: libhifir.cpp libhifir.h
	${CXX} -c ${LIBHIFIR_CXXFLAGS} ${CPPFLAGS} -DLIBHIFIR_INT_SIZE=64 $< -o $@

install:
	@mkdir -p ${PREFIX}/include
	@mkdir -p ${PREFIX}/lib
	cp -a *.so* ${PREFIX}/lib
	cp -a *.a ${PREFIX}/lib
	cp -a libhifir.h ${PREFIX}/include

.PHONY: clean

clean:
	rm -f *.so* *.a *.o
