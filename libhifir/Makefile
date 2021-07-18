#-----------------------------------------------------------------------------#
#              This file is part of the HIFIR library                         #
#-----------------------------------------------------------------------------#

CXX = g++

ifneq ($(HIFIR_HOME),)
    CPPFLAGS = -I$(HIFIR_HOME)/include
else
    CPPFLAGS = -I../src
endif

LIBHIFIR_CXXFLAGS = -g -fPIC -fopenmp -Wall -Wno-comment $(CXXFLAGS)
LIBHIFIR_OPT_FLAGS = $(LIBHIFIR_CXXFLAGS) -O3 -march=native -ffast-math \
	-DNDEBUG $(CPPFLAGS)
LIBHIFIR_DBG_FLAGS = $(LIBHIFIR_CXXFLAGS) -O2 -DHIF_DEBUG $(CPPFLAGS)
LIBHIFIR_LIBS = $(LIBS)
ifeq ($(LIBHIFIR_LAPACK_LIB),)
	LIBHIFIR_LIBS += -llapack -lblas
else
	LIBHIFIR_LIBS += $(LIBHIFIR_LAPACK_LIB)
endif
LIBHIFIR_LIBS  += -fopenmp -pthread -lm
LIBHIFIR_LDFLAGS = $(LDFLAGS)
CURRENT_USER = $(shell echo $$USER)
ifeq ($(CURRENT_USER), root)
	DEFAULT_PREFIX = /opt/libhifir
else
	DEFAULT_PREFIX = $(shell echo $$HOME)/.local
endif
ifeq ($(PREFIX),)
	PREFIX = $(DEFAULT_PREFIX)
endif

default: libhifir.so

lib: libhifir.so libhifir-debug.so

libhifir.so: libhifir.so.0
	ln -s $< $@

libhifir.so.0: libhifir.so.0.1.0
	ln -s $< $@

libhifir.so.0.1.0: libhifir.o
	$(CXX) -shared -fPIC -Wl,-soname,libhifir.so.0 -o $@ $< \
		$(LIBHIFIR_LDFLAGS) $(LIBHIFIR_LIBS)
	@chmod a-x $@

libhifir-debug.so: libhifir-debug.so.0
	ln -s $< $@

libhifir-debug.so.0: libhifir-debug.so.0.1.0
	ln -s $< $@

libhifir-debug.so.0.1.0: libhifir-debug.o
	$(CXX) -shared -fPIC -Wl,-soname,libhifir-debug.so.0 -o $@ $< \
		$(LIBHIFIR_LDFLAGS) $(LIBHIFIR_LIBS)
	@chmod a-x $@

libhifir.o: libhifir.cpp libhifir.h
	$(CXX) -c $(LIBHIFIR_OPT_FLAGS) $<

libhifir-debug.o: libhifir.cpp libhifir.h
	$(CXX) -c $(LIBHIFIR_DBG_FLAGS) $< -o $@

clean:
	rm -f *.so* *.o

.PHONY: clean

install: default
	@mkdir -p $(PREFIX)/include
	@mkdir -p $(PREFIX)/lib
	cp -a *.so* $(PREFIX)/lib
	cp -a libhifir.h $(PREFIX)/include