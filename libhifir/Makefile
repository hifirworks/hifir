#-----------------------------------------------------------------------------#
#              This file is part of the HIFIR library                         #
#-----------------------------------------------------------------------------#

CXX = g++

ifneq ($(HIFIR_HPP),)
	HIFIR_HOME = $(HIFIR_HPP)
else
    HIFIR_HOME = ../src
endif
CPPFLAGS = -I$(HIFIR_HOME)

GLOBAL_VERSION = $(shell grep -i "define hif_global" $(HIFIR_HOME)/hif/version.h|awk '{print $$3}')
MAJOR_VERSION = $(shell grep -i "define hif_major" $(HIFIR_HOME)/hif/version.h|awk '{print $$3}')
MINOR_VERSION = $(shell grep -i "define hif_minor" $(HIFIR_HOME)/hif/version.h|awk '{print $$3}')

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

.PRECIOUS: \
	libhifir.so.$(GLOBAL_VERSION).$(MAJOR_VERSION) \
	libhifir_i64.so.$(GLOBAL_VERSION).$(MAJOR_VERSION) \
	libhifir-debug.so.$(GLOBAL_VERSION).$(MAJOR_VERSION) \
	libhifir_i64-debug.so.$(GLOBAL_VERSION).$(MAJOR_VERSION)

default: libhifir.so

all: libhifir.so libhifir_i64.so

debug: libhifir-debug.so

all-debug: libhifir-debug.so libhifir_i64-debug.so

%.so: %.so.$(GLOBAL_VERSION).$(MAJOR_VERSION)
	ln -sf $< $@

%.so.$(GLOBAL_VERSION).$(MAJOR_VERSION): %.o
	$(CXX) -shared -fPIC -Wl,-soname,$@ -o ${@}.${MINOR_VERSION} $< \
		$(LIBHIFIR_LDFLAGS) $(LIBHIFIR_LIBS)
	@chmod a-x ${@}.0
	ln -sf ${@}.0 $@

libhifir.o: libhifir.cpp libhifir.h
	$(CXX) -c $(LIBHIFIR_OPT_FLAGS) $<

libhifir_i64.o: libhifir.cpp libhifir.h
	$(CXX) -c $(LIBHIFIR_OPT_FLAGS) -DLIBHIFIR_INT_SIZE=64 $< -o $@

libhifir-debug.o: libhifir.cpp libhifir.h
	$(CXX) -c $(LIBHIFIR_DBG_FLAGS) $< -o $@

libhifir_i64-debug.o: libhifir.cpp libhifir.h
	$(CXX) -c $(LIBHIFIR_DBG_FLAGS) -DLIBHIFIR_INT_SIZE=64 $< -o $@

clean:
	rm -f *.so* *.o

.PHONY: clean

install: default
	@mkdir -p $(PREFIX)/include
	@mkdir -p $(PREFIX)/lib
	cp -a *.so* $(PREFIX)/lib
	cp -a libhifir.h $(PREFIX)/include
