#-----------------------------------------------------------------------------#
#              This file is part of the HIFIR library                         #
#-----------------------------------------------------------------------------#

CC = cc
PREFIX =

ifneq ($(PREFIX),)
	CPPFLAGS = -I$(PREFIX)/include
	LIB_PATH = -L$(PREFIX)/lib
	LDFLAGS = -Wl,-rpath,$(PREFIX)/lib
endif

.PHONY: clean

all: test_real.exe test_complex.exe

all_ext: test_real.exe test_complex.exe test_real_i64.exe test_complex_i64.exe

test_real.exe: test_real.c
	$(CC) -g -O2 $(CPPFLAGS) $< -o $@ -lm $(LIB_PATH) -lhifir_i32 $(LDFLAGS)

test_real_i64.exe: test_real.c
	$(CC) -g -O2 -DLIBHIFIR_INT_SIZE=64 $(CPPFLAGS) $< -o $@ -lm $(LIB_PATH) -lhifir_i64 $(LDFLAGS)

test_complex.exe: test_complex.c
	$(CC) -g -O2 $(CPPFLAGS) $< -o $@ -lm $(LIB_PATH) -lhifir_i32 $(LDFLAGS)

test_complex_i64.exe: test_complex.c
	$(CC) -g -O2 -DLIBHIFIR_INT_SIZE=64 $(CPPFLAGS) $< -o $@ -lm $(LIB_PATH) -lhifir_i64 $(LDFLAGS)

clean:
	@rm -f *.exe *.dSYM
