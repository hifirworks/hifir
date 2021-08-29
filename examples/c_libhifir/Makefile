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

all: demo_real.exe demo_complex.exe

all_ext: demo_real.exe demo_complex.exe demo_real_i64.exe demo_complex_i64.exe

demo_real.exe: demo_real.c
	$(CC) -g -O2 $(CPPFLAGS) $< -o $@ -lm $(LIB_PATH) -lhifir_i32 $(LDFLAGS)

demo_real_i64.exe: demo_real.c
	$(CC) -g -O2 -DLIBHIFIR_INT_SIZE=64 $(CPPFLAGS) $< -o $@ -lm $(LIB_PATH) -lhifir_i64 $(LDFLAGS)

demo_complex.exe: demo_complex.c
	$(CC) -g -O2 $(CPPFLAGS) $< -o $@ -lm $(LIB_PATH) -lhifir_i32 $(LDFLAGS)

demo_complex_i64.exe: demo_complex.c
	$(CC) -g -O2 -DLIBHIFIR_INT_SIZE=64 $(CPPFLAGS) $< -o $@ -lm $(LIB_PATH) -lhifir_i64 $(LDFLAGS)

clean:
	@rm -f *.exe *.dSYM
