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

demo_real.exe: demo_real.c
	$(CC) -g -O2 $(CPPFLAGS) $< -o $@ -lm $(LIB_PATH) -lhifir $(LDFLAGS)

demo_real_i64.exe: demo_real.c
	$(CC) -g -O2 -DLIBHIFIR_INT_SIZE=64 $(CPPFLAGS) $< -o $@ -lm $(LIB_PATH) -lhifir_i64 $(LDFLAGS)

clean:
	@rm -f *.exe *.dSYM
