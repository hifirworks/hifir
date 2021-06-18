#-----------------------------------------------------------------------------#
#              This file is part of the HIFIR library                         #
#-----------------------------------------------------------------------------#

include ../make.inc

lib: libhifir.so libhifir-debug.so

libhifir.so: libhifir.so.0
	ln -s $< $@

libhifir.so.0: libhifir.so.0.1.0
	ln -s $< $@

libhifir.so.0.1.0: libhifir.o
	$(CXX) -shared -fPIC -Wl,-soname,libhifir.so.0 -o $@ $< $(LIBHIFIR_LIBS) \
		$(LIBHIFIR_LDFLAGS)

libhifir-debug.so: libhifir-debug.so.0
	ln -s $< $@

libhifir-debug.so.0: libhifir-debug.so.0.1.0
	ln -s $< $@

libhifir-debug.so.0.1.0: libhifir-debug.o
	$(CXX) -shared -fPIC -Wl,-soname,libhifir-debug.so.0 -o $@ $< \
		$(LIBHIFIR_LIBS) $(LIBHIFIR_LDFLAGS)

libhifir.o: libhifir.cpp libhifir.h
	$(CXX) -c $(LIBHIFIR_CXXFLAGS) $(LIBHIFIR_OPT_FLAGS) $<

libhifir-debug.o: libhifir.cpp libhifir.h
	$(CXX) -c $(LIBHIFIR_CXXFLAGS) $(LIBHIFIR_DBG_FLAGS) $< -o $@

clean:
	rm -f *.so* *.o

.PHONY: clean
