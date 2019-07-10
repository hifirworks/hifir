DOXYGEN = doxygen

all:
	$(DOXYGEN) Doxyfile

pdf: all
	$(MAKE) -C latex

clean:
	@rm -rf html latex xml

.PHONY: clean all pdf
