DOXYGEN = doxygen
LATEX_BATCHMODE = YES

all:
	@env LATEX_BATCHMODE=$(LATEX_BATCHMODE) $(DOXYGEN) Doxyfile

pdf: all
	$(MAKE) -C latex

clean:
	@rm -rf html latex xml

.PHONY: clean all pdf
