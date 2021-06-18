#-----------------------------------------------------------------------------#
#              This file is part of the HIFIR library                         #
#-----------------------------------------------------------------------------#

CPPFLAGS = -I../src
include ../Makefile.in

DEMOS_SRC = demo.cpp
DEMOS = $(DEMOS_SRC:%.cpp=%)
DEMOS_EXE = $(DEMOS_SRC:%.cpp=%.exe)

.PHONY: clean
.PRECIOUS: $(DEMOS_EXE)

all: $(DEMOS_EXE)

define MAKE_EXE_RULE =
$(1).d: $(1).cpp
	@$(CXX) $(CPPFLAGS) -MM -MP -MT $$@ -MF $$@ $$<
include $(1).d

$(1).exe: $(1).cpp $(1).d
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $$< -o $$@ $(LDFLAGS) $(LIBS)
endef

ifneq ("$(MAKECMDGOALS)","clean")
    $(foreach test,$(DEMOS),$(eval $(call MAKE_EXE_RULE, $(test))))
endif

demo_libhifir.exe: demo_libhifir.cpp
	$(HIF_CXX) $(HIF_CXXFLAGS) -I../src -I../libhifir $< -o $@ \
		-L../libhifir -lhifir -Wl,-rpath,../libhifir

clean:
	@rm -f $(DEMOS_EXE) *.dSYM *.d
