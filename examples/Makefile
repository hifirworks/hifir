#-----------------------------------------------------------------------------#
#              This file is part of the HIFIR library                         #
#-----------------------------------------------------------------------------#

CPPFLAGS = -I../src -I.
include ../Makefile.in

DEMOS_SRC = $(shell \ls beginner/demo_*.cpp) \
	$(shell \ls intermediate/demo_*.cpp) \
	$(shell \ls advanced/demo_*.cpp)
DEMOS = $(DEMOS_SRC:%.cpp=%)
DEMOS_EXE = $(DEMOS_SRC:%.cpp=%.exe)

.PHONY: clean
.PRECIOUS: $(DEMOS_EXE)

all: $(DEMOS_EXE)

define MAKE_EXE_RULE =
$(1).d: $(1).cpp
	@$(CXX) $(CPPFLAGS) $(CXXFLAGS) -MM -MP -MT $$@ -MF $$@ $$<
include $(1).d

$(1).exe: $(1).cpp $(1).d
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $$< -o $$@ $(LDFLAGS) $(LIBS)
endef

ifneq ("$(MAKECMDGOALS)","clean")
    $(foreach test,$(DEMOS),$(eval $(call MAKE_EXE_RULE, $(test))))
endif

clean:
	@rm -f $(DEMOS_EXE) *.dSYM $(DEMOS_SRC:%.cpp=%.d) *.exe

