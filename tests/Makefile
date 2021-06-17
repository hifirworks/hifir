###############################################################################
#                   This file is part of the HIFIR library                    #
###############################################################################

# automatic makefile for unit testing
# authors:
#		Qiao Chen
#		Xiangmin Jiao

DEBUG = 1
HIFIR_HOME = '../'

include ../Makefile.in

# Specify the path for GTEST. If not defined, search for it in common places.
GTEST_DIRS = /opt/local $HOME/.local
GTEST_HOME := $(shell for d in $(GTEST_DIRS); \
	do if [ -r $$d/include/gtest.h ]; then echo $$d; break; fi; done)
ifeq ($(GTEST_INCDIR),)
	GTEST_INCDIR = $(GTEST_HOME)/include
endif
ifeq ($(GTEST_LIBDIR),)
	GTEST_LIBDIR = $(GTEST_HOME)/lib
endif

CPPFLAGS += -I$(GTEST_INCDIR)
LIBS += -L$(GTEST_LIBDIR) -lgtest -lgtest_main

TEST_SRCS = $(shell \ls test_*.cpp)
TESTS = $(TEST_SRCS:%.cpp=%)
EXECS = $(TEST_SRCS:%.cpp=%.exe)
VPATH = $(HIFIR_HOME)/src

.PHONY: clean
.PRECIOUS: $(EXECS)

all: $(TESTS)

test_%: test_%.exe
	./$<

define MAKE_EXE_RULE =
$(1).d: $(1).cpp | $(DEPDIR)
	@$(CXX) $(CPPFLAGS) -MM -MP -MT $$@ -MF $$@ $$<
include $(1).d

$(1).exe: $(1).cpp common.hpp $(1).d
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $$< -o $$@ $(LIBS) $(LDFLAGS)
endef

ifneq ("$(MAKECMDGOALS)","clean")
$(foreach test,$(TESTS),$(eval $(call MAKE_EXE_RULE, $(test))))
endif

clean:
	@rm -f $(EXECS) *.dSYM foo.bin foo?.hif *.d
