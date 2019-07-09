# automatic makefile for unit testing
# authors:
#		Qiao,

USE_GTEST = 1

include ../Makefile.in

TEST_SRCS = $(shell \ls test_*.cpp)
TESTS = $(TEST_SRCS:%.cpp=%)
EXECS = $(TEST_SRCS:%.cpp=%.exe)
VPATH = $(PSMILU_ROOT)/src
LAPACK_LIB =
ifneq ($(LAPACK_LIB),)
# add lapack blas libraries, use generic libraries for testing
	LIBS += $(LAPACK_LIB)
endif
.PHONY: clean
.PRECIOUS: $(EXECS)

all: $(TESTS)

test_%: test_%.exe
	./$<

define MAKE_EXE_RULE =
$(1).exe: $(1).cpp common.hpp $$(shell ./get_deps.sh $(1))
	$(CXX) $(CXXFLAGS) $$< -o $$@ $(LIBS) $(LDFLAGS)
endef

$(foreach test,$(TESTS),$(eval $(call MAKE_EXE_RULE, $(test))))

clean:
	@rm -f $(EXECS) *.dSYM
