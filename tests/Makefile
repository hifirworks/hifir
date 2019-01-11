# automatic makefile for unit testing
# authors:
#		Qiao,

include ../Makefile.in

TEST_SRCS = $(shell \ls test_*.cpp)
TESTS = $(TEST_SRCS:%.cpp=%)
EXECS = $(TEST_SRCS:%.cpp=%.exe)
VPATH = $(PSMILU_ROOT)/include

.PHONY: clean
.PRECIOUS: $(EXECS)

all: $(TESTS)

test_%: test_%.exe
	./$<

define MAKE_EXE_RULE =
$(1).exe: $(1).cpp $$(shell ./get_deps.sh $(1))
	$(CXX) $(CXXFLAGS) $$< -o $$@ $(LIBS) $(LDFLAGS)
endef

$(foreach test,$(TESTS),$(eval $(call MAKE_EXE_RULE, $(test))))

clean:
	@rm -f $(EXECS) *.dSYM
