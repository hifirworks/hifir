include ../../Makefile.in

TEST_SRCS = $(shell \ls test_*.cpp)
TESTS = $(TEST_SRCS:%.cpp=%)
EXECS = $(TEST_SRCS:%.cpp=%.exe)
VPATH = $(PSMILU_ROOT)/bench

.PHONY: clean
.PRECIOUS: $(EXECS)

all: $(TESTS)

test_%: test_%.exe
	./$<

test_%.exe: test_%.cpp GMRES.hpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LIBS) $(LDFLAGS)

clean:
	@rm -f $(EXECS) *.dSYM