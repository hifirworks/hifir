CXX = g++
CXXFLAGS = -std=c++11 -g

all: demo.exe demo2.exe
	@echo "Run amd_demo...\n\n"
	@./demo.exe
	@echo "\nRun amd_demo2...\n\n"
	@./demo2.exe

demo.exe: demo.cpp ../amd.hpp
	$(CXX) $(CXXFLAGS) demo.cpp -o $@

demo2.exe: demo2.cpp ../amd.hpp
	$(CXX) $(CXXFLAGS) demo2.cpp -o $@

clean:
	rm -f *.exe
