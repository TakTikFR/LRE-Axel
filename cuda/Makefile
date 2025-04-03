CC = nvcc
CXXFLAGS = -std=c++20 -Iinclude --expt-relaxed-constexpr -Wno-deprecated-gpu-targets -g -G -rdc=true
MAIN = main.cpp
CPP_FILES = $(wildcard src/*.cpp)
CU_FILES = $(wildcard src/*.cu)
OBJ_FILES = $(CPP_FILES:.cpp=.o) $(CU_FILES:.cu=.o) $(MAIN:.cpp=.o)

%.o: %.cpp
	$(CC) $(CXXFLAGS) -dc $< -o $@

%.o: %.cu
	$(CC) $(CXXFLAGS) -dc $< -o $@

executable: $(OBJ_FILES)
	$(CC) $(CXXFLAGS) $(OBJ_FILES) -o $@

clean:
	rm -f src/*.o *.o executable
