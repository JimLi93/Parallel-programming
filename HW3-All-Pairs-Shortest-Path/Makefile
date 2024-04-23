CC = gcc
CXX = g++
NVCC = nvcc
CXXFLAGS = -O3 -lm -fopenmp
CFLAGS = -O3 -lm -fopenmp
NVFLAGS = -std=c++11 -O3 -Xptxas="-v" -arch=sm_61 -Xcompiler -fopenmp
LDFLAGS = -lm
TARGETS = hw3-1 hw3-2 hw3-3

.PHONY: all
all: $(TARGETS)

.PHONY: clean
clean:
	rm -f $(TARGETS)

hw3-1: hw3-1.cc
	$(CXX) $(CFLAGS) -o $@ $?

hw3-2: hw3-2.cu
	$(NVCC) $(NVFLAGS) $(LDFLAGS) -o $@ $?

hw3-3: hw3-3.cu
	$(NVCC) $(NVFLAGS) $(LDFLAGS) -o $@ $?
