# All-Pairs Shortest Path

## Introduction

Solve the all-pairs shortest path problem by Blocked Floyd-Warshall algorithm using OpenMP and CUDA.

## GPU Information

<img src="https://github.com/JimLi93/Parallel-programming/blob/main/HW3-All-Pairs-Shortest-Path/img/GPU_INFO.png" alt="GPU information" width="400">

## Code Specification

`hw3-1.cc` : CPU version. Use OpenMP to parallelize the computation.

`hw3-2.cu` : Use 1 GPU and CUDA programming to accelerate the computation.

`hw3-2.cu` : Use 2 GPU and CUDA programming.

## GPU Optimization

* Coalesced Memory Access
* Share Memory
* Blocking Factor Tuning 
* uUnroll


## Compile

`g++ -O3 -lm -fopenmp -o hw3-1 hw3-1.cc`

`nvcc -std=c++11 -O3 -Xptxas="-v" -arch=sm_61 -Xcompiler -fopenmp -lm -o hw3-2 hw3-2.cc`

`nvcc -std=c++11 -O3 -Xptxas="-v" -arch=sm_61 -Xcompiler -fopenmp -lm -o hw3-3 hw3-3.cc`