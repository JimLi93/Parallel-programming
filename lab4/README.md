# Lab4 Edge Detection with Optimization (CUDA)

## Goal

Use two 5x5 filter matrix convolved with the original image to calculate approximatations of the derivatives for both horizontal and vertical changes.

Apply optimizations to improve the preformance.

## Optimization

* Coalesced Memory Access
* Lower Precision
* Shared Memory

## Compile

`nvcc  -O3 -std=c++11 -Xptxas=-v -arch=sm_61 -lpng -lz -o lab4 lab4.cu`

