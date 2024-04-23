# Lab3 Edge Detection (CUDA)

## Goal

Use two 5x5 filter matrix convolved with the original image to calculate approximatations of the derivatives for both horizontal and vertical changes.

## Compile

`nvcc  -O3 -std=c++11 -Xptxas=-v -arch=sm_61 -lpng -lz -o lab3 lab3.cu`

