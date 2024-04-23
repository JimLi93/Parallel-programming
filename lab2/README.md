# Lab2 Pixels in Circle (Pthread & OpenMP)

## Goal

Calculate the pixels when drawing circle of radius r on a 2D monitor using Pthread and OpenMP. 

<img src="https://github.com/JimLi93/Parallel-programming/raw/main/lab1/img/pixels_equation.png" alt="Pixels Equation" width="300">

## Compile

`g++ -lm -O3 -pthread lab2_pthread.cc -o lab2_pthread`

`g++ -lm -O3 -fopenmp lab2_omp.cc -o lab2_omp`

`mpicxx -lm -O3 -fopenmp lab2_hybrid.cc -o lab2_hybrid`

## Input & Output 

### Input
`r` : The radius of the circle (integer)

`k` : Integer

### Output
`pixels % k`
