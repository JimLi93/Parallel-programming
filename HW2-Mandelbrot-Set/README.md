# Mandelbrot Set

## Introduction

Parallelize the sequential Mandelbrot Set using Pthread, MPI, and OpenMP.

## Code Specification

`hw2a.cc` : Single node shared memory programming using Pthread.

`hw2b.cc` : Multi-node hybrid parallelism programming using MPI + OpenMP.

## Compile

`g++ -lm -O3 -pthread hw2a.cc -lpng -o hw2a`

`mpicxx -lm -O3 -fopenmp hw2b.cc -lpng -o hw2b`