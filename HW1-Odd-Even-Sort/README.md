# Odd-Even Sort

## Introduction

Implement the parallel odd-even sort algorithm using MPI.

## Odd-Even Sort Algorithm

Odd-even sort is a comparison sort consists of two main phases: *even-phase* and *odd-phase*.

In the even-phase, all even/odd indexed pairs of adjacent elements are compared. 

In the even-phase, all odd/even indexed pairs of adjacent elements are compared. 

The algorithm alternates between two phases until array is sorted.

## Compile

`mpicxx -O3 -lm hw1.cc -o hw1`