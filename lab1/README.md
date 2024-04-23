# Lab1 Pixels in Circle

## Goal

Calculate the pixels when drawing circle of radius r on a 2D monitor. 

<img src="https://github.com/JimLi93/Parallel-programming/raw/main/lab1/img/pixels_equation.png" alt="Pixels Equation" width="300">

## Compile

`mpicxx -O3 lab1.cc -lm -o lab1`

## Input & Output 

### Input
`r` : The radius of the circle (integer)

`k` : Integer

### Output
`pixels % k`
