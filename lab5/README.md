# Lab5 DNN Model Calculation (OpenACC)

## Goal

Apply OpenACC to 3 types of calculations in DNN model.
* Single Layer
* Sigmoid
* Argmax

## Compile

`nvc++ -O0 -std=c++14 -fast -acc -gpu=cc60 -Minfo=accel -I/home/pp23/share/lab5/testcases/weights/mnist/include -o lab5 lab5.cpp`

