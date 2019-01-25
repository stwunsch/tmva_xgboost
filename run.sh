#!/bin/bash

python train.py
g++ main.cxx -o main -std=c++11
./main
