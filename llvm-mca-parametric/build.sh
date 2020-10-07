#!/usr/bin/env bash

cd "$(dirname ${0})"
mkdir -p build
cd build
cmake -G Ninja ../llvm
cmake --build . --target llvm-mca
cmake --build . --target llvm-get-tables
cmake --build . --target llvm-mc
