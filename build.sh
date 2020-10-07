#!/usr/bin/env bash
cd "$(dirname "$0")"

sudo docker build --build-arg HOST_UID=$(id -u) . -t difftune:latest

./run_docker.sh difftune/llvm-mca-parametric/build.sh
./run_docker.sh bash -lc 'cd difftune/exegesis-parametric; bazel build llvm_sim/x86:faucon --override_repository=llvm_git=/home/difftune/difftune/llvm-mca-parametric/llvm'
