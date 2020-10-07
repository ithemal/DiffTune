#!/usr/bin/env bash
pushd `dirname $0`

bazel run llvm_sim/x86:faucon \
      --override_repository=llvm_git=/home/difftune/difftune/llvm-mca-parametric/llvm \
      -- -input_type=att_asm -max_iters=100 -

popd
