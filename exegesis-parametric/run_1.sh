#!/usr/bin/env bash
pushd `dirname $0`

bazel --output_user_root=/nobackup/users/renda/bazel_cache run llvm_sim/x86:faucon --override_repository=llvm_git=/home/renda/diffsim-code/llvm/llvm -- -input_type=att_asm -max_iters=1 -

popd
