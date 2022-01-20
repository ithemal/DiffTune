#!/usr/bin/env bash

NVIDIA_ARGS=()
COMMAND_ARGS=(bash -l)

if command -v nvidia-smi >/dev/null 2>/dev/null; then
	NVIDIA_ARGS=(--gpus all --runtime=nvidia -e NVIDIA_DRIVER_CAPABILITIES=compute,utility -e NVIDIA_VISIBLE_DEVICES=all)
fi

if [[ "$#" -gt 0 ]]; then
	COMMAND_ARGS=( "${@}" )
fi

if [ -t 0 ]; then
    tflag="t"
else
    tflag=""
fi

sudo docker run -i${tflag} -v "$(realpath "$(dirname "${0}")")":/home/difftune/difftune --rm ${NVIDIA_ARGS[@]} difftune:latest "${COMMAND_ARGS[@]}"
