# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
# Modified from UNITER
# (https://github.com/ChenRocks/UNITER)

TXT_DB=$1
VID_DIR=$2
OUTPUT=$3
PRETRAIN_DIR=$4

if [ -z $CUDA_VISIBLE_DEVICES ]; then
    CUDA_VISIBLE_DEVICES='all'
fi

if [ "$5" = "--prepro" ]; then
    RO=""
else
    RO=",readonly"
fi

docker run --gpus '"'device=$CUDA_VISIBLE_DEVICES'"' --ipc=host --network=host --rm -it \
    --mount src=$(pwd),dst=/src,type=bind \
    --mount src=$OUTPUT,dst=/storage,type=bind \
    --mount src=$PRETRAIN_DIR,dst=/pretrain,type=bind,readonly \
    --mount src=$TXT_DB,dst=/txt,type=bind$RO \
    --mount src=$VID_DIR,dst=/video,type=bind,readonly \
    -e NVIDIA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
    -w /src linjieli222/hero
