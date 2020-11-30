#!/usr/bin/env bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

DATA=$1  # txt_db

for SPLIT in 'val' 'train'; do
    CMD="python scripts/prepro_tvc.py \
         --annotation /txt/tvc_${SPLIT}_release.jsonl \
         --subtitles /txt/tvqa_preprocessed_subtitles.jsonl \
         --output /txt/tvc_${SPLIT}_new.db"

    docker run --ipc=host --rm \
        --mount src=$(pwd),dst=/src,type=bind \
        --mount src=$DATA,dst=/txt,type=bind \
        -w /src linjieli222/hero \
        bash -c "$CMD"
done
