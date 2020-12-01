# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
# Modified from UNITER
# https://github.com/ChenRocks/UNITER

OUT_DIR=$1
ANN_DIR=$2

set -e

# annotations
URL='https://raw.githubusercontent.com/jayleicn/TVRetrieval/master/data'
BLOB='https://convaisharables.blob.core.windows.net/hero'

if [ ! -d $OUT_DIR ]; then
    mkdir -p $OUT_DIR
fi
if [ ! -d $ANN_DIR ]; then
    mkdir -p $ANN_DIR
fi

for SPLIT in 'train' 'val' 'test_public'; do
    if [ ! -f $ANN_DIR/tvr_$SPLIT.jsonl ]; then
        echo "downloading ${SPLIT} annotations..."
        wget $URL/tvr_${SPLIT}_release.jsonl -O $ANN_DIR/tvr_$SPLIT.jsonl
    fi

    echo "preprocessing tvr ${SPLIT} annotations..."
    docker run --ipc=host --rm -it \
        --mount src=$(pwd),dst=/src,type=bind \
        --mount src=$OUT_DIR,dst=/txt_db,type=bind \
        --mount src=$ANN_DIR,dst=/ann,type=bind,readonly \
        -w /src linjieli222/hero \
        python script/prepro_query.py --annotation /ann/tvr_$SPLIT.jsonl \
                         --output /txt_db/tvr_${SPLIT}.db \
                         --task tvr
done

wget $URL/tvqa_preprocessed_subtitles.jsonl -O $ANN_DIR/tv_subtitles.jsonl
wget $BLOB/tv_vid2nframe.json -O $ANN_DIR/tv_vid2nframe.json
wget $URL/tvr_video2dur_idx.json -O $ANN_DIR/vid2dur_idx.json
echo "preprocessing tv subtitles..."
docker run --ipc=host --rm -it \
    --mount src=$(pwd),dst=/src,type=bind \
    --mount src=$OUT_DIR,dst=/txt_db,type=bind \
    --mount src=$ANN_DIR,dst=/ann,type=bind,readonly \
    -w /src linjieli222/hero \
    /bin/bash -c "python script/prepro_sub.py --annotation /ann/tv_subtitles.jsonl --output /txt_db/tv_subtitles.db --vid2nframe /ann/tv_vid2nframe.json --frame_length 1.5; cp /ann/vid2dur_idx.json /txt_db/tv_subtitles.db/"
echo "done"