# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
# Modified from UNITER
# (https://github.com/ChenRocks/UNITER)

DOWNLOAD=$1

for FOLDER in 'video_db' 'txt_db' 'pretrained' 'finetune'; do
    if [ ! -d $DOWNLOAD/$FOLDER ] ; then
        mkdir -p $DOWNLOAD/$FOLDER
    fi
done

BLOB='https://convaisharables.blob.core.windows.net/hero'

# video dbs
if [ ! -d $DOWNLOAD/video_db/tv/ ] ; then
    wget $BLOB/video_db/tv.tar -P $DOWNLOAD/video_db/
    tar -xvf $DOWNLOAD/video_db/tv.tar -C $DOWNLOAD/video_db --strip-components 1
    rm $DOWNLOAD/video_db/tv.tar
fi

# text dbs
for SPLIT in 'train' 'val' ; do
    wget $BLOB/txt_db/tvc_$SPLIT.db.tar -P $DOWNLOAD/txt_db/
    tar -xvf $DOWNLOAD/txt_db/tvc_$SPLIT.db.tar -C $DOWNLOAD/txt_db
    rm $DOWNLOAD/txt_db/tvc_$SPLIT.db.tar
done
if [ ! -d $DOWNLOAD/txt_db/tv_subtitles.db/ ] ; then
    wget $BLOB/txt_db/tv_subtitles.db.tar -P $DOWNLOAD/txt_db/
    tar -xvf $DOWNLOAD/txt_db/tv_subtitles.db.tar -C $DOWNLOAD/txt_db
    rm $DOWNLOAD/txt_db/tv_subtitles.db.tar
fi

# pretrained
if [ ! -f $DOWNLOAD/pretrained/hero-tv-ht100.pt ] ; then
    wget $BLOB/pretrained/hero-tv-ht100.pt -P $DOWNLOAD/pretrained/
fi

# raw data
RAW_URL=https://raw.githubusercontent.com/jayleicn/TVCaption/66666ec08657d8963b165b18eafabd6427d44261/data/
for SPLIT in 'train' 'val' 'test_public'; do
    wget $RAW_URL/tvc_${SPLIT}_release.jsonl -P $DOWNLOAD/txt_db
done
wget $RAW_URL/tvqa_preprocessed_subtitles.jsonl -P $DOWNLOAD/txt_db
