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
wget $BLOB/video_db/tv.tar -P $DOWNLOAD/video_db/
tar -xvf $DOWNLOAD/video_db/tv.tar -C $DOWNLOAD/video_db

# text dbs
wget $BLOB/txt_db/tv_subtitles.tar -P $DOWNLOAD/txt_db/
tar -xvf $DOWNLOAD/txt_db/tv_subtitles.tar -C $DOWNLOAD/txt_db

# pretrain splits
wget $BLOB/txt_db/pretrain_splits.tar -P $DOWNLOAD/txt_db/
tar -xvf $DOWNLOAD/txt_db/pretrain_splits.tar -C $DOWNLOAD/txt_db

# converted RoBERTa
wget $BLOB/pretrained/pretrain-tv-init.bin -P $DOWNLOAD/pretrained/
