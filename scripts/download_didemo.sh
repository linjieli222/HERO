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
if [ ! -d $DOWNLOAD/video_db/didemo/ ] ; then
    wget $BLOB/video_db/didemo.tar -P $DOWNLOAD/video_db/
    tar -xvf $DOWNLOAD/video_db/didemo.tar -C $DOWNLOAD/video_db --strip-components 1
    rm $DOWNLOAD/video_db/didemo.tar
fi

# text dbs
for SPLIT in 'train' 'val' 'test'; do
    wget $BLOB/txt_db/didemo_$SPLIT.db.tar -P $DOWNLOAD/txt_db/
    tar -xvf $DOWNLOAD/txt_db/didemo_$SPLIT.db.tar -C $DOWNLOAD/txt_db
    rm $DOWNLOAD/txt_db/didemo_$SPLIT.db.tar
done
if [ ! -d $DOWNLOAD/txt_db/didemo_subtitles.db/ ] ; then
    wget $BLOB/txt_db/didemo_subtitles.db.tar -P $DOWNLOAD/txt_db/
    tar -xvf $DOWNLOAD/txt_db/didemo_subtitles.db.tar -C $DOWNLOAD/txt_db
    rm $DOWNLOAD/txt_db/didemo_subtitles.db.tar
fi

# pretrainedsd
if [ ! -f $DOWNLOAD/pretrained/hero-tv-ht100.pt ] ; then
    wget $BLOB/pretrained/hero-tv-ht100.pt -P $DOWNLOAD/pretrained/
fi
