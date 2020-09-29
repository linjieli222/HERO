# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
# Modified from UNITER
# (https://github.com/ChenRocks/UNITER)

DOWNLOAD=$1

if [ ! -d $DOWNLOAD/pretrained ] ; then
    mkdir -p $DOWNLOAD/pretrained
fi

BLOB='https://convaisharables.blob.core.windows.net/hero'

# This will overwrite models
wget $BLOB/pretrained/hero-tv-ht100.pt -O $DOWNLOAD/pretrained/hero-tv-ht100.pt
