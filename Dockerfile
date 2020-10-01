FROM nvcr.io/nvidia/pytorch:19.10-py3

# basic python packages
RUN pip install transformers==2.0.0 \
                tensorboardX==1.7 ipdb==0.12 lz4==2.1.9 lmdb==0.97

####### horovod for multi-GPU (distributed) training #######
# horovod
RUN HOROVOD_GPU_ALLREDUCE=NCCL HOROVOD_NCCL_LINK=SHARED HOROVOD_WITH_PYTORCH=1 \
    pip install --no-cache-dir horovod==0.18.2 &&\
    ldconfig

# ssh
RUN apt-get update &&\
    apt-get install -y --no-install-recommends openssh-client openssh-server &&\
    mkdir -p /var/run/sshd

# Allow OpenSSH to talk to containers without asking for confirmation
RUN cat /etc/ssh/ssh_config | grep -v StrictHostKeyChecking > /etc/ssh/ssh_config.new && \
    echo "    StrictHostKeyChecking no" >> /etc/ssh/ssh_config.new && \
    mv /etc/ssh/ssh_config.new /etc/ssh/ssh_config

# captioning

# captioning eval tool (java for PTBtokenizer and METEOR)
RUN apt-get install -y --no-install-recommends openjdk-8-jdk && apt-get clean

# binaries for cococap eval
ARG PYCOCOEVALCAP=https://github.com/tylin/coco-caption/raw/master/pycocoevalcap
RUN mkdir /workspace/cococap_bin/ && \
    wget $PYCOCOEVALCAP/meteor/meteor-1.5.jar -P /workspace/cococap_bin/ && \
    wget $PYCOCOEVALCAP/meteor/data/paraphrase-en.gz -P /workspace/cococap_bin/ && \
    wget $PYCOCOEVALCAP/tokenizer/stanford-corenlp-3.4.1.jar -P /workspace/cococap_bin/

# add new command here

WORKDIR /src
