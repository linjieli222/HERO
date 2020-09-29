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

# captioning eval tool (nlg-eval)
RUN apt-get install -y openjdk-8-jdk && apt-get clean &&\
    pip install git+https://github.com/Maluuba/nlg-eval.git@95af2dcce66feb0d1e2b01e1213eb90b52c9c330 &&\
    nlg-eval --setup

# original TVC MMT
RUN python -c "import nltk; nltk.download('punkt')" &&\
    pip install easydict

# add new command here

WORKDIR /src
