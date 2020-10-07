FROM debian:stretch

RUN apt update && DEBIAN_FRONTEND=noninteractive apt install -yq sudo curl wget build-essential cmake ninja-build python3-dev
RUN apt update && DEBIAN_FRONTEND=noninteractive apt install -yq libcurl3-gnutls apt-transport-https ca-certificates
RUN curl -fsSL https://bazel.build/bazel-release.pub.gpg | gpg --dearmor > bazel.gpg && \
        mv bazel.gpg /etc/apt/trusted.gpg.d/ && \
        echo "deb [arch=amd64] https://storage.googleapis.com/bazel-apt stable jdk1.8" | tee /etc/apt/sources.list.d/bazel.list
RUN apt update && DEBIAN_FRONTEND=noninteractive apt install -yq bazel


ARG HOST_UID=1000
ENV HOST_UID $HOST_UID

RUN groupadd -g 1000 difftune && useradd -m -s /bin/bash -r -u $HOST_UID -g difftune difftune
USER difftune
WORKDIR /home/difftune

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
        bash Miniconda3-latest-Linux-x86_64.sh -bf && \
        rm Miniconda3-latest-Linux-x86_64.sh

RUN miniconda3/bin/conda create -yn difftune python=3.7 numba=0.49.1 numpy=1.18.1 pandas=1.0.3 tqdm=4.46.0 cloudpickle=1.4.1 ipython=7.13.0 pytorch=1.2.0 cudatoolkit=10.0.130 scipy=1.5.2
RUN miniconda3/bin/conda init
RUN echo conda activate difftune >> ~/.bashrc
