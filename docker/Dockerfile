FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    less \
    vim \
    tree \
    htop \
    iotop \
    unzip \
    make \
    build-essential \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    wget \
    curl \
    llvm \
    libncurses5-dev \
    xz-utils \
    tk-dev \
    libxml2-dev \
    libxmlsec1-dev \
    libffi-dev \
    liblzma-dev && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

ENV PYENV_ROOT /opt/pyenv
ENV PATH ${PYENV_ROOT}/shims:${PYENV_ROOT}/bin:${PATH}
RUN curl https://pyenv.run | bash

ENV PYTHON_VERSION 3.7.2
RUN pyenv install ${PYTHON_VERSION} && \
    pyenv global ${PYTHON_VERSION}

RUN pip install -U pip setuptools
RUN pip install cython numpy
RUN git clone https://github.com/hysts/pytorch_yolov3.git && \
    cd pytorch_yolov3 && \
    python setup.py install

WORKDIR /pytorch_yolov3
ENTRYPOINT ["/bin/bash"]
