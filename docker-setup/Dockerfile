# FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

# Set up locales
RUN apt-get clean && apt-get update && \
    apt-get install -y locales && \
    locale-gen en_US.UTF-8

ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US:en
ENV LC_ALL=en_US.UTF-8

# Set timezone, update system and install dependencies
RUN ln -fs /usr/share/zoneinfo/UTC /etc/localtime && \
    apt-get update && apt-get upgrade -y && \
    apt-get install -y \
      sudo gosu ack wget curl rsync unzip vim git cmake gcc-12 g++-12 python3-pip \
      ninja-build software-properties-common libeigen3-dev libncurses5-dev \
      python-is-python3 libgtest-dev python3-cffi tzdata sqlite3 gdb tmux \
      graphviz graphviz-dev openssh-server && \
    dpkg-reconfigure --frontend noninteractive tzdata && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Add PPA for the latest Boost and install Boost packages
RUN add-apt-repository -y ppa:mhier/libboost-latest && \
    apt-get update && \
    apt-get install -y \
      libboost-json1.81-dev libboost-program-options1.81-dev \
      libboost-filesystem1.81-dev libboost-log1.81-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Update alternatives for gcc/g++
RUN for tool in gcc g++; do \
      update-alternatives --install /usr/bin/$tool $tool /usr/bin/${tool}-12 60 || true; \
    done

# Python dependencies
RUN pip3 install --ignore-installed ipython natsort tqdm termcolor cffi numpy matplotlib bokeh scipy flask plotly dash packaging pygraphviz

ENV PYTHONPATH=/workspace/repo/py

RUN groupadd -g 1000 devgroup && \
    useradd -u 1000 -g devgroup devuser && \
    echo "devuser ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

RUN mkdir -p /home/devuser && chown devuser:devgroup /home/devuser
RUN mkdir -p /workspace && chown devuser:devgroup /workspace

# Copy entrypoint script
COPY entrypoint.sh /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]

ARG VERSION="2.1.0"
LABEL version=$VERSION
ENV DOCKER_IMAGE_VERSION=$VERSION

WORKDIR /workspace/repo

CMD ["/start.sh"]
