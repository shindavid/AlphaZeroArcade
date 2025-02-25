FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04

# apt update/upgrade, set up locales
RUN apt-get clean && apt-get update && apt-get upgrade -y  && \
    apt-get install -y locales && \
    locale-gen en_US.UTF-8

# Install libtorch
# We put this at the top of the Dockerfile to avoid re-downloading libtorch when rebuilding the
# image when adding new dependencies.
ARG LIBTORCH_VERSION="libtorch-cxx11-abi-shared-with-deps-2.6.0+cu126"
ARG LIBTORCH_BASE_DIR="/usr/local"
ARG LIBTORCH_DIR="${LIBTORCH_BASE_DIR}/${LIBTORCH_VERSION}"
RUN LIBTORCH_VERSION_ESCAPED="$(echo "$LIBTORCH_VERSION" | sed 's/+/%2B/g')" && \
    LIBTORCH_URL="https://download.pytorch.org/libtorch/cu126/${LIBTORCH_VERSION_ESCAPED}.zip" && \
    apt-get install -y wget unzip && \
    wget -O /tmp/libtorch.zip "${LIBTORCH_URL}" && \
    unzip -q /tmp/libtorch.zip -d "${LIBTORCH_BASE_DIR}" && \
    mv "${LIBTORCH_BASE_DIR}/libtorch" "${LIBTORCH_DIR}" && \
    ln -s "${LIBTORCH_DIR}" "${LIBTORCH_BASE_DIR}/libtorch" && \
    rm /tmp/libtorch.zip

ENV LIBTORCH_CPP_DIR="${LIBTORCH_BASE_DIR}/libtorch"

ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US:en
ENV LC_ALL=en_US.UTF-8

# Set timezone, update system and install dependencies
RUN ln -fs /usr/share/zoneinfo/UTC /etc/localtime && \
    apt-get install -y \
      sudo gosu ack curl rsync vim git cmake gcc-12 g++-12 python3-pip \
      ninja-build software-properties-common libeigen3-dev libncurses5-dev \
      python-is-python3 libgtest-dev python3-cffi tzdata sqlite3 gdb tmux \
      graphviz graphviz-dev openssh-server && \
    dpkg-reconfigure --frontend noninteractive tzdata

# Add PPA for the latest Boost and install Boost packages
RUN add-apt-repository -y ppa:mhier/libboost-latest && \
    apt-get update && \
    apt-get install -y \
      libboost-json1.81-dev libboost-program-options1.81-dev \
      libboost-filesystem1.81-dev libboost-log1.81-dev

# Update alternatives for gcc/g++
RUN for tool in gcc g++; do \
      update-alternatives --install /usr/bin/$tool $tool /usr/bin/${tool}-12 60 || true; \
    done

# Python dependencies
#
# Technically, we don't need the google stuff within the Docker container, as it's only used
# externally. But it's common to attach vscode to the container, and having the google stuff
# installed allows for better code completion when working with the Docker-external code.
RUN pip3 install --ignore-installed ipython natsort tqdm termcolor cffi numpy matplotlib \
    bokeh scipy flask plotly dash packaging pygraphviz google-cloud-compute \
    google-api-python-client torch torchvision torchaudio

################################################################################################
# FOR FASTER DOCKER IMAGE REBUILDS: put new apt-get/pip3 installs below this line. This avoids #
# re-installing all the above when rebuilding the image. We can periodically move these back   #
# up for cleanliness.                                                                          #
################################################################################################

RUN pip3 install --ignore-installed notebook

# Clean up
RUN apt-get clean && rm -rf /var/lib/apt/lists/*

ENV PYTHONPATH=/workspace/repo/py

COPY devuser-setup.sh /usr/local/bin/devuser-setup.sh
RUN chmod +x /usr/local/bin/devuser-setup.sh

COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

ARG VERSION="3.4.1"
LABEL version=$VERSION
ENV DOCKER_IMAGE_VERSION=$VERSION

CMD ["/bin/bash"]
