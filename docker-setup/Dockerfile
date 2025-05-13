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

###################################################################################################
# PPA Installations: These are packages that are not available with our current base image.       #
# In the future, if we switch to a newer base image, we can remove these specialized PPA installs #
# and simply add the packages to the above apt-get install command.                               #
###################################################################################################

# gcc-13/g++-13
RUN apt-get install -y software-properties-common && \
    add-apt-repository ppa:ubuntu-toolchain-r/test && \
    apt-get update && \
    apt-get install -y gcc-13 g++-13

# Boost
#
# TODO: remove libbost-log from below, we don't need it anymore. Refraining from doing this for now
# so as to not inconvenience devs with a slow pull of a new image.
RUN add-apt-repository -y ppa:mhier/libboost-latest && \
    apt-get update && \
    apt-get install -y \
      libboost-json1.81-dev libboost-program-options1.81-dev \
      libboost-filesystem1.81-dev libboost-log1.81-dev

# Update alternatives for gcc/g++
RUN for tool in gcc g++; do \
      update-alternatives --install /usr/bin/$tool $tool /usr/bin/${tool}-13 60 || true; \
    done

#####################
# Misc dependencies #
#####################

# clangd, needed for vscode plugin
RUN wget https://apt.llvm.org/llvm.sh && \
    chmod +x llvm.sh && \
    ./llvm.sh 19 && \
    apt-get install -y clangd-19 && \
    update-alternatives --install /usr/bin/clangd clangd /usr/bin/clangd-19 100 && \
    rm llvm.sh

#######################
# Python dependencies #
#######################

# Technically, we don't need the google/runpod stuff within the Docker container, as it's only used
# externally. But it's common to attach vscode to the container, and having that stuff
# installed allows for better code completion when working with the Docker-external code.
RUN pip3 install --ignore-installed ipython natsort tqdm termcolor cffi numpy matplotlib \
    bokeh scipy flask plotly dash packaging pygraphviz google-cloud-compute \
    google-api-python-client torch torchvision torchaudio \
    notebook sortedcontainers runpod

#####################
# Misc dependencies #
#####################

# Install spdlog (compiled version)
RUN git clone --branch v1.15.2 --depth 1 https://github.com/gabime/spdlog.git && \
    cd spdlog && \
    mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_POSITION_INDEPENDENT_CODE=ON && \
    cmake --build . --target install && \
    cd ../.. && rm -rf spdlog

################################################################################################
# FOR FASTER DOCKER IMAGE REBUILDS: put new dependencies below this line. This avoids          #
# re-installing all the above when rebuilding the image. We can periodically move these back   #
# up for cleanliness.                                                                          #
################################################################################################

# optional, for onnx.checker or shape inference
RUN pip3 install onnx

ARG TRT_VER=10.10.0
ARG CUDA_VER=12.9
ARG DOWNLOADS_URL=https://developer.nvidia.com/downloads
ARG URL_HEAD=${DOWNLOADS_URL}/compute/machine-learning/tensorrt/${TRT_VER}/local_repo
ARG URL_TAIL=nv-tensorrt-local-repo-ubuntu2204-${TRT_VER}-cuda-${CUDA_VER}_1.0-1_amd64.deb
ARG FULL_URL=${URL_HEAD}/${URL_TAIL}
ARG VAR_DIR=/var/nv-tensorrt-local-repo-ubuntu2204-${TRT_VER}-cuda-${CUDA_VER}
ARG TMP_DEB=/tmp/nv-tensorrt-local-repo.deb

# 1) Remove any stale NVIDIA ML repo entry
RUN rm -f /etc/apt/sources.list.d/nvidia-ml.list

# 2) Download and install the TensorRT “local‐repo” .deb
RUN wget -qO ${TMP_DEB} "${FULL_URL}" && dpkg -i ${TMP_DEB}

# 3) Install every package from that local repo
RUN dpkg -i ${VAR_DIR}/*.deb

# 4) Pull in any missing dependencies
RUN apt-get install -y --no-install-recommends -f

# ARG TRT_VER=10.10.0
# ARG CUDA_VER=12.9
# ARG DOWNLOADS_URL=https://developer.nvidia.com/downloads
# ARG URL_HEAD=${DOWNLOADS_URL}/compute/machine-learning/tensorrt/${TRT_VER}/local_repo
# ARG URL_TAIL=nv-tensorrt-local-repo-ubuntu2204-${TRT_VER}-cuda-${CUDA_VER}_1.0-1_amd64.deb
# ARG FULL_URL=${URL_HEAD}/${URL_TAIL}
# ARG VAR_DIR=/var/nv-tensorrt-local-repo-ubuntu2204-${TRT_VER}-cuda-${CUDA_VER}
# ARG TMP_DEB=/tmp/nv-tensorrt-local-repo.deb

# # 1) Remove any stale remote ML repo
# RUN rm -f /etc/apt/sources.list.d/nvidia-ml.list

# # 2) Download & install the local-repo .deb
# RUN wget -qO ${TMP_DEB} ${FULL_URL} && dpkg -i ${TMP_DEB}

# # 2b) Remove the auto-generated, unsigned source list so apt only sees the trusted one
# RUN rm -f /etc/apt/sources.list.d/nv-tensorrt-local-repo-*

# # 3) Register the local-repo as a trusted apt source
# RUN echo "deb [trusted=yes] file:${VAR_DIR} ./" > /etc/apt/sources.list.d/tensorrt-local.list

# # 3b) Allow unsigned file:// repositories
# RUN printf 'Acquire::AllowInsecureRepositories \"true\";\nAPT::Get::AllowUnauthenticated \"true\";\n' \
#     > /etc/apt/apt.conf.d/99allow-insecure

# # 4) Update & install all TensorRT packages, then fix deps
# RUN apt-get update && dpkg -i ${VAR_DIR}/*.deb && apt-get install -y --no-install-recommends -f

###################
# Clean up + Misc #
###################

RUN apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/*.deb

ENV PYTHONPATH=/workspace/repo/py

COPY devuser-setup.sh /usr/local/bin/devuser-setup.sh
RUN chmod +x /usr/local/bin/devuser-setup.sh

COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

ARG VERSION="4.0.3"
LABEL version=$VERSION
ENV DOCKER_IMAGE_VERSION=$VERSION

CMD ["/bin/bash"]
