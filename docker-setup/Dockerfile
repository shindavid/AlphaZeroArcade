FROM nvcr.io/nvidia/tensorrt:25.06-py3
WORKDIR /workspace/repo

# 1) Locale & timezone
RUN apt-get update && \
    apt-get install -y locales tzdata && \
    locale-gen en_US.UTF-8 && \
    ln -fs /usr/share/zoneinfo/UTC /etc/localtime && \
    dpkg-reconfigure --frontend noninteractive tzdata && \
    rm -rf /var/lib/apt/lists/*

ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8

# 2) Core tools, compilers & Boost from Ubuntu 24.04
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      sudo gosu curl rsync vim git cmake ninja-build build-essential \
      python3-pip python3-dev python-is-python3 libeigen3-dev libncurses5-dev \
      tmux plocate graphviz graphviz-dev openssh-server ack \
      gcc-13 g++-13 clang-19 clangd-19 clang-format-19 clang-tidy-19 libgtest-dev sqlite3 gdb \
      libboost-program-options-dev libboost-filesystem-dev libboost-json-dev \
      libboost-system-dev \
      libabsl-dev && \
    rm -rf /var/lib/apt/lists/*

# register the -19 tools as the system default
RUN update-alternatives --install /usr/bin/clang   clang   /usr/bin/clang-19   100 \
 && update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-19 100 \
 && update-alternatives --install /usr/bin/clang-format clang-format /usr/bin/clang-format-19 100 \
 && update-alternatives --install /usr/bin/clang-tidy  clang-tidy  /usr/bin/clang-tidy-19  100 \
 && update-alternatives --install /usr/bin/clangd  clangd  /usr/bin/clangd-19  100

# 3) Build spdlog inline
RUN apt-get update && apt-get install -y --no-install-recommends git cmake build-essential && \
    git clone --depth 1 --branch v1.15.2 https://github.com/gabime/spdlog.git /tmp/spdlog && \
    mkdir /tmp/spdlog/build && cd /tmp/spdlog/build && \
      cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_POSITION_INDEPENDENT_CODE=ON .. && \
      cmake --build . --target install && \
    rm -rf /tmp/spdlog

# 4) Python packages
RUN pip3 install --pre --upgrade --force-reinstall --no-cache-dir \
    --index-url https://download.pytorch.org/whl/nightly/cu129 \
    --constraint /dev/null torch torchvision torchaudio

RUN pip3 install --no-cache-dir \
    ipython natsort tqdm termcolor cffi numpy matplotlib bokeh \
    scipy flask plotly dash packaging pygraphviz \
    google-cloud-compute google-api-python-client \
    notebook sortedcontainers runpod onnx onnxscript onnxruntime

# 5) Node.js via NodeSource (includes npm & npx)
RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && \
    apt-get install -y nodejs

# 6) Temporary home for extra dependencies. Putting them here allows us to quickly rebuild the image
#    without having to rebuild the entire image. Items here should later be moved to the more
#    appropriate location above.
RUN apt-get update && \
    apt-get install -y lsof libonnx-dev libprotobuf-dev protobuf-compiler docker.io && \
    rm -rf /var/lib/apt/lists/*

# 7) Miscellaneous
ENV PYTHONPATH=/workspace/repo/py

COPY devuser-setup.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/devuser-setup.sh

COPY entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

ARG VERSION="16.0.1"
LABEL version=$VERSION
ENV DOCKER_IMAGE_VERSION=$VERSION

CMD ["/bin/bash"]
