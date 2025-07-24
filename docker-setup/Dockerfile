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
      tmux plocate graphviz graphviz-dev openssh-server \
      gcc-13 g++-13 clang clang-format clang-tidy libgtest-dev sqlite3 gdb \
      libboost-program-options-dev libboost-filesystem-dev libboost-json-dev && \
    rm -rf /var/lib/apt/lists/*

# 3) Build spdlog inline
RUN apt-get update && apt-get install -y --no-install-recommends git cmake build-essential && \
    git clone --depth 1 --branch v1.15.2 https://github.com/gabime/spdlog.git /tmp/spdlog && \
    mkdir /tmp/spdlog/build && cd /tmp/spdlog/build && \
      cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_POSITION_INDEPENDENT_CODE=ON .. && \
      cmake --build . --target install && \
    rm -rf /tmp/spdlog

# 4) Python packages
RUN pip3 install --no-cache-dir \
    ipython natsort tqdm termcolor cffi numpy matplotlib bokeh \
    scipy flask plotly dash packaging pygraphviz \
    google-cloud-compute google-api-python-client \
    torch torchvision torchaudio notebook sortedcontainers \
    runpod onnx

# 5) Node.js via NodeSource (includes npm & npx)
RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && \
    apt-get install -y nodejs

# 6) Miscellaneous
ENV PYTHONPATH=/workspace/repo/py

COPY devuser-setup.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/devuser-setup.sh

COPY entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

ARG VERSION="11.0.1"
LABEL version=$VERSION
ENV DOCKER_IMAGE_VERSION=$VERSION

CMD ["/bin/bash"]
