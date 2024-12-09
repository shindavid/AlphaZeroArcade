FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

LABEL version="1.0.0"

RUN apt-get clean && apt-get update
RUN apt-get install locales
RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

COPY install_deps.sh .
RUN ./install_deps.sh

COPY entrypoint.sh /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]

WORKDIR /workspace/repo

CMD ["bash"]
