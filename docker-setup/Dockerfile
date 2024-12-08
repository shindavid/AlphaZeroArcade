FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

LABEL version="1.0.0"

COPY install_deps.sh .
RUN ./install_deps.sh

COPY entrypoint.sh /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]

WORKDIR /workspace/repo

CMD ["bash"]
