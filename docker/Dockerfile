FROM docker.io/nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

COPY install_deps.sh .
RUN ./install_deps.sh

ENTRYPOINT ["bash", "-c"]
