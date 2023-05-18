# Download base image ubuntu 22.04
FROM ubuntu:latest

RUN apt update
RUN apt install g++-12 -y
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 60 --slave /usr/bin/g++ g++ /usr/bin/g++-12