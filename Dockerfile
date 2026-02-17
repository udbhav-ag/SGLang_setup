FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=en_US.UTF-8

RUN apt update && apt install -y \
    locales \
    iputils-ping \
    wget \
    curl \
    btop \
    nano \
    net-tools \
    htop \
    python3 \
    python3-pip \
    git \
    build-essential \
    iperf3 \
    tmux && \
    locale-gen en_US.UTF-8 \
    libnuma-dev \
    libnuma1 \
    build-essential g++ pkg-config libstdc++-13-dev nvidia-cuda-toolkit


WORKDIR /Udbhav