# FROM ubuntu:18.04
FROM nvidia/cuda:11.4.3-cudnn8-devel-ubuntu18.04

# For bash-specific commands
SHELL ["/bin/bash", "-c"]

# Fix Nvidia repo key rotation issue
# https://forums.developer.nvidia.com/t/notice-cuda-linux-repository-key-rotation/212771
# https://forums.developer.nvidia.com/t/18-04-cuda-docker-image-is-broken/212892/10
# https://code.visualstudio.com/remote/advancedcontainers/reduce-docker-warnings#:~:text=Warning%3A%20apt%2Dkey%20output%20should,not%20running%20from%20a%20terminal.
RUN if [ "${BUILD_CUDA_MODULE}" = "ON" ]; then \
    export APT_KEY_DONT_WARN_ON_DANGEROUS_USAGE=DontWarn; \
    apt-key del 7fa2af80; \
    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/3bf863cc.pub; \
    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub; \
    fi

ENV PROJECT_NAME=pose_classifier

# Prevent interactive inputs when installing packages
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Moscow
ENV SUDO=command

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1
# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# Dependencies: basic
RUN apt-get update && apt-get install -y \
    git  \
    wget \
    curl \
    build-essential \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /root/project

# COPY requirements.txt .
# RUN python3 -m pip install --no-cache-dir -r requirements.txt

# COPY . ${PROJECT_NAME}

WORKDIR /root/project/${PROJECT_NAME}
