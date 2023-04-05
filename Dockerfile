# FROM ubuntu:18.04
FROM nvidia/cuda:11.4.3-cudnn8-devel-ubuntu18.04
FROM python:3.10.4

# For bash-specific commands
SHELL ["/bin/bash", "-c"]

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
    apt-utils \
    git \
    wget \
    curl \
    ca-certificates \
    sudo \
    tmux \
    build-essential \
    pkg-config \
    libx11-6 \
    screen \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    tqdm \
    wandb \
    plotly \
    mrob \
    filterpy \
    opencv-python \
    open3d \
    imageio \
    mediapipe \
    torch \
    torchvision \
    torchaudio

# WORKDIR /root/project
# COPY requirements.txt .
# RUN python3 -m pip install --no-cache-dir -r requirements.txt

# COPY . ${PROJECT_NAME}

WORKDIR /root/project/${PROJECT_NAME}

CMD ["bash"]
