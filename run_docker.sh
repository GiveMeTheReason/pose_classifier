#!/bin/bash

echo "Running docker image build..."

docker build \
    -f "Dockerfile" \
    -t pose_classifier:latest "."

echo "Docker image build completed!"
echo "Starting Docker container..."

docker run \
    --rm \
    -it \
    --gpus all \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v ~/.Xauthority:/tmp/.Xauthority \
    -v ~/personal/pose_classifier:/root/project/pose_classifier \
    -v ~/personal/datasets/HuaweiGesturesDataset/undistorted:/root/project/pose_classifier/data/undistorted \
    -e XAUTHORITY=/tmp/.Xauthority \
    --env "DISPLAY" \
    --env-file "./.env" \
    pose_classifier:latest
