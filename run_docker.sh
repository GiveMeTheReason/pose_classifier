#!/bin/bash

docker build \
    -f "Dockerfile" \
    -t pose_classifier:latest "."

docker run \
    --rm \
    -it \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v ~/.Xauthority:/tmp/.Xauthority \
    -v ~/personal/pose_classifier:/root/project/pose_classifier \
    -e XAUTHORITY=/tmp/.Xauthority \
    --env="DISPLAY" \
    pose_classifier:latest
