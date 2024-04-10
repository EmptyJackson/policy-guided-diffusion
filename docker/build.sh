#!/bin/bash

echo 'Building Dockerfile with image name pgd'
docker build \
    --build-arg UID=$(id -u ${USER}) \
    --build-arg GID=1234 \
    --build-arg REQS="$(cat requirements.txt | tr '\n' ' ')" \
    -t pgd \
    -f Dockerfile \
    .
