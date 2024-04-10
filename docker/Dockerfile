FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04
LABEL org.opencontainers.image.authors="jackson@robots.ox.ac.uk"

ENV CUDA_PATH /usr/local/cuda
ENV CUDA_INCLUDE_PATH /usr/local/cuda/include
ENV CUDA_LIBRARY_PATH /usr/local/cuda/lib64

# Set timezone
ENV TZ=Europe/London DEBIAN_FRONTEND=noninteractive

# Add old libraries (Python 3.8) to Ubuntu 22.04
RUN apt-get update
RUN apt-get install -y libosmesa6-dev \
    libgl1-mesa-glx \
    libglfw3 \
    software-properties-common \
    patchelf \
    libglew-dev
RUN apt-add-repository ppa:deadsnakes/ppa
RUN apt-get update
RUN apt-get install -y \
    git \
    curl \
    python3.9 \
    python3-pip \
    python3.9-venv \
    python3-setuptools \
    python3-wheel \
    python3.9-dev

# Create local user
# https://jtreminio.com/blog/running-docker-containers-as-current-host-user/
ARG UID
ARG GID
RUN if [ ${UID:-0} -ne 0 ] && [ ${GID:-0} -ne 0 ]; then \
    groupadd -g ${GID} duser &&\
    useradd -l -u ${UID} -g duser duser &&\
    install -d -m 0755 -o duser -g duser /home/duser &&\
    chown --changes --silent --no-dereference --recursive ${UID}:${GID} /home/duser \
    ;fi

USER duser
WORKDIR /home/duser

# Install Python packages
ENV PATH="/home/duser/.local/bin:$PATH"
RUN python3.9 -m pip install --upgrade pip
ARG REQS
RUN python3.9 -m pip install --default-timeout=100 $REQS -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Install MuJoCo
RUN git clone https://github.com/rail-berkeley/d4rl.git
WORKDIR /home/duser/d4rl
RUN python3.9 -m pip install -e .
WORKDIR /home/duser
RUN curl -L https://github.com/google-deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz --output mujoco210.tar.gz
RUN mkdir /home/duser/.mujoco
RUN tar -xvf mujoco210.tar.gz --directory /home/duser/.mujoco
RUN curl https://www.roboti.us/file/mjkey.txt --output /home/duser/.mujoco/mjkey.txt
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/duser/.mujoco/mujoco210/bin
ENV D4RL_SUPPRESS_IMPORT_ERROR=1

# Import d4rl and gym to cache the installation
RUN python3.9 -c "import d4rl, gym"

WORKDIR /home/duser/policy-guided-diffusion
