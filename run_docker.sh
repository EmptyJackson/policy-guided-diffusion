#!/bin/bash
gpu=$1
script_and_args="${@:2}"
WANDB_API_KEY=$(cat ./docker/wandb_key)
git pull

echo "Launching container pgd_$gpu on GPU $gpu"
docker run \
    --env CUDA_VISIBLE_DEVICES=$gpu \
    --gpus all \
    -e WANDB_API_KEY=$WANDB_API_KEY \
    -v $(pwd):/home/duser/policy-guided-diffusion \
    --name pgd\_$gpu \
    --user $(id -u) \
    --rm \
    pgd \
    /bin/bash -c "$script_and_args"
