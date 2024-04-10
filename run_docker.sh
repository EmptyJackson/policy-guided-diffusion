#!/bin/bash
WANDB_API_KEY=$(cat ./docker/wandb_key)
git pull

if [ $1 == "all" ]; then
    gpus="0 1 2 3 4 5 6 7"
else
    gpus=$1
fi

for gpu in $gpus; do
    echo "Launching container pgd_$gpu on GPU $gpu"
    docker run \
        --env CUDA_VISIBLE_DEVICES=$gpu \
        --gpus all \
        -e WANDB_API_KEY=$WANDB_API_KEY \
        -v $(pwd):/home/duser/policy-guided-diffusion \
        --name pgd\_$gpu \
        --user $(id -u) \
        --rm \
        -d \
        pgd \
        /bin/bash -c "$script_and_args"
done
