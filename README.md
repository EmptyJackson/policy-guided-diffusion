<h1 align="center">Policy-Guided Diffusion</h1>

<p align="center">
    <a href= "https://arxiv.org/abs/2404.06356">
        <img src="https://img.shields.io/badge/arXiv-2404.06356-b31b1b.svg" /></a>
    <a href= "https://github.com/psf/black">
        <img src="https://img.shields.io/badge/code%20style-black-000000.svg" /></a>
    <a href= "https://github.com/EmptyJackson/policy-guided-diffusion/blob/main/LICENSE">
        <img src="https://img.shields.io/badge/license-MIT-blue.svg" /></a>
    <a href= "https://twitter.com/JacksonMattT/status/1778090124862959733">
        <img src="https://img.shields.io/badge/Twitter-1DA1F2.svg" /></a>
</p>

<p align="center">
    <img src="media/demo.gif" alt="animated" width="75%"/>
</p>

The official implementation of *Policy-Guided Diffusion* (https://arxiv.org/abs/2404.06356) - built by [Matthew Jackson](https://github.com/EmptyJackson) and [Michael Matthews](https://github.com/MichaelTMatthews).

<!-- [[ArXiv](https://arxiv.org/abs/2404.06356) | [NeurIPS RoboLearn Workshop](https://neurips.cc/virtual/2023/77263) | [Twitter]()] -->

   * Offline RL agents (**TD3+BC**, **IQL**),
   * Trajectory-level **U-Net** diffusion model,
   * **EDM** diffusion training and sampling,
   * Runs on the **D4RL** benchmark.

Diffusion and agent training is implemented entirely in Jax, with extensive JIT-compilation and parallelization!

<!-- [**Running experiments**](#running-experiments) | [**Citation**](#cite) -->

# Running experiments
Diffusion and agent training is executed with `python3 train_diffusion.py` and `python3 train_agent.py`, with all arguments found in [`util/args.py`](https://github.com/EmptyJackson/policy-guided-diffusion/blob/main/util/args.py).
* `--log --wandb_entity [entity] --wandb_project [project]` enables logging to WandB.
* `--debug` disables JIT compilation.

### Docker installation
1. Build docker image
```
cd docker & ./build.sh & cd ..
```

2. (To enable WandB logging) Add your [account key](https://wandb.ai/authorize) to `setup/wandb_key`:
```
echo [KEY] > setup/wandb_key
```

### Launching experiments
```
./run_docker.sh [GPU index] python3.9 [train_script] [args]
```
Diffusion training example:
```
./run_docker.sh 0 python3.9 train_diffusion.py --log --wandb_project diff --wandb_team flair --dataset_name walker2d-medium-v2
```
Agent training example:
```
./run_docker.sh 6 python3.9 train_agent.py --log --wandb_project agents --wandb_team flair --dataset_name walker2d-medium-v2 --agent iql
```

# Citation
If you use this implementation in your work, please cite us with the following:
```
@misc{jackson2024policyguided,
      title={Policy-Guided Diffusion},
      author={Matthew Thomas Jackson and Michael Tryfan Matthews and Cong Lu and Benjamin Ellis and Shimon Whiteson and Jakob Foerster},
      year={2024},
      eprint={2404.06356},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
