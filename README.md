# Curriculum Learning for Visual Manipulation
Open-ended curriculum learning for visual manipulation tasks

## Benchmarking Single RL training for Tasks in LIBERO
```
python scripts/train.py --bddl_file_name libero_90/KITCHEN_SCENE6_close_the_microwave.bddl
```
To enable `Wandb` logging, simply pass two additional arguments: `--wandb`, `--wandb_entity <YOUR_WANDB_ENTITY>`

## LIBERO Installation
LIBERO is a pre-requisite for running this repo.
To install LIBERO and its dependency, refer to the [LIBERO Github page](https://github.com/Lifelong-Robot-Learning/LIBERO).