#/bin/bash

for seed in 13 14 14
do
    # MUJOCO_GL=egl
    # MUJOCO_GL=osmesa
    python scripts/train.py \
        --bddl_file_name libero_90/KITCHEN_SCENE3_turn_on_the_stove.bddl \
        --exploration_alg icm \
        --total_timesteps 1000000 \
        --num_envs 8 \
        --wandb \
        --wandb_entity zifanxu \
        --device cuda:$1
done