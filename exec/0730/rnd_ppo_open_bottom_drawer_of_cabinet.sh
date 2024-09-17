#/bin/bash

for seed in 13 14 14
do
    # MUJOCO_GL=egl
    # MUJOCO_GL=osmesa
    python scripts/train.py \
        --bddl_file_name libero_90/KITCHEN_SCENE1_open_the_bottom_drawer_of_the_cabinet.bddl \
        --alg ppo \
        --save_path ~/../../var/local/faraz/models/0731_rnd_ppo_open_the_bottom_drawer_of_the_cabinet_seed21 \
        --exploration_alg rnd \
        --learning_rate 0.0003 \
        --total_timesteps 1000000 \
        --seed 21 \
        --num_envs 8 \
        --wandb \
        --wandb_entity farazh \
        --device cuda:$1
done