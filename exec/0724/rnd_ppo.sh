#/bin/bash

for seed in 13 14 14
do
    # MUJOCO_GL=egl
    # MUJOCO_GL=osmesa
    python scripts/train.py \
        --bddl_file_name libero_90/KITCHEN_SCENE3_turn_on_the_stove.bddl \
        --alg ppo\
        --save_path ~/../../var/local/faraz/models/0730_rnd_ppo \
        --exploration_alg rnd \
        --learning_rate 0.0003 \
        --total_timesteps 1000000 \
        --num_envs 8 \
        --wandb \
        --wandb_entity farazh \
        --device cuda:$1
done