python scripts/train.py \
    --bddl_file_name libero_90/LIVING_ROOM_SCENE1_pick_up_the_ketchup_and_put_it_in_the_basket.bddl \
    --alg ppo \
    --save_path ~/../../var/local/faraz/models/0916_pick_up_ketchup \
    --total_timesteps 200000 \
    --seed 21 \
    --num_envs 8 \