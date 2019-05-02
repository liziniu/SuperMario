python run.py \
    --env SuperMarioBros-v0 \
    --env_type atari \
    --network cnn \
    --num_env 4 \
    --reward_scale 0.0667 \
    --alg acer \
    --num_timesteps 1e6 \
    --aux_task RF \
    --gpu 12,13,2