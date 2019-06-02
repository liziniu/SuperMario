env=SuperMarioBrosGoal-1-1-v0
env_type=atari
network=cnn
alg=her
num_timesteps=5e5
seed=2019
num_env=1
num_exp=1
gpu=6
desired_x_pos=500
policy_inputs=7
replay_k=1

python run.py --env ${env} --env_type ${env_type} --num_env ${num_env} --network ${network} --seed ${seed} \
              --alg ${alg} --num_timesteps ${num_timesteps} --gpu ${gpu} --desired_x_pos ${desired_x_pos} \
              --replay_k ${replay_k} --policy_inputs ${policy_inputs} --num_exp ${num_exp}