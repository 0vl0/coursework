import torch 
import gymnasium as gym
from stable_baselines3 import A2C
import wandb
import tensorboard
import panda_gym

wandb.init(project='a2c_training', sync_tensorboard=True, name='pandareach')

env = gym.make('PandaReachJointsDense-v3')

model = A2C('MultiInputPolicy', env, verbose=1, tensorboard_log='./a2c_pandareach_tensorboard')
model.learn(total_timesteps=5e5, tb_log_name='training_panda_reach')

print('Training over. Saving model...')
model.save('trained_a2c/a2c_pandareach')

print('Model succesfully saved, rendering trained model...')
vec_env = model.get_env()
obs = vec_env.reset()
for i in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    vec_env.render('human')