import torch 
import gymnasium as gym
from stable_baselines3 import A2C
import wandb
import tensorboard

wandb.init(project="a2c_training", sync_tensorboard=True)

env = gym.make("CartPole-v1", render_mode="rgb_array")

model = A2C("MlpPolicy", env, verbose=1, tensorboard_log='./a2c_cartpole_tensorboard')
model.learn(total_timesteps=25000, tb_log_name="training")

vec_env = model.get_env()
obs = vec_env.reset()
for i in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    vec_env.render("human")

model.save("trained_a2c/a2c_cartpole")