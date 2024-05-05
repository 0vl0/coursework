import matplotlib.pyplot as plt
import gym
import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.optim import Adam
import numpy as np 

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


class CartPoleNN(nn.Module):

    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        """ Get probability distribution over actions. """
        return self.mlp(x)
    
    def get_actions(self, state):
        """ Sample an action based on the probability distribution. """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self.forward(state).cpu()
        m = Categorical(probs)
        action = m.sample() 
        return action.item(), m.log_prob(action)


N_epoch = 500
env = gym.make("CartPole-v1", render_mode=None)
model_cartpole = CartPoleNN()
model_cartpole.to(device)

LR = 5e-3
optimizer = Adam(model_cartpole.parameters(), lr=LR)

gamma = 0.99
total_rewards = [0]*N_epoch

for epoch in range(N_epoch):
    terminated, truncated = False, False
    log_probabilities = []
    rewards = []
    state = env.reset()[0]
    while not (terminated or truncated):
        action, log_proba = model_cartpole.get_actions(state)
        state, reward, terminated, truncated, info = env.step(action)
        log_probabilities.append(log_proba)
        rewards.append(reward)
    
    total_rewards[epoch] = sum(rewards)
    if epoch % 10 == 1:
        print(f'Epoch {epoch}, total reward = {total_rewards[epoch]}')

    discounted_rewards = [0]*len(rewards)
    discounted_rewards[0] = sum((gamma**i)*rewards[i] for i in range(len(rewards)))
    for i in range(1,len(rewards)):
        discounted_rewards[i] = (discounted_rewards[i-1]-rewards[i-1])/gamma
    discounted_rewards = (discounted_rewards-np.mean(discounted_rewards))/len(discounted_rewards)
    discounted_rewards = torch.tensor(discounted_rewards)

    loss = []
    for log_prob, disc_return in zip(log_probabilities, discounted_rewards):
        loss.append(-log_prob * disc_return)
    loss = torch.cat(loss).sum()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


plt.plot(range(N_epoch), total_rewards)
plt.xlabel('Epoch')
plt.ylabel('Total Rewards')
plt.title('Total Rewards per Epoch')
plt.grid(True)
plt.savefig('rewards_across_epoch.png')