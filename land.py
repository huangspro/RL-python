'''
using q-learning for BipedalWalker problem
'''
import gymnasium as gym
import pickle
import random
import os
import numpy
import torch

greedy = 0.8
learning_ratio = 0.005
discounted = 0.99
device = torch.device('cpu')
class Q(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(8+1, 28)
        self.linear2 = torch.nn.Linear(28, 14)
        self.linear3 = torch.nn.Linear(14, 1)
    def forward(self, state, action):  
        state = torch.tensor(state, dtype=torch.float32).to(device)
        action = torch.tensor([action], dtype=torch.float32).to(device)
        x = torch.nn.functional.relu(self.linear1(torch.cat([state, action])))
        x = torch.nn.functional.relu(self.linear2(x))
        x = self.linear3(x)
        return x

#Q_model = Q().to(device)
Q_model = torch.load("land/model.pth", weights_only=False).to(device)
def take_action(s):
    a = random.random()
    if a>greedy:
        return random.randint(0, 3)
    else:
        actions = [0,1,2,3]
        output = []
        for action in actions:
            output.append(Q_model(s, action).detach().squeeze())
        return actions[output.index(max(output))]     

def find_max(s):
    actions = [0,1,2,3]
    output = []
    for action in actions:
        output.append(Q_model(s, action).detach().squeeze())
    return max(output)
        
env = gym.make("LunarLander-v3", render_mode=None)
observation, _ = env.reset()

for i in range(2000):
    episode_over = False
    rr = 0
    epoch = 0
    while not episode_over:
        epoch += 1
        action = take_action(observation)
        old_q = Q_model(observation, action)
        new_observation, reward, terminated, truncated, _ = env.step(action)
        rr += reward
        
        old_q.backward()
        
        #update
        episode_over = terminated or truncated
        # if episode is over
        if episode_over:
            observation, _ = env.reset()
        # if the episode is not over
        else:
             with torch.no_grad():
                for p in Q_model.parameters():
                    p += learning_ratio*(reward - find_max(new_observation) - old_q)*p.grad
                    p.grad.zero_()
        observation = new_observation
        
    greedy -= 0.0007
    if epoch%10 == 0:
        torch.save(Q_model, "land/model.pth")
        print(rr)

