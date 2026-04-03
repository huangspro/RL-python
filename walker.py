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
learning_ratio = 0.5
discounted = 0.99

class Q(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(24+4, 512)
        self.linear2 = torch.nn.Linear(512, 512)
        self.linear3 = torch.nn.Linear(512, 512)
        self.linear4 = torch.nn.Linear(512, 1)
    def forward(self, state, action):  
        x = torch.nn.functional.relu(self.linear1(torch.cat([state, action], dim=-1)))
        x = torch.nn.functional.relu(self.linear2(x))
        x = torch.nn.functional.relu(self.linear3(x))
        x = torch.nn.functional.tanh(self.linear4(x))
        return x

Q_model = Q()

def take_action(s):
    a = random.random()
    if a>greedy:
        return numpy.array([random.random()*2-1,random.random()*2-1,random.random()*2-1,random.random()*2-1])
    else:
        A = []
        output = []
        for i1 in numpy.arange(0, 2, 1):
            for i2 in numpy.arange(0, 2, 1):
                for i3 in numpy.arange(0, 2, 1):
                    for i4 in numpy.arange(0, 2, 1):
                        action = [i1-1,i2-1,i3-1,i4-1]
                        A.append(action)
                        output.append(Q_model(torch.tensor(s, dtype=torch.float32), torch.tensor(action, dtype=torch.float32)).detach())
        return numpy.array(A[output.index(max(output))])        

def find_max(s):
    output = []
    for i1 in numpy.arange(0, 2, 1):
        for i2 in numpy.arange(0, 2, 1):
            for i3 in numpy.arange(0, 2, 1):
                for i4 in numpy.arange(0, 2, 1):
                    output.append(Q_model(torch.tensor(s, dtype=torch.float32), torch.tensor(action, dtype=torch.float32)).detach())
    return max(output)
        
env = gym.make("BipedalWalker-v3", render_mode=None)
observation, _ = env.reset()

for i in range(50):
    episode_over = False
    rr = 0
    epoch = 0
    while not episode_over:
        epoch += 1
        if epoch > 100:
            break
        action = take_action(observation)
        old_q = Q_model(torch.tensor(observation, dtype=torch.float32), torch.tensor(action, dtype=torch.float32))
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
    greedy -= 0.00002
    
    print(rr)

