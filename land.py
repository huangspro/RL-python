'''
using q-learning for BipedalWalker problem
'''
import gymnasium as gym
import pickle
import random
import os
import numpy
import torch

greedy = 0
learning_ratio = 0.0001
discounted = 0.99
device = torch.device('cpu')
class Q(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(8+1, 128)
        self.linear2 = torch.nn.Linear(128, 256)
        self.linear3 = torch.nn.Linear(256, 512)
        self.linear4 = torch.nn.Linear(512, 1)
    def forward(self, state, action):  
        state = torch.tensor(state, dtype=torch.float32).to(device)
        action = torch.tensor([action], dtype=torch.float32).to(device)
        x = torch.nn.functional.relu(self.linear1(torch.cat([state, action])))
        x = torch.nn.functional.relu(self.linear2(x))
        x = torch.nn.functional.relu(self.linear3(x))
        x = self.linear4(x)
        return x

#Q_model = Q().to(device)
Q_model = torch.load("land/model.pth", weights_only=False).to(device)
optimizer = torch.optim.Adam(Q_model.parameters(), lr=learning_ratio)
def take_action(s):
    a = random.random()
    if a>greedy:
        actions = [0,1,2,3]
        output = []
        for action in actions:
            output.append(Q_model(s, action).detach().squeeze())
        return actions[output.index(max(output))]   
    else:
        return random.randint(0, 3)

def find_max(s):
    actions = [0,1,2,3]
    output = []
    for action in actions:
        output.append(Q_model(s, action).detach().squeeze())
    return max(output)
        
env = gym.make("LunarLander-v3", render_mode='human')
observation, _ = env.reset()

for i in range(10):
    episode_over = False
    rr = 0
    epoch = 0
    while not episode_over:
        epoch += 1
        action = take_action(observation)
        old_q = Q_model(observation, action)
        new_observation, reward, terminated, truncated, _ = env.step(action)
        rr += reward

        with torch.no_grad():
            target = reward + discounted * find_max(new_observation)
        loss = (old_q - target)**2
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        #update
        episode_over = terminated or truncated
        # if episode is over
        if episode_over:
            observation, _ = env.reset()
        observation = new_observation
        
    greedy -= 0.0007
    if epoch%10 == 0:
        torch.save(Q_model, "land/model.pth")
        print(rr)

