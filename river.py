'''
using q-learning for river problem
'''
import gymnasium as gym
import pickle
import random
import os
import numpy
import torch
import ale_py

greedy = 0.8
learning_ratio = 0.001
discounted = 0.99
device = torch.device('cuda')
class Q(torch.nn.Module):
    def __init__(self):
        super().__init__()
        #state
        self.conv1 = torch.nn.Conv2d(3, 10, kernel_size=10, stride=10, padding=0)
        self.conv2 = torch.nn.Conv2d(10, 1, kernel_size=3, stride=1, padding=1)
        self.linear1 = torch.nn.Linear(21*16, 150)
        self.linear2 = torch.nn.Linear(150, 100)
        # combine
        self.linear3 = torch.nn.Linear(101, 36)
        self.linear4 = torch.nn.Linear(36, 28)
        self.linear5 = torch.nn.Linear(28, 1)
        
        
    def forward(self, state, action):  
        state = torch.tensor(state, dtype=torch.float32).permute(2,0,1).unsqueeze(0).to(device)
        action = torch.tensor([[action]], dtype=torch.float32).to(device)
        
        s = self.conv2(self.conv1(state))
        s = s.view(s.size(0), -1)
        s = torch.nn.functional.relu(self.linear2(torch.nn.functional.relu(self.linear1(s))))
        
        x = torch.nn.functional.relu(self.linear3(torch.cat([s, action], dim = -1)))
        x = torch.nn.functional.relu(self.linear4(x))
        x = self.linear5(x)
        
        return x

Q_model = Q().to(device)
#Q_model = torch.load("river/model.pth", weights_only=False).to(device)
optimizer = torch.optim.Adam(Q_model.parameters(), lr=learning_ratio)
def take_action(s):
    a = random.random()
    if a>greedy:
        actions = [i for i in range(18)]
        output = []
        for action in actions:
            output.append(Q_model(s, action).detach()[0].squeeze())
        return actions[output.index(max(output))] 
    else:
        return random.randint(0, 6)

def find_max(s):
    actions = [i for i in range(18)]
    output = []
    for action in actions:
        output.append(Q_model(s, action).detach()[0].squeeze())
    return max(output)
        
env = gym.make("ALE/Riverraid-v5", render_mode=None)
observation, _ = env.reset()

for i in range(1):
    episode_over = False
    rr = 0
    epoch = 0
    while not episode_over:
        epoch += 1
        action = take_action(observation)
        old_q = Q_model(observation, action)[0]
        new_observation, reward, terminated, truncated, _ = env.step(action)
        rr += reward
        
        #update
        episode_over = terminated or truncated
        # if episode is over
        if episode_over:
            observation, _ = env.reset()
        # if the episode is not over
        with torch.no_grad():
            target = reward + discounted * find_max(new_observation)
        loss = (old_q - target)**2
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        observation = new_observation
    greedy -= 0.03
    torch.save(Q_model, "river/model.pth")
    print(rr)

