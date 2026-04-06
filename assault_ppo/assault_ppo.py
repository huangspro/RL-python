'''
using ppo to solve assault problem
'''

import gymnasium as gym
import pickle
import random
import os
import numpy
import torch
from torch.distributions.categorical import Categorical

greedy = 0
learning_ratio = 0.0001
discounted = 0.99
device = torch.device('cpu')

class AC(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1)
        
        self.shared = torch.nn.Linear(105*80, 128)
        self.actor1 = torch.nn.Linear(128, 128)
        self.actor2 = torch.nn.Linear(128, 7)
        self.critic1 = torch.nn.Linear(128, 256)
        self.critic2 = torch.nn.Linear(256, 1)
        
    def forward(self, state):  
        state = torch.tensor(state/255, dtype=torch.float32).permute(2,0,1).unsqueeze(0).to(device)
        s = torch.nn.functional.relu(self.conv1(state))
        s = torch.nn.functional.max_pool2d(s, kernel_size = 2, stride = 2)
        s = torch.nn.functional.relu(self.conv2(state))
        s = torch.nn.functional.max_pool2d(s, kernel_size = 2, stride = 2)
        s = torch.nn.functional.relu(self.conv3(state))
        s = torch.nn.functional.max_pool2d(s, kernel_size = 2, stride = 2)
        
        s = s.flatten(start_dim=1)
        s = torch.nn.functional.relu(self.shared(s))
        #actor
        A = torch.nn.functional.relu(self.actor1(s))
        A = torch.nn.functional.softmax(self.actor2(s)) # output a probablity distribution
        #critic
        C = torch.nn.functional.relu(self.critic1(x))
        C = self.critic2(x) # output the state value
        
        # build up a distribution of actions
        dist = Categorical(logits=A)
        # sample from the distribution
        action = dist.sample()
        # calculate the probability of choosing the current action
        prob = dist.log_prob(action)
        #calculate out the entropy of the distributiob
        entropy = dist.entropy()
        
        return action, prob, entropy, C
        
AC_model = AC().to(device)
AC_model = torch.load("assault/model.pth", weights_only=False).to(device)


