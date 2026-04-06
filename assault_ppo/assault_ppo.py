'''
using ppo to solve assault problem
'''

import gymnasium as gym
import pickle, random, os, numpy, torch
import ale_py
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
        
        self.shared = torch.nn.Linear(520, 512)
        
        self.actor1 = torch.nn.Linear(512, 256)
        self.actor2 = torch.nn.Linear(256, 7)
        self.critic1 = torch.nn.Linear(512, 256)
        self.critic2 = torch.nn.Linear(256, 1)
        
    def forward(self, state):  
        state = torch.tensor(state/255, dtype=torch.float32).permute(2,0,1).unsqueeze(0).to(device)
        
        s = torch.nn.functional.relu(self.conv1(state))
        s = torch.nn.functional.max_pool2d(s, kernel_size = 2, stride = 2)
        s = torch.nn.functional.relu(self.conv2(s))
        s = torch.nn.functional.max_pool2d(s, kernel_size = 2, stride = 2)
        s = torch.nn.functional.relu(self.conv3(s))
        s = torch.nn.functional.max_pool2d(s, kernel_size = 2, stride = 2)
        
        s = s.flatten(start_dim=1)
        s = torch.nn.functional.relu(self.shared(s))
        #actor
        A = torch.nn.functional.relu(self.actor1(s))
        A = torch.nn.functional.softmax(self.actor2(A), dim=-1) # output a probablity distribution
        #critic
        C = torch.nn.functional.relu(self.critic1(s))
        C = self.critic2(C) # output the state value
        
        # build up a distribution of actions
        dist = Categorical(logits=A)
        # sample from the distribution
        action = dist.sample()
        # calculate the probability of choosing the current action
        prob = dist.log_prob(action)
        #calculate out the entropy of the distributiob
        entropy = dist.entropy()
        
        return action.detach().cpu().numpy(), prob, entropy.detach(), C
        
AC_model = AC().to(device)
#AC_model = torch.load("assault/model.pth", weights_only=False).to(device)
optimizer = torch.optim.Adam(AC_model.parameters(), lr=learning_ratio)




def collect(number_of_states):
    observations = [0]*number_of_states
    dones = torch.zeros(number_of_states).to(device)
    rewards = torch.zeros(number_of_states).to(device)
    values = torch.zeros(number_of_states).to(device)
    actions = [0]*number_of_states
    action_probs = torch.zeros(number_of_states).to(device)
    Advantages = torch.zeros(number_of_states).to(device)
    A = 0
    
    env = gym.make("ALE/Assault-v5", render_mode=None)
    observation = env.reset()[0]
    
    #collect states information
    for i in range(number_of_states):
        observations[i] = observation
        # get action, action_prob, value
        actions[i], action_probs[i], _, values[i] = AC_model(observation)
        # get reward, done
        print(actions[i])
        next_observation, reward, done, _ = env.step(actions[i])
        rewards[i] = torch.tensor(reward, dtype=torch.float32).to(device)
        dones[i] = torch.tensor(done, dtype=torch.float32).to(device)
        if dones[i]:
            break
        else:
            observation = next_observation
            
    #calculate the advantages value
    Number = len(observation)
    with torch.no_grad():
        for i in range(Number):
            Advantages[Number - 1 - i] = A
            if Number-1-i != 0:
                A = discounted*A + discounted*values[i] - values[i-1] + rewards[i-1]
            
    return observations, dones, rewards, values, actions, action_probs, Advantages




def train(collection):
    for k in range(10):
        new_action_probs = torch.zeros(len(collection)).to(device)
        new_value = torch.zeros(len(collection)).to(device)
        new_entropy = torch.zeros(len(collection)).to(device)
        for id,(observation, done, reward, value, action, action_prob, Advantage) in enumerate(collection):
            _,new_action_probs[id],new_entropy[id],new_value[id] = AC_model(observation)
        
        rt = new_action_probs/collection[5]
        com1 = torch.clamp(rt, 1-0.2, 1+0.2)*collection[6]
        com2 = rt*collection[6]
        loss_actor = torch.min(com1, com2).mean()
        loss_critic = torch.mse(new_value - collection[3])
        Loss = -loss_actor + 0.5*loss_critic - 0.05*new_entropy.mean()
        
        optimizer.zero_grad()
        Loss.backward()
        optimizer.step()
    
for i in range(5):
    collection = collect(50)
    print(torch.sum(collection[2]))
    train(collection)
    
