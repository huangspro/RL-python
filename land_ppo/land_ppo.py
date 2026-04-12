'''
using ppo to solve pole problem
'''

import gymnasium as gym
import pickle, random, os, numpy, torch
import ale_py
from torch.distributions.categorical import Categorical

learning_ratio = 4e-5
discounted = 0.99
device = torch.device('cuda')

class AC(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.shared1 = torch.nn.Linear(8, 128)
        self.shared2 = torch.nn.Linear(128, 256)
        self.shared3 = torch.nn.Linear(256, 128)
        
        self.actor1 = torch.nn.Linear(128, 4)
        self.critic1 = torch.nn.Linear(128, 1)
        
    def forward(self, state, Action=None):  
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        
        s = torch.nn.functional.relu(self.shared1(state))
        s = torch.nn.functional.relu(self.shared2(s))
        s = torch.nn.functional.relu(self.shared3(s))
        # actor
        A = torch.nn.functional.softmax(self.actor1(s), dim=-1) # output a probablity distribution
        # critic
        C = torch.nn.functional.relu(self.critic1(s)) # output the state value
        
        # build up a distribution of actions
        dist = Categorical(probs=A)
        # sample from the distribution
        action = dist.sample()
        # calculate the probability of choosing the current action
        if Action!=None:
            prob = dist.log_prob(torch.tensor(Action, dtype=torch.float32).to(device))
        else:
            prob = dist.log_prob(action)
        #calculate out the entropy of the distributiob
        entropy = dist.entropy()
        
        return action.detach().cpu().numpy(), prob, entropy.detach(), C
        
#AC_model = AC().to(device)
AC_model = torch.load("model/model.pth", weights_only=False).to(device)
optimizer = torch.optim.Adam(AC_model.parameters(), lr=learning_ratio)




def collect(number_of_states):
    # number_of_states is the maximum number of state expected
    observations = [0]*number_of_states
    rewards = torch.zeros(number_of_states).to(device)
    values = torch.zeros(number_of_states).to(device)
    actions = [0]*number_of_states
    action_probs = torch.zeros(number_of_states).to(device)
    # calculate the advantage function
    Advantages = torch.zeros(number_of_states).to(device)
    A = 0
    
    env = gym.make("LunarLander-v3", render_mode='human')
    observation = env.reset()[0]

    # collect states information
    for i in range(number_of_states):
        observations[i] = observation
        # get action, action_prob, value
        actions[i], action_probs[i], _, values[i] = AC_model(observation)
        # get reward, done
        next_observation, reward, a, b, _= env.step(int(actions[i][0]))
        rewards[i] = torch.tensor(reward, dtype=torch.float32).to(device)
        if a or b:
            break
        else:
            observation = next_observation
    env.close()
                    
    # calculate the advantages value, from back to begini
    Number = i + 1  # return Number
    with torch.no_grad():
        for t in reversed(range(Number)):
            delta = rewards[t] + discounted * (values[t+1] if t+1 < Number else 0) - values[t]
            A = delta + discounted * A
            Advantages[t] = A
    
    # output the number of practical number of states, and clip them
    return Number, observations[:Number], rewards[:Number], values[:Number], actions[:Number], action_probs.detach()[:Number], Advantages.detach()[:Number]




def train(collection):
    for k in range(3):     
        Number, observations, rewards, values, actions, action_probs, Advantages = collection
        # create new actions and values
        new_action_probs = torch.zeros(Number).to(device)
        new_value = torch.zeros(Number).to(device)
        new_entropy = torch.zeros(Number).to(device)

        
        for id in range(Number):
            _,new_action_probs[id],new_entropy[id],new_value[id] = AC_model(observations[id], actions[id])
        
        rt = torch.exp(new_action_probs - action_probs)
        Advantages = (Advantages - Advantages.mean()) / (Advantages.std() + 1e-8)
        com1 = torch.clamp(rt, 1-0.2, 1+0.2)*Advantages
        com2 = rt*Advantages
        loss_actor = torch.min(com1, com2).mean()
        loss_critic = torch.mean((new_value - Advantages - values.detach())**2)
        Loss = -loss_actor + 0.5*loss_critic - 0.01*new_entropy.mean()
        
        optimizer.zero_grad()
        Loss.backward()
        optimizer.step()
        
        #print(f"\t {k}, Loss: {Loss}")
        
        torch.save(AC_model, "model/model.pth")
        
for i in range(10):
    collection = collect(800)
    print("frames: ", collection[0])
    print("reward: ", sum(collection[2]).item())
    train(collection)
    
