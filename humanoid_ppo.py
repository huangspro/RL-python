#ppo version


import gymnasium as gym
import torch
import numpy
import torchvision
import os
import torch.distributions

device = torch.device("cuda")
learning_rate = 1e-6
discounted = 0.99

class PI(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(348, 512)
        self.linear2 = torch.nn.Linear(512, 512)
        self.linear3 = torch.nn.Linear(512, 34)
    def forward(self, x):  
        x = torch.nn.functional.relu(self.linear1(x))
        x = torch.nn.functional.relu(self.linear2(x))
        x = torch.nn.functional.tanh(self.linear3(x))*0.4
        return x
        
class V(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(348, 512)
        self.linear2 = torch.nn.Linear(512, 512)
        self.linear3 = torch.nn.Linear(512, 1)
    def forward(self, x):  
        x = torch.nn.functional.relu(self.linear1(x))
        x = torch.nn.functional.relu(self.linear2(x)) 
        x = self.linear3(x)
        return x
if os.path.exists("bot_ppo/PI_model.pth") and os.path.exists("bot_ppo/V_model.pth"):
    PI_model = torch.load("bot_ppo/PI_model.pth", weights_only=False).to(device)
    V_model = torch.load("bot_ppo/V_model.pth", weights_only=False).to(device)
else:
    PI_model = PI().to(device)
    V_model = V().to(device)



optimizer_PI = torch.optim.Adam(PI_model.parameters(), lr = learning_rate)
optimizer_V = torch.optim.Adam(V_model.parameters(), lr = learning_rate)
env = gym.make("Humanoid-v5", render_mode=None)

old_action_prob = 1
epoch = 0
for i in range(0, 5000):
    observation, _ = env.reset(seed=42)
    #collect an episode
    state, A, R, action_prob = [], [], [], 0
    episode_over = False
    state.append(observation)
    stepp = 0
    
    
    while not episode_over:
        output = PI_model(torch.tensor(observation, dtype=torch.float32).unsqueeze(0).to(device))
        mu = output[:, :17]
        div = torch.nn.functional.softplus(output[:, 17:]) + 1e-5
        
        dist = torch.distributions.Normal(mu, div)
        action = dist.sample()
        A.append(action)
        observation, reward, terminated, truncated, _ = env.step(action.detach().cpu().numpy()[0])
        state.append(observation)
        R.append(reward)
        episode_over = terminated or truncated
        if episode_over:
            observation, _ = env.reset(seed=42)
        stepp += 1
    print(sum(R), stepp)
    next_state = torch.stack([torch.tensor(state[ii+1], dtype=torch.float32) for ii in range(len(state)-1)]).to(device)
    state.pop()
    state = torch.stack([torch.tensor(ii, dtype=torch.float32) for ii in state]).to(device)
    R = torch.tensor(R, dtype=torch.float32).to(device)
    output = PI_model(torch.tensor(observation, dtype=torch.float32).unsqueeze(0).to(device))
    mu = output[:, :17]
    div = torch.nn.functional.softplus(output[:, 17:]) + 1e-5
    dist = torch.distributions.Normal(mu, div)
    action = dist.rsample()
    action_prob = dist.log_prob(action).sum(-1)
    
    
    advantage = R + discounted * V_model(next_state) - V_model(state)
    loss1 = torch.mean(advantage**2)
    
    clip = torch.clamp(action_prob/old_action_prob, 1-0.2, 1+0.2)
    loss2 = -torch.mean(torch.min(clip*advantage.detach(), advantage.detach()*action_prob/old_action_prob))
    old_action_prob = action_prob.detach().squeeze()
    
    optimizer_V.zero_grad()
    loss1.backward()
    optimizer_V.step()

    optimizer_PI.zero_grad()
    loss2.backward()
    optimizer_PI.step()
    
    epoch += 1
    
    if epoch%50 == 0:
        torch.save(PI_model, "bot_ppo/PI_model.pth")
        torch.save(V_model, "bot_ppo/V_model.pth")   
        
env.close()





























