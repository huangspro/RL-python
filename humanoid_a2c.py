#actor-critic version


import gymnasium as gym
import torch
import numpy
import torchvision
import os
import torch.distributions

device = torch.device("cuda")
learning_rate = 2e-6
discounted = 0.99

class PI(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(348, 512)
        self.linear2 = torch.nn.Linear(512, 512)
        self.linear3 = torch.nn.Linear(512, 512)
        self.linear4 = torch.nn.Linear(512, 34)
    def forward(self, x):  
        x = torch.nn.functional.relu(self.linear1(x))
        x = torch.nn.functional.relu(self.linear2(x))
        x = torch.nn.functional.relu(self.linear3(x))
        x = torch.nn.functional.tanh(self.linear4(x))
        return x
        
class V(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(348, 512)
        self.linear2 = torch.nn.Linear(512, 512)
        self.linear3 = torch.nn.Linear(512, 512)
        self.linear4 = torch.nn.Linear(512, 1)
    def forward(self, x):  
        x = torch.nn.functional.relu(self.linear1(x))
        x = torch.nn.functional.relu(self.linear2(x))
        x = torch.nn.functional.relu(self.linear3(x))
        x = self.linear4(x)
        return x
if os.path.exists("bot/PI_model.pth") and os.path.exists("bot/V_model.pth"):
    PI_model = torch.load("bot/PI_model.pth", weights_only=False).to(device)
    V_model = torch.load("bot/V_model.pth", weights_only=False).to(device)
else:
    PI_model = PI().to(device)
    V_model = V().to(device)



optimizer_PI = torch.optim.SGD(PI_model.parameters(), lr = learning_rate)
optimizer_V = torch.optim.SGD(V_model.parameters(), lr = learning_rate)
env = gym.make("Humanoid-v5", render_mode="human")


epoch = 0
total_reward = 0.01
total_number = 0.01
r = 0
for i in range(0, 1000):
    observation, _ = env.reset()
    episode_over = False

    
    while not episode_over:
        #observe and sample the actions
        output = PI_model(torch.tensor(observation, dtype=torch.float32).to(device))
        Value = V_model(torch.tensor(observation, dtype=torch.float32).to(device))
        mu = output[:17]
        div = torch.nn.functional.softplus(output[17:]) + 1e-5
        
        dist = torch.distributions.Normal(mu, div)
        action = dist.sample()
        entropy = dist.entropy().sum()
        action_prob = dist.log_prob(action).sum()
        #remember the state and actions
        observation, reward, terminated, truncated, _ = env.step(action.detach().cpu().numpy())
        r += reward
        advantage = reward + discounted * V_model(torch.tensor(observation, dtype=torch.float32).to(device)).detach() - Value
        loss1 = torch.mean(advantage**2)
        loss2 = (-advantage.detach() * action_prob) - 0.5*entropy
        
        optimizer_V.zero_grad()
        loss1.backward()
        optimizer_V.step()

        optimizer_PI.zero_grad()
        loss2.backward()
        optimizer_PI.step()
        
        episode_over = terminated or truncated
        if episode_over:
            observation, _ = env.reset()
    epoch += 1
    if epoch%100 == 0:
        torch.save(PI_model, "bot/PI_model.pth")
        torch.save(V_model, "bot/V_model.pth")   
        print(r/100, entropy)
        r=0
env.close()





























