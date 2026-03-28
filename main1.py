import gymnasium as gym
import torch
import numpy
import torchvision
import os

device = torch.device("cuda")
learning_rate = 1e-5
discounted = 0.99


class PI(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(4, 256)
        self.linear2 = torch.nn.Linear(256, 512)
        self.linear3 = torch.nn.Linear(512, 256)
        self.linear4 = torch.nn.Linear(256, 128)
        self.linear5 = torch.nn.Linear(128, 56)
        self.linear6 = torch.nn.Linear(56, 2)
    def forward(self, x):  
        x = torch.nn.functional.relu(self.linear1(x))
        x = torch.nn.functional.relu(self.linear2(x))
        x = torch.nn.functional.relu(self.linear3(x))
        x = torch.nn.functional.relu(self.linear4(x))
        x = torch.nn.functional.relu(self.linear5(x))
        x = self.linear6(x)
        return x
        
class V(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(4, 256)
        self.linear2 = torch.nn.Linear(256, 512)
        self.linear3 = torch.nn.Linear(512, 256)
        self.linear4 = torch.nn.Linear(256, 128)
        self.linear5 = torch.nn.Linear(128, 56)
        self.linear6 = torch.nn.Linear(56, 1)
    def forward(self, x):  
        x = torch.nn.functional.relu(self.linear1(x))
        x = torch.nn.functional.relu(self.linear2(x))
        x = torch.nn.functional.relu(self.linear3(x))
        x = torch.nn.functional.relu(self.linear4(x))
        x = torch.nn.functional.relu(self.linear5(x))
        x = self.linear6(x)
        return x
        
if os.path.exists("bar/PI_model.pth"):
    PI_model = torch.load("bar/PI_model.pth", weights_only=False).to(device)
else:
    PI_model = PI().to(device)

if os.path.exists("bar/V_model.pth"):
    V_model = torch.load("bar/V_model.pth", weights_only=False).to(device)
else:
    V_model = V().to(device)
    
optimizer_PI = torch.optim.Adam(PI_model.parameters(), lr = learning_rate)
optimizer_V = torch.optim.Adam(V_model.parameters(), lr = learning_rate)

env = gym.make("CartPole-v1", render_mode="human")
for i in range(0, 2000):
    observation, _ = env.reset(seed=42)
    episode_over = False
    current_obs = torch.tensor(observation).unsqueeze(0).to(device)
    
    R = 0
    while not episode_over:
        # take an action
        output = torch.nn.functional.softmax(PI_model(current_obs), dim = 1)
        m = torch.distributions.Categorical(output)
        action_index = m.sample()
        action = 0
        if action_index == 0:
            action = 0
        else:
            action = 1
        
        action_prob = output[0][action_index]
        observation, reward, terminated, truncated, info = env.step(action)
        R += reward
        next_state = torch.tensor(observation).unsqueeze(0).to(device)
        current_value = V_model(current_obs)  #the state value
        next_value = V_model(next_state)  #the state value of next state
        
        advantage = (reward + discounted * next_value.detach()[0][0] - current_value[0][0]).squeeze()
        advantage = torch.clamp(advantage, -0.8, 0.8)
        loss1 = advantage.pow(2).mean()
        loss2 = -advantage.detach()*torch.log(action_prob[0].squeeze())
        
        # 更新值网络
        optimizer_V.zero_grad()
        loss1.backward()
        optimizer_V.step()

        # 更新策略网络
        optimizer_PI.zero_grad()
        loss2.backward()
        optimizer_PI.step()
        
        current_obs = next_state
        episode_over = terminated or truncated  
        if terminated or truncated:
            observation, info = env.reset()
    print(R) 
    torch.save(PI_model, "bar/PI_model.pth")
    torch.save(V_model, "bar/V_model.pth")   
env.close()

