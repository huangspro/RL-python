import gymnasium as gym
import pickle, random, os, numpy, torch
import ale_py

device = torch.device('cuda')

class world_model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Linear(8, 256)
        self.layer2 = torch.nn.Linear(256, 128)
        self.bn = torch.nn.Linear(128, 4)
        
        self.layer4 = torch.nn.Linear(4, 128)
        self.layer5 = torch.nn.Linear(128, 256)
        self.layer6 = torch.nn.Linear(256, 8)
        
    def forward(self, state, mode="training"):  
        if mode == "training":
            x = torch.nn.functional.relu(self.layer1(state))
            x = torch.nn.functional.relu(self.layer2(x))
            x = self.bn(x)
            x = torch.nn.functional.relu(self.layer4(x))
            x = torch.nn.functional.relu(self.layer5(x))
            x = self.layer6(x)
            return x
        else:
            x = torch.nn.functional.relu(self.layer1(state))
            x = torch.nn.functional.relu(self.layer2(x))
            x = self.bn(x)
            return x


#WM = world_model().to(device)
WM = torch.load("model.pth", weights_only=False).to(device)
optimizer = torch.optim.Adam(WM.parameters(), lr=1e-4)
criterion = torch.nn.MSELoss()

env = gym.make("LunarLander-v3", render_mode=None)
observation = env.reset()[0]

for i in range(2000):
    L = 0
    episode_over = False
    while not episode_over:
        new_observation, reward, terminated, truncated, _ = env.step(env.action_space.sample())
        new_observation = torch.tensor(new_observation, dtype = torch.float32).to(device)
        
        output = WM(new_observation)
        if i%100==0:
            print(output, new_observation)
        loss = criterion(output, new_observation)
        L += loss.detach().item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()    
        
        episode_over = terminated or truncated
        observation = new_observation
        if episode_over:
            observation, _ = env.reset()
    print(L)
torch.save(WM, "model.pth")    
