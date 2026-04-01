import gymnasium as gym
import torch
import numpy
import torchvision
import os

device = torch.device("cuda")
learning_rate = 1e-4
discounted = 0.99

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToPILImage(),
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])

if os.path.exists("carracing/PI_model.pth"):
    PI_model = torch.load("carracing/PI_model.pth", weights_only=False)
else:
    PI_model = torchvision.models.resnet18(pretrained=True)

PI_model.fc = torch.nn.Linear(PI_model.fc.in_features, 4)
PI_model = PI_model.to(device)

if os.path.exists("carracing/V_model.pth"):
    V_model = torch.load("carracing/V_model.pth", weights_only=False)
else:
    V_model = torchvision.models.resnet18(pretrained=True)

V_model.fc = torch.nn.Linear(V_model.fc.in_features, 1)
V_model = V_model.to(device)
'''
class PI(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        
        self.linear1 = torch.nn.Linear(24*24, 256)
        self.linear2 = torch.nn.Linear(256, 128)
        self.linear3 = torch.nn.Linear(128, 4)
    def forward(self, x):  
        x = self.conv1(x)
        x = torch.nn.functional.max_pool2d(x, kernel_size = 2, stride = 2)
        x = self.conv2(x)
        x = torch.nn.functional.max_pool2d(x, kernel_size = 2, stride = 2)
        x = self.conv3(x)
        x = torch.flatten(x, start_dim=1)
        
        x = torch.nn.functional.relu(self.linear1(x))
        x = torch.nn.functional.relu(self.linear2(x))
        x = self.linear3(x)
        return x
        
class V(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        
        self.linear1 = torch.nn.Linear(24*24, 256)
        self.linear2 = torch.nn.Linear(256, 128)
        self.linear3 = torch.nn.Linear(128, 1)  #one output, represtenting the state value   
    def forward(self, x):  
        x = self.conv1(x)
        x = torch.nn.functional.max_pool2d(x, kernel_size = 2, stride = 2)
        x = self.conv2(x)
        x = torch.nn.functional.max_pool2d(x, kernel_size = 2, stride = 2)
        x = self.conv3(x)
        
        x = torch.flatten(x, start_dim=1)
        
        x = torch.nn.functional.relu(self.linear1(x))
        x = torch.nn.functional.relu(self.linear2(x))
        x = self.linear3(x)
        return x
        
PI_model, V_model = PI(), V()
PI_model, V_model = PI_model.to(device), V_model.to(device)
'''
optimizer_PI = torch.optim.Adam(PI_model.parameters(), lr = learning_rate)
optimizer_V = torch.optim.Adam(V_model.parameters(), lr = learning_rate)
env = gym.make("CarRacing-v3", render_mode="human")
temm = 0
for i in range(0, 5):
    if temm > 400:
        observation, _ = env.reset(seed=42)
        temm = 0
    if i == 0:
        observation, _ = env.reset(seed=42)
    #collect an episode
    state, A, R, action_prob = [], [], [], 0
    episode_over = False
    state.append(observation)
    tem = 0
    while not episode_over:
        output = torch.nn.functional.softmax(PI_model(transform(observation).unsqueeze(0).to(device)), dim = 1)
        m = torch.distributions.Categorical(output)
        action_index = m.sample().item()
        action = 0
        if action_index == 0:
            action = numpy.array([1,1,0])
        elif action_index == 1:
            action = numpy.array([-1,1,0])
        elif action_index == 2:
            action = numpy.array([0,1,0])
        elif action_index == 3:
            action = numpy.array([0,0.3,0.3])
        A.append(action)
        observation, reward, terminated, truncated, _ = env.step(action)
        state.append(observation)
        R.append(reward)
        episode_over = terminated or truncated  
        if episode_over:
            observation, _ = env.reset(seed=42)
        if tem > 100:
            break
        
        tem += 1
        temm += 1
    
    print(len(state), len(A))
    
    next_state = torch.stack([transform(state[ii+1]) for ii in range(len(state)-1)]).to(device)
    state.pop()
    state = torch.stack([transform(ii) for ii in state]).to(device)
    R = torch.tensor(R).to(device)
    action_prob = torch.log(torch.max(torch.nn.functional.softmax(PI_model(state), dim = 1), dim = 1).values)
    
    print(next_state.shape, state.shape)
    
    advantage = R + discounted * V_model(next_state) - V_model(state)
    loss1 = torch.mean(advantage**2)
    loss2 = torch.mean(-advantage.detach() * action_prob)
    
    optimizer_V.zero_grad()
    loss1.backward()
    optimizer_V.step()

    optimizer_PI.zero_grad()
    loss2.backward()
    optimizer_PI.step()
    
    torch.save(PI_model, "carracing/PI_model.pth")
    torch.save(V_model, "carracing/V_model.pth")   
    
env.close()





























