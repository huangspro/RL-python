import gymnasium as gym
import torch
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
learning_rate = 1e-4
discounted = 0.99

# Policy network
class PI(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(1, 64)
        self.fc2 = torch.nn.Linear(64, 128)
        self.fc3 = torch.nn.Linear(128, 4)

    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Value network
class V(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(1, 64)
        self.fc2 = torch.nn.Linear(64, 128)
        self.fc3 = torch.nn.Linear(128, 1)

    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 加载模型
if os.path.exists("bar/PI_model.pth") and os.path.exists("bar/V_model.pth"):
    PI_model = torch.load("bar/PI_model.pth", weights_only=False).to(device)
    V_model = torch.load("bar/V_model.pth", weights_only=False).to(device)
else:
    PI_model = PI().to(device)
    V_model = V().to(device)

optimizer_PI = torch.optim.Adam(PI_model.parameters(), lr=learning_rate)
optimizer_V = torch.optim.Adam(V_model.parameters(), lr=learning_rate)

env = gym.make("FrozenLake-v1", render_mode='human')
observation, _ = env.reset(seed=42)
rr = 0
for step in range(10000):
    # 当前状态
    state = torch.tensor([[observation]], dtype=torch.float32).to(device)  # shape [1,1]

    # 策略网络输出动作
    logits = PI_model(state)
    action_prob = torch.nn.functional.softmax(logits, dim=1)
    action_dist = torch.distributions.Categorical(action_prob)
    action = action_dist.sample()  # 采样动作
    action_index = action.item()

    # 环境执行动作
    next_observation, reward, terminated, truncated, _ = env.step(action_index)
    next_state = torch.tensor([[next_observation]], dtype=torch.float32).to(device)
    rr += reward
    # 计算优势
    R = torch.tensor([[reward]], dtype=torch.float32).to(device)
    advantage = R + discounted * V_model(next_state) * (1 - int(terminated)) - V_model(state)

    # 损失
    value_loss = advantage.pow(2).mean()
    policy_loss = -(advantage.detach() * action_dist.log_prob(action).unsqueeze(1)).mean()

    # 更新价值网络
    optimizer_V.zero_grad()
    value_loss.backward()
    optimizer_V.step()

    # 更新策略网络
    optimizer_PI.zero_grad()
    policy_loss.backward()
    optimizer_PI.step()

    # 保存模型
    if step % 50 == 0:
        torch.save(PI_model, "bar/PI_model.pth")
        torch.save(V_model, "bar/V_model.pth")

    # 更新状态
    observation = next_observation
    if terminated or truncated:
        observation, _ = env.reset(seed=42)
print(rr)
env.close()
