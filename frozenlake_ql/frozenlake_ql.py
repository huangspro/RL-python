'''
using q-learning for frozen lake problem
'''
import gymnasium as gym
import pickle
import random
import os
Q = []
        
greedy = 0.8
learning_ratio = 0.5
discounted = 0.99
#prepare the action-value tabular
for i in range(16):
    Q.append([0,0,0,0])

def greedy_act(s):
    a = random.random()
    if(a>greedy):
        return Q[s].index(max(Q[s]))
    else:
        return random.randint(0, 3)
        


env = gym.make("FrozenLake-v1", render_mode=None, is_slippery=False)
observation, _ = env.reset()

for i in range(400000):
    episode_over = False
    rr = 0
    while not episode_over:
        action = greedy_act(observation)
        old_q = Q[observation][action]
        new_observation, reward, terminated, truncated, _ = env.step(action)
        #reward = -0.005 if reward==0 else 1
        rr += reward
        #update
        Q[observation][action] = old_q + learning_ratio*(reward + discounted*max(Q[new_observation]) - old_q)
        episode_over = terminated or truncated
        observation = new_observation
        if episode_over:
            observation, _ = env.reset()
        
    greedy -= 0.00002
    if i%100000 == 0:
        print(Q)
        print(rr)

env = gym.make("FrozenLake-v1", render_mode='human', is_slippery=False)
observation, _ = env.reset()
for i in range(5):
    episode_over = False
    rr = 0
    while not episode_over:
        action = greedy_act(observation)
        old_q = Q[observation][action]
        new_observation, reward, terminated, truncated, _ = env.step(action)
        reward = -0.005 if reward==0 else 1
        rr += reward
        episode_over = terminated or truncated
        observation = new_observation
        if episode_over:
            observation, _ = env.reset()
        
    
    if i%100000 == 0:
        print(Q)
        print(rr)
