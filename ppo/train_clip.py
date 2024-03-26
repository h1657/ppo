import torch
import gym
import numpy as np
import matplotlib.pyplot as plt
from ppo_clip import PPOAgent
from torch.distributions import Categorical
import torch.optim as optim
import torch.nn as nn


env = gym.make('LunarLander-v2' ,render_mode='rgb_array')
state_dim = 8   # 游戏的状态是个8维向量
action_dim = 4  # 游戏的输出有4个取值
update_timestep = 1200
lr = 0.002
betas = (0.9, 0.999)
gamma = 0.99
K_epochs = 4                # 控制了使用同一批数据进行策略优化的迭代次数
eps_clip = 0.2
random_seed = 1

agent = PPOAgent(state_dim, action_dim, lr, betas, gamma, K_epochs, eps_clip, update_timestep)

EPISODE_PER_BATCH = 5  # 决定了每次策略更新前要玩多少次游戏（即收集多少回合的数据）
NUM_BATCH = 200     # 定义了整个训练过程要重复多少次数据收集和策略更新的循环


avg_total_rewards, avg_final_rewards = [], []

for m in range(NUM_BATCH):

    log_probs, rewards = [], []
    total_rewards, final_rewards = [], []
    for episode in range(EPISODE_PER_BATCH):
        # 重开一把游戏
        state = env.reset()[0]
        total_reward, total_step = 0, 0
        for i in range(1000):

            # 第一步，根据当前策略给出动作，同时通过act计算出动作的对数概率
            # 第二步，与环境交互，得到真实奖励和下一步的状态和是否结束
            # 第三步，将真实奖励赋值到rewards数组便于后续画图和计算
            # 第四步，step进行不断地收集数据，将reward传入memory
            # 第五步，step中达到update_timestep就会进行update
            # 第六步，update中会利用上面收集的数据进行策略更新，并更新策略
            # 第七步，结束update后会清空memory，然后回到第一步再执行action = agent.act(state)并完成改变策略

            action = agent.act(state)    # 按照策略网络输出的概率随机采样一个动作
            next_state, reward, done, _, _ = env.step(action)  # 与环境state进行交互，输出reward 和 环境next_state
            state = next_state
            total_reward += reward
            total_step += 1
            rewards.append(reward)
            agent.step(reward, done)  # 传入的是真实奖励，done是环境给出的
            if done:  # 游戏结束
                final_rewards.append(reward)
                total_rewards.append(total_reward)
                break

    # avg_total_reward是一整局游戏下来的总得分
    # avg_final_reward是一局游戏最后一步的得分，100分意味着成功着陆

    if len(final_rewards)> 0 and len(total_rewards) > 0:
        avg_total_reward = sum(total_rewards) / len(total_rewards)
        avg_final_reward = sum(final_rewards) / len(final_rewards)
        avg_total_rewards.append(avg_total_reward)
        avg_final_rewards.append(avg_final_reward)

        # 这里可以打印出每个批次的平均奖励
        print(f"Batch {m}/{NUM_BATCH}, Average Total Reward: {avg_total_reward}, Average Final Reward: {avg_final_reward}")

plt.figure(figsize=(10, 5))
plt.plot(avg_total_rewards, color='blue')
plt.title("Total Rewards")
plt.xlabel("Batch")
plt.ylabel("Average Total Reward")
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(avg_final_rewards, color='blue')
plt.title("Final Rewards")
plt.xlabel("Batch")
plt.ylabel("Average Final Reward")
plt.show()