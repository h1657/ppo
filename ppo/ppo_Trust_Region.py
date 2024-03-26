import torch
import gym
import numpy as np
from torch.distributions import Categorical
import torch.optim as optim
import torch.nn as nn
import copy

class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

class ActorCriticDiscrete(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCriticDiscrete, self).__init__()

        # actor
        self.action_layer = nn.Sequential(
                nn.Linear(state_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, action_dim),
                nn.Softmax(dim=-1)
                )

        # critic
        self.value_layer = nn.Sequential(
               nn.Linear(state_dim, 128),
               nn.ReLU(),
               nn.Linear(128, 64),
               nn.ReLU(),
               nn.Linear(64, 1)
               )

    def act(self, state, memory):
        state = torch.from_numpy(state).float()

        action_probs = self.action_layer(state)

        dist = Categorical(action_probs)

        action = dist.sample()

        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(dist.log_prob(action))

        return action.item()

    def evaluate(self, state, action):
        action_probs = self.action_layer(state)
        dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)

        dist_entropy = dist.entropy()

        state_value = self.value_layer(state)

        return action_logprobs, torch.squeeze(state_value), dist_entropy

class PPOAgent:
    def __init__(self, state_dim, action_dim, lr, betas, gamma, K_epochs, eps_clip, kl_max, update_timestep):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.timestep = 0
        self.memory = Memory()
        self.update_timestep = update_timestep
        self.kl_max = kl_max

        self.policy = ActorCriticDiscrete(state_dim, action_dim)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)  # betas是Adam中的参数
        self.policy_old = ActorCriticDiscrete(state_dim, action_dim)  # 初始化
        self.policy_old.load_state_dict(self.policy.state_dict())

        # 计算函数的损失
        self.MseLoss = nn.MSELoss()

    def update(self):
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.memory.rewards), reversed(self.memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # 归一化
        rewards = torch.tensor(rewards, dtype=torch.float32)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        old_states = torch.stack(self.memory.states).detach()
        old_actions = torch.stack(self.memory.actions).detach()
        old_logprobs = torch.stack(self.memory.logprobs).detach()

        old_policy = copy.deepcopy(self.policy)

        for _ in range(self.K_epochs):
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            ratios = torch.exp(logprobs - old_logprobs.detach())

            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # 计算KL散度
            kl = torch.mean(old_logprobs - logprobs)

            if kl > self.kl_max:
                self.policy.load_state_dict(old_policy.state_dict())
                print("Exceeded KL max, rollback!")
                break

            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy + kl

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

            # 把当前策略的参数复制到旧策略
        self.policy_old.load_state_dict(self.policy.state_dict())

    def step(self, reward, done):
        self.timestep += 1

        self.memory.rewards.append(reward)
        self.memory.is_terminals.append(done)

        # update if its time
        if self.timestep % self.update_timestep == 0:
            self.update()
            self.memory.clear_memory()
            self.timestep = 0

    def act(self, state):
        return self.policy_old.act(state, self.memory)